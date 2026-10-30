"""Backend/resource and outcome policy for the existing instance supervisor."""
import os
import json
from pathlib import Path
import signal
import time
from scheduler_backends import DeviceBackend, DeviceLeases, host_resources
from scheduler_state import Ledger
from single_scheduler_helpers import read, save, sha, proc, live, attempt_processes


class ObservedProcess:
    """A restarted supervisor does not invent the old process's exit status."""
    def __init__(self, worker):
        self.worker = worker
        self.returncode = None

    def poll(self):
        return None

    def wait(self, timeout=10):
        deadline = time.monotonic()+timeout
        while live(self.worker):
            if time.monotonic() > deadline:
                raise RuntimeError('Adopted worker did not exit')
            time.sleep(.1)


class ExecutionPolicy:
    def __init__(self, meta, root, max_workers):
        self.meta = meta; self.root = root; self.rows = {r['module']: r for r in meta['tests']}
        self.backend = DeviceBackend(meta)
        first = self.backend.sample()
        if not first['devices']:
            raise RuntimeError('Cannot discover accelerator devices: '+repr(first['device_query']))
        directory = meta.get('device_lock_dir') or str(Path.home()/'.cache/api-test-relations/device-locks')
        self.leases = DeviceLeases(self.backend.ids, self.backend.kind, directory)
        self.ledger = Ledger(meta['database'])
        self.max_workers = max_workers if max_workers is not None else len(self.backend.ids)
        self.phase_order = meta.get('phase_order', ['regular', 'distributed'])
        if sorted(self.phase_order) != ['distributed', 'regular']:
            raise ValueError('phase_order must contain regular and distributed once')
        for name in ['execution-events', 'execution-xml', 'launches']:
            (root/name).mkdir(exist_ok=True)

    def resource(self):
        return {**host_resources(self.meta, self.root, self.meta['database']), **self.backend.sample()}

    def remaining(self, shard, raw_remaining):
        done = self.ledger.done(shard['module'], shard['nodeids'])
        return [node for node in raw_remaining if node not in done]

    def kind(self, shard):
        return 'distributed' if self.rows[shard['module']].get('resource') == 'all' else 'regular'

    def choose(self, states, shards, workers):
        # A whole phase drains before the other can acquire the same devices.
        # Ordinary launches otherwise fill every free card in one dispatch pass.
        for kind in self.phase_order:
            pending = [sid for sid, shard in shards.items() if states[sid] == 'pending' and self.kind(shard) == kind]
            running = [w for w in workers.values() if w.get('execution_kind') == kind]
            if not pending and not running:
                continue
            if kind == 'distributed' and workers:
                return None
            for sid in pending:
                module = shards[sid]['module']
                if self.rows[module].get('module_exclusive') and any(w['module'] == module for w in workers.values()):
                    continue
                return sid
            return None
        return None

    def acquire(self, attempt, shard, resource):
        minimum = self.meta.get('minimum_device_free_mib', 3072)
        eligible = [d['id'] for d in resource['devices'] if d['free_mib'] >= minimum]
        return self.leases.acquire(attempt, self.kind(shard) == 'distributed', eligible)

    def fields(self, shard, attempt, allowed, lease):
        return {'attempt': attempt, 'module': shard['module'], 'shard_id': shard['shard_id'],
                'allowed_nodeids': allowed, 'devices': lease['devices'],
                'events': str(self.root/'execution-events'/(attempt+'.jsonl')),
                'execution_kind': self.kind(shard), 'started_at': time.time(),
                'lease_inherited': True, 'allow_missing_selection': True,
                'short_temp_root': self.meta.get('short_temp_root')}

    def intent(self, worker):
        self.ledger.register(worker)
        save(self.root/'launches'/(worker['attempt']+'.json'), worker)

    def remember(self, worker):
        save(self.root/'launches'/(worker['attempt']+'.json'), worker)

    def inspect(self, worker):
        events = self.ledger.ingest(worker)
        worker['stack_ready'] = any(e['kind']=='session_start' and e.get('stack_ready') for e in events)
        done = self.ledger.done(worker['module'], worker['allowed_nodeids'])
        active = [e for e in events if e['kind'] == 'test_start' and e['nodeid'] not in done]
        now = time.time()
        if active:
            expired = now-active[-1]['at'] > self.meta['case_timeout_seconds']+self.meta.get('timeout_grace_seconds', 30)
            if expired:
                worker.update(timed_out=True, termination_reason='Per-case time budget exceeded: '+active[-1]['nodeid'])
        elif not any(e['kind'] == 'collection_finished' for e in events):
            expired = now-worker['started_at'] > self.meta.get('collection_timeout_seconds', 300)
            if expired:
                worker.update(timed_out=True, termination_reason='Collection/initialization time budget exceeded')
        else:
            expired = False
        if now-worker['started_at'] > self.meta['module_timeout_seconds']:
            expired = True
            worker.update(timed_out=True, termination_reason='Attempt total time budget exceeded')
        return expired

    def finish(self, worker, returncode):
        self.ledger.bindings(worker)
        events = self.ledger.ingest(worker)
        done = self.ledger.done(worker['module'], worker['allowed_nodeids'])
        active = [e['nodeid'] for e in events if e['kind']=='test_start' and e['nodeid'] not in done]
        if active:
            worker['interrupted'] = True
            worker['interrupted_nodes'] = sorted(set(worker.get('interrupted_nodes', [])) | {active[-1]})
        return self.ledger.finish(worker, returncode, interrupted=worker.get('interrupted', False))

    def release(self, worker):
        self.leases.release(worker['attempt'])

    def committed(self, worker):
        """Reconcile a crash after append committed but before its file receipt."""
        if not self.ledger.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='capture_runs'").fetchone():
            return None
        row = self.ledger.db.execute('SELECT detail FROM capture_runs WHERE attempt=?', (worker['output'],)).fetchone()
        if not row:
            return None
        receipt = json.loads(row[0])
        if receipt['module'] != worker['module'] or receipt['source'] != worker['output']:
            raise ValueError('Committed capture lineage mismatch')
        selection = receipt.get('selection') or {}
        if set(selection.get('selected', [])) | set(selection.get('missing', [])) != set(worker['allowed_nodeids']):
            raise ValueError('Committed capture selection mismatch')
        # A killed cleanup can leave a subset of original files. Any changed
        # or new content still blocks reconciliation; retained bytes are archived.
        for path in Path(worker['output']).rglob('*'):
            if path.is_file() and receipt['source_hashes'].get(str(path.relative_to(worker['output']))) != sha(path):
                raise ValueError('Capture changed after its committed append')
        return {'state': 'recovered_committed', 'checks': receipt['checks'],
                'relations': receipt['relations'], 'tests_touched': receipt['tests_touched']}

    def restore(self, root, shards, states, workers, handles):
        for path in sorted((root/'launches').glob('*.json')):
            worker = read(path); attempt = worker['attempt']
            if (root/'receipts'/(attempt+'.json')).exists():
                if read(root/'receipts'/(attempt+'.json')).get('state') == 'verification_blocked':
                    raise RuntimeError('Resolve the prior capture verification failure before resuming')
                continue
            sid = worker['shard_id']
            if sid not in shards:
                raise ValueError('Unknown prior instance shard')
            if not worker.get('pid'):
                matches = []
                marker = ('RELATION_CAPTURE_ATTEMPT='+worker['output']).encode()
                for directory in Path('/proc').glob('[0-9]*'):
                    try:
                        if marker in (directory/'environ').read_bytes().split(b'\0'):
                            item = proc(int(directory.name))
                            if item and item['command'] == worker['command']:
                                matches.append(item)
                    except OSError:
                        pass
                if len(matches) > 1:
                    raise RuntimeError('Ambiguous orphan worker identity')
                if matches:
                    worker.update(matches[0]); self.remember(worker)
                elif not Path(worker['events']).exists() and not Path(worker['output']).exists():
                    self.ledger.capture_state(attempt, 'launch_not_started')
                    save(root/'receipts'/(attempt+'.json'), {'state': 'launch_not_started', 'worker': worker})
                    continue
                else:
                    worker.update(pid=-1, start_ticks='unknown')
            if sid in workers:
                raise RuntimeError('Duplicate orphan shard')
            self.leases.adopt(attempt, worker['devices'])
            workers[sid] = worker; handles[sid] = ObservedProcess(worker); states[sid] = 'collecting'
