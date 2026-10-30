"""Independent instance workers, one verified writer; concurrency is a CLI policy."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import time
import sys

# Remain runnable when the deployment enables Python's safe-path mode.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from instance_sharding import (positive_workers, validate_plan, remaining_nodes,
                               validate_capture, retry_state, dispatch_allowed)
from single_scheduler_helpers import (read, save, sha, proc, live, launch_spec, resources,
                                     attempt_processes, terminate_attempt, backend_exclusion_reason)
from api_test_relations.single_store import completed, append_capture
from scheduler_policy import ExecutionPolicy


def connect(path):
    return sqlite3.connect(Path(path).as_uri()+'?mode=ro', uri=True, timeout=3)


def counts(database):
    with connect(database) as db:
        return {key: db.execute(sql).fetchone()[0] for key, sql in {
            'complete_tests': "SELECT count(*) FROM tests WHERE capture_state='complete'",
            'tests': 'SELECT count(*) FROM tests', 'observations': 'SELECT count(*) FROM observations',
            'capture_gaps': 'SELECT count(*) FROM capture_gaps'}.items()}


def entered_nodes(worker):
    result = set()
    for path in Path(worker['shards']).glob('*.sqlite'):
        with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True, timeout=3) as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'meta' in tables and dict(db.execute('SELECT key,value FROM meta')).get('role') == 'child':
                continue
            if 'phases' in tables:
                result.update(n for n, in db.execute("SELECT nodeid FROM phases WHERE state!='closed'"))
    return sorted(result)


def preserve_attempt(root, worker):
    if attempt_processes(worker):
        raise RuntimeError('Cannot archive a live attempt')
    source = Path(worker['output']); archive = root/'preserved'/worker['attempt']
    original = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
    if archive.exists():
        prior = read(archive/'provenance.json')
        copied = {str(p.relative_to(archive/'capture')): sha(p) for p in (archive/'capture').rglob('*') if p.is_file()}
        if (prior['source'] != str(source) or copied != prior['source_hashes']
                or any(prior['source_hashes'].get(name) != digest for name,digest in original.items())):
            raise RuntimeError('Prior raw archive or remaining source changed')
        return str(archive/'capture')
    archive.mkdir()
    shutil.copytree(source, archive/'capture')
    for path in (archive/'capture').rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    copied = {str(p.relative_to(archive/'capture')): sha(p) for p in (archive/'capture').rglob('*') if p.is_file()}
    if copied != original:
        raise RuntimeError('Raw archive hash mismatch')
    save(archive/'provenance.json', {'source': str(source), 'source_hashes': original,
         'worker': worker, 'log_retained': worker['log'], 'scratch_retained': True})
    return str(archive/'capture')


def attempt_pss_mib(worker):
    total = 0
    for item in attempt_processes(worker).values():
        try:
            fields = Path('/proc', str(item['pid']), 'smaps_rollup').read_text().splitlines()
            total += int(next(line.split()[1] for line in fields if line.startswith('Pss:')))
        except (OSError, StopIteration):
            continue
    return (total+1023)//1024


def resource_sample(database):
    result = resources()
    result['host_c_free_gib'] = shutil.disk_usage(Path(database).parent).free/2**30
    result['database_volume_free_gib'] = shutil.disk_usage(Path(database).parent).free/2**30
    return result


def run(root, max_workers=None, resume=False):
    max_workers = positive_workers(max_workers) if max_workers is not None else None
    meta = read(root/'manifest.json'); database = meta['database']
    if (root/'supervisor.json').exists() and not resume:
        raise RuntimeError('Reconcile a prior start explicitly; never duplicate workers')
    if resume and (root/'supervisor.json').exists() and live(read(root/'supervisor.json')):
        raise RuntimeError('The prior supervisor is still running')
    with (root/'supervisor.lock').open('a') as own_lock, Path(database+'.scheduler.lock').open('a') as shared_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(shared_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        for path, digest in meta['frozen_files'].items():
            if sha(path) != digest:
                raise RuntimeError('Frozen source changed: '+path)
        if meta.get('commit') is not None and subprocess.check_output(['git','rev-parse','HEAD'],cwd=meta['repo'],text=True).strip() != meta['commit']:
            raise RuntimeError('PyTorch source changed')
        rows = {row['module']:row for row in meta['tests']}
        plan = read(root/'instance-plan.json')
        validate_plan(plan, rows, lambda row:backend_exclusion_reason(meta,row))
        shards = {s['shard_id']:s for s in plan['shards']}
        states = {sid:'pending' for sid in shards}; workers = {}; handles = {}
        policy = ExecutionPolicy(meta, root, max_workers) if meta.get('execution_results') else None
        if policy:
            max_workers = policy.max_workers
        elif max_workers is None:
            raise ValueError('Legacy capture plans require --max-workers')
        if resume:
            if policy is None:
                raise ValueError('Explicit instance-worker adoption requires an execution ledger')
            policy.restore(root, shards, states, workers, handles)
        queue = read(root/'queue.json'); limit = max_workers; stopped = False; last_launch = 0; last_publish = 0
        aggregate = counts(database)
        def stop(*_):
            nonlocal stopped
            stopped = True
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        save(root/'supervisor.json', {**proc(os.getpid()), 'started_at':time.time(),
             'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'max_workers':max_workers})
        try:
            while True:
                now = time.time(); resource = policy.resource() if policy else resource_sample(database); changed = False
                memory_victim = max(workers, key=lambda sid:workers[sid]['started_at']) if workers and resource['memory_available_mib'] < 2048 else None
                for sid, worker in list(workers.items()):
                    handle = handles[sid]; handle.poll()
                    expired = now-worker['started_at'] >= meta['module_timeout_seconds']
                    if policy:
                        expired = policy.inspect(worker) or expired
                    low_disk = min(resource['host_c_free_gib'], resource['database_volume_free_gib']) < 2
                    resource_stop = sid == memory_victim or low_disk
                    if (stopped or expired or low_disk or resource_stop) and live(worker):
                        if stopped:
                            worker['pause_requested'] = True
                        if resource_stop:
                            worker['resource_stop'] = True
                        if policy and worker.get('stack_ready'):
                            try: os.kill(worker['pid'], signal.SIGUSR1)
                            except ProcessLookupError: pass
                        worker['terminated_pids'] = terminate_attempt(worker)
                        worker['interrupted'] = True
                        # Observe parent phases after every writer has committed and stopped.
                        worker['interrupted_nodes'] = entered_nodes(worker)
                        handle.wait(timeout=10)
                        if resource_stop:
                            limit = max(1, len(workers)-1)
                            worker['resource_stop'] = True
                            if len(workers) == 1:
                                stopped = True
                    if live(worker):
                        continue
                    worker.setdefault('parent_exited_at', now)
                    remaining_processes = attempt_processes(worker)
                    child_wait = min(120, max(0, meta['module_timeout_seconds']-(now-worker['started_at'])))
                    if remaining_processes and not worker.get('interrupted') and now-worker['parent_exited_at'] < child_wait:
                        continue
                    if remaining_processes:
                        worker['terminated_pids'] = terminate_attempt(worker); worker['interrupted'] = True
                        worker['interrupted_nodes'] = entered_nodes(worker)
                    try:
                        if policy:
                            policy.finish(worker, handle.returncode)
                        if worker.get('checkpoint_barrier',{}).get('missing') and not policy:
                            raise RuntimeError('Checkpoint writer acknowledgement missing')
                        selection_exists = (Path(worker['output'])/'resume-selection.json').exists()
                        collection_failed = policy and any(e['kind']=='collection_error' for e in policy.ledger.events(worker))
                        committed = policy.committed(worker) if policy else None
                        if committed:
                            archive = preserve_attempt(root, worker) if Path(worker['output']).exists() else None
                            result = committed
                        elif policy and (not selection_exists or collection_failed):
                            events = policy.ledger.events(worker)
                            if not selection_exists and any(e['kind'] in ('test_start', 'report') for e in events):
                                raise ValueError('Missing strict selection receipt after test execution')
                            archive = preserve_attempt(root, worker) if Path(worker['output']).exists() else None
                            result = {'state':'capture_unavailable', 'raw_preserved':True}
                            with sqlite3.connect(database) as db:
                                mid = db.execute('SELECT id FROM modules WHERE module=?',(worker['module'],)).fetchone()[0]
                                db.execute('INSERT INTO capture_gaps VALUES (?,?,?,?)',
                                           (mid,None,'collection','scheduler_collection_unavailable:'+worker['attempt']))
                        else:
                            validate_capture(worker)
                            archive = preserve_attempt(root, worker)
                            # Keep all original append ownership/subset/integrity gates.
                            extra = {}
                            if policy:
                                extra['boundary_gaps'] = ['checkpoint_barrier_missing:'+str(pid)
                                    for pid in worker.get('checkpoint_barrier',{}).get('missing',[])]
                            result = append_capture(Path(worker['output']), Path(database), worker['module'],
                                interrupted=worker.get('interrupted',False),
                                interrupted_nodes=worker.get('interrupted_nodes',()), **extra)
                        with connect(database) as db:
                            after = remaining_nodes(db, worker['module'], shards[sid]['nodeids'])
                        if policy:
                            after = policy.remaining(shards[sid], after)
                            policy.ledger.capture_state(worker['attempt'], result['state'])
                            states[sid] = 'pending' if after else 'attempted'
                        else:
                            states[sid] = retry_state(worker['allowed_nodeids'], after, worker.get('resource_stop',False))
                        if states[sid] == 'continuation_needs_diagnosis':
                            stopped = True
                        save(root/'receipts'/(worker['attempt']+'.json'), {**result, 'module':worker['module'],
                             'shard_id':sid, 'worker':worker, 'remaining':after, 'queue_state':states[sid],
                             'archive':archive, 'finished_at':time.time()})
                        aggregate = counts(database)
                    except Exception as exc:
                        states[sid] = 'verification_blocked'; stopped = True
                        save(root/'receipts'/(worker['attempt']+'.json'), {'module':worker['module'],
                             'shard_id':sid, 'state':'verification_blocked', 'error':repr(exc),
                             'worker':worker, 'raw_preserved':True, 'at':time.time()})
                        if policy:
                            policy.ledger.capture_state(worker['attempt'], 'verification_blocked')
                    if policy:
                        policy.release(worker)
                    del workers[sid]; del handles[sid]; changed = True
                active_pss = [attempt_pss_mib(worker) for worker in workers.values()]
                resource['estimated_worker_start_mib'] = max(active_pss, default=0)
                interval = meta.get('dispatch_interval_seconds', 0 if policy else 30)
                while not stopped and dispatch_allowed(resource, len(workers), max_workers, limit,
                                                      30 if time.time()-last_launch>=interval else 0, resource['estimated_worker_start_mib']):
                    sid = policy.choose(states, shards, workers) if policy else next((sid for sid in shards if states[sid]=='pending'), None)
                    if sid is None:
                        break
                    if sid is not None:
                        shard = shards[sid]; module = shard['module']; row = rows[module]
                        if backend_exclusion_reason(meta,row) or sha(row['source']) != row['sha256']:
                            raise RuntimeError('Excluded backend or changed test source')
                        with connect(database) as db:
                            allowed = remaining_nodes(db,module,shard['nodeids']); retained = sorted(completed(db,module))
                        if policy:
                            allowed = policy.remaining(shard, allowed)
                        if not allowed:
                            states[sid] = 'captured'; changed = True
                        else:
                            attempt = sid+'-'+str(time.time_ns())
                            lease = policy.acquire(attempt, shard, resource) if policy else None
                            if policy and lease is None:
                                break
                            save(root/'selections'/(attempt+'.json'), {'job':module, 'shard_id':sid,
                                 'retained':retained, 'allowed':allowed})
                            cmd, env, output = launch_spec(meta,root,row,attempt, devices=lease['devices']) if policy else launch_spec(meta,root,row,attempt)
                            cmd += ['-p','no:cacheprovider']; env['PYTHONDONTWRITEBYTECODE']='1'
                            log = root/'logs'/(attempt+'.log')
                            extra = {}
                            if policy:
                                extra = policy.fields(shard, attempt, allowed, lease)
                                policy.intent({**extra, 'command':cmd, 'output':str(output),
                                               'shards':str(output/'shards'), 'log':str(log)})
                            with log.open('wb') as stream:
                                child = subprocess.Popen(cmd,cwd=meta.get('worker_cwd',meta['repo']),env=env,stdout=stream,
                                                         stderr=subprocess.STDOUT,start_new_session=True,
                                                         pass_fds=tuple(lease['fds']) if policy else ())
                            actual = proc(child.pid)
                            if not actual:
                                raise RuntimeError('Worker exited before identity tracking')
                            worker = {**actual, 'command':cmd, 'module':module, 'file':row['file'],
                                      'shard_id':sid, 'attempt':attempt, 'allowed_nodeids':allowed,
                                      'output':str(output), 'shards':str(output/'shards'), 'log':str(log),
                                      'started_at':time.time(), 'excluded_complete_instances':len(retained),
                                      'temp_alias':env.get('RELATION_TEMP_ALIAS'), **extra}
                            if policy:
                                policy.remember(worker)
                            workers[sid]=worker; handles[sid]=child; states[sid]='collecting'
                            last_launch=time.time(); changed=True
                            if policy:
                                resource['memory_available_mib'] -= max(meta.get('worker_start_reserve_mib',2048), resource['estimated_worker_start_mib'])
                    if not policy:
                        break
                for module in plan['remaining']:
                    module_states = [states[sid] for sid,s in shards.items() if s['module']==module]
                    queue[module] = ('verification_blocked' if 'verification_blocked' in module_states else
                        'continuation_needs_diagnosis' if 'continuation_needs_diagnosis' in module_states else
                        'collecting' if 'collecting' in module_states else 'pending' if 'pending' in module_states else 'attempted' if policy else 'captured')
                finished = not workers and (stopped or 'pending' not in states.values())
                if changed or finished or time.time()-last_publish >= 15:
                    progress = {'state':'paused_for_review' if stopped else 'collection_finished' if finished else
                                'collecting' if workers else 'waiting_resources', 'updated_at':time.time(),
                        'worker_limit':max_workers, 'effective_resource_worker_limit':limit,
                        'active_workers':len(workers), 'current_modules':[x['module'] for x in workers.values()],
                        'current_shards':list(workers), 'instance_shard_states':states,
                        'queue':dict(__import__('collections').Counter(queue.values())), 'resources':resource,
                        'counts':aggregate, 'counts_exclude_inflight':True, 'database':database}
                    if policy:
                        progress.update(execution=policy.ledger.summary(), devices=policy.backend.ids,
                                        device_leases={key:val['devices'] for key,val in policy.leases.owned.items()})
                    save(root/'queue.json',queue); save(root/'instance-queue.json',states)
                    for target in dict.fromkeys([root,Path(meta['publish_root'])]):
                        save(target/'progress.json',progress)
                        save(target/'current-processes.json',{'updated_at':time.time(),'workers':list(workers.values())})
                    last_publish=time.time()
                if finished:
                    break
                time.sleep(meta.get('poll_interval_seconds', 2))
        except BaseException as exc:
            emergency = []
            for worker in workers.values():
                try:
                    worker['terminated_pids'] = terminate_attempt(worker)
                    emergency.append(worker)
                except Exception as stop_error:
                    emergency.append({'worker':worker, 'stop_error':repr(stop_error)})
            save(root/'scheduler-error.json', {'reason':repr(exc),'workers':emergency,'at':time.time()})
            raise
        finally:
            if policy:
                for attempt in list(policy.leases.owned):
                    policy.leases.release(attempt)
                policy.ledger.db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-workers','--max',type=positive_workers,
                        help='Concurrency ceiling; portable plans default to all discovered devices')
    parser.add_argument('--resume',action='store_true',help='Reconcile and observe prior workers without duplicating them')
    args = parser.parse_args()
    run(args.out.resolve(),args.max_workers,args.resume)


if __name__ == '__main__':
    main()
