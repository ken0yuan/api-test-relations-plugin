"""Single-writer execution ledger; capture completeness is never a test verdict."""
import ast
from collections import Counter
import json
from pathlib import Path
import sqlite3
import time


BASE_SCHEMA = '''
CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
CREATE TABLE apis(id INTEGER PRIMARY KEY,api_id TEXT UNIQUE,name TEXT,language TEXT,kind TEXT);
CREATE TABLE modules(id INTEGER PRIMARY KEY,module TEXT UNIQUE,file TEXT,capture_state TEXT);
CREATE TABLE tests(id INTEGER PRIMARY KEY,module_id INTEGER REFERENCES modules(id),nodeid TEXT,file TEXT,capture_state TEXT,source_version TEXT,source_attempt TEXT,UNIQUE(module_id,nodeid));
CREATE TABLE phases(id INTEGER PRIMARY KEY,name TEXT UNIQUE);
CREATE TABLE alias_groups(id INTEGER PRIMARY KEY,api_ids_json TEXT UNIQUE);
CREATE TABLE observations(api_key INTEGER REFERENCES apis(id),test_key INTEGER REFERENCES tests(id),phase_key INTEGER REFERENCES phases(id),alias_group_key INTEGER REFERENCES alias_groups(id),PRIMARY KEY(api_key,test_key,phase_key,alias_group_key));
CREATE INDEX observations_by_test ON observations(test_key);
CREATE TABLE test_phases(test_id INTEGER REFERENCES tests(id),phase_id INTEGER REFERENCES phases(id),state TEXT,PRIMARY KEY(test_id,phase_id));
CREATE TABLE capture_gaps(module_id INTEGER REFERENCES modules(id),test_id INTEGER REFERENCES tests(id),phase TEXT,reason TEXT);
'''


def create_database(path, catalog, rows):
    if Path(path).exists():
        raise ValueError('Initial database must not exist')
    with sqlite3.connect(path) as db:
        db.executescript(BASE_SCHEMA)
        db.execute('INSERT INTO metadata VALUES (?,?)', ('schema', json.dumps('api_test_relations_single_database/2')))
        db.executemany('INSERT INTO apis(api_id,name,language,kind) VALUES (?,?,?,?)',
                       [(a['api_id'], a['qualified_name'], a['language'], a['kind']) for a in catalog['apis']])
        for row in rows:
            mid = db.execute('INSERT INTO modules(module,file,capture_state) VALUES (?,?,?)',
                             (row['module'], row['file'], 'pending')).lastrowid
            db.executemany('INSERT INTO tests(module_id,nodeid,file,capture_state) VALUES (?,?,?,?)',
                           [(mid, node, row['file'], 'unobserved') for node in row['nodeids']])
    with Ledger(path):
        pass


def verdict(reports):
    failure = next((r for r in reports if r['outcome'] == 'failed'), None)
    if failure:
        if 'Timeout (' in failure.get('detail', '') and 'from pytest-timeout' in failure.get('detail', ''):
            return 'timeout', failure['detail']
        status = {'setup': 'setup_error', 'teardown': 'teardown_error'}.get(failure['phase'], 'failed')
        return status, failure.get('detail', '')
    skipped = next((r for r in reports if r['outcome'] == 'skipped'), None)
    if skipped:
        detail = skipped.get('detail', '')
        if skipped.get('wasxfail') or ('addExpectedFailure' in detail and '_pytest.outcomes.XFailed' in detail):
            return 'expected_failure', detail
        try:
            value = ast.literal_eval(detail)
            if isinstance(value, tuple) and len(value) == 3:
                detail = str(value[2])
        except (ValueError, SyntaxError):
            pass
        return 'skipped', detail
    if any(r['phase'] == 'call' and r.get('wasxfail') for r in reports):
        return 'unexpected_pass', ''
    if any(r['phase'] == 'call' for r in reports):
        return 'passed', ''
    return 'result_unavailable', 'No call-phase result was recorded'


class Ledger:
    def __init__(self, database):
        self.db = sqlite3.connect(database, timeout=60)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
CREATE TABLE IF NOT EXISTS execution_attempts(attempt TEXT PRIMARY KEY,module TEXT,allowed TEXT,devices TEXT,started REAL,finished REAL,returncode INTEGER,capture_state TEXT);
CREATE TABLE IF NOT EXISTS execution_events(attempt TEXT,sequence INTEGER,data TEXT,PRIMARY KEY(attempt,sequence));
CREATE TABLE IF NOT EXISTS execution_results(module TEXT,nodeid TEXT,status TEXT,reason TEXT,duration REAL,attempt TEXT,test_started INTEGER,call_reported INTEGER,finished REAL,PRIMARY KEY(module,nodeid));
CREATE TABLE IF NOT EXISTS api_binding_states(api_id TEXT REFERENCES apis(api_id),state TEXT,attempt TEXT,PRIMARY KEY(api_id,state));
''')

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.db.close()

    def register(self, worker):
        self.db.execute('INSERT INTO execution_attempts VALUES (?,?,?,?,?,NULL,NULL,?)',
                        (worker['attempt'], worker['module'], json.dumps(worker['allowed_nodeids']),
                         json.dumps(worker['devices']), worker['started_at'], 'running'))
        self.db.commit()

    def events(self, worker):
        path = Path(worker['events'])
        rows = []
        if path.exists():
            for line in path.read_bytes().split(b'\n')[:-1]:
                if not line:
                    continue
                rows.append(json.loads(line.decode('utf-8')))
        allowed = set(worker['allowed_nodeids'])
        for index, row in enumerate(rows):
            if row.get('kind') in ('test_start', 'report', 'selection_missing') and row.get('nodeid') not in allowed:
                raise ValueError('Execution result escaped assigned shard')
            encoded = json.dumps(row, ensure_ascii=False)
            prior = self.db.execute('SELECT data FROM execution_events WHERE attempt=? AND sequence=?',
                                    (worker['attempt'], index)).fetchone()
            if prior and prior[0] != encoded:
                raise ValueError('Durable execution event history changed')
            self.db.execute('INSERT OR IGNORE INTO execution_events VALUES (?,?,?)',
                            (worker['attempt'], index, encoded))
        self.db.commit()
        return rows

    def done(self, module, nodes=None):
        if nodes is None:
            return {r[0] for r in self.db.execute('SELECT nodeid FROM execution_results WHERE module=?', (module,))}
        result = set()
        for start in range(0, len(nodes), 500):
            part = nodes[start:start+500]
            result.update(r[0] for r in self.db.execute(
                'SELECT nodeid FROM execution_results WHERE module=? AND nodeid IN ('+','.join('?' for _ in part)+')',
                (module, *part)))
        return result

    def record(self, worker, node, status, reason, phases=(), started=False):
        prior = self.db.execute('SELECT attempt,status FROM execution_results WHERE module=? AND nodeid=?',
                                (worker['module'], node)).fetchone()
        if prior and prior != (worker['attempt'], status):
            raise ValueError('Completed execution result cannot be replaced')
        self.db.execute('INSERT OR IGNORE INTO execution_results VALUES (?,?,?,?,?,?,?,?,?)',
                        (worker['module'], node, status, reason, sum(r.get('duration', 0) for r in phases),
                         worker['attempt'], int(started), int(any(r.get('phase') == 'call' for r in phases)), time.time()))

    def ingest(self, worker):
        events = self.events(worker); reports = {}; started = set()
        for event in events:
            if event['kind'] == 'selection_missing':
                self.record(worker, event['nodeid'], 'not_collected', event['detail'], started=False)
            elif event['kind'] == 'test_start':
                started.add(event['nodeid'])
            elif event['kind'] == 'report':
                reports.setdefault(event['nodeid'], []).append(event)
        for node, phases in reports.items():
            if any(r['phase'] == 'teardown' for r in phases):
                self.record(worker, node, *verdict(phases), phases, node in started)
        self.db.commit()
        return events

    def finish(self, worker, returncode, interrupted=False):
        events = self.ingest(worker); done = self.done(worker['module'], worker['allowed_nodeids'])
        remaining = [n for n in worker['allowed_nodeids'] if n not in done]
        started = [e['nodeid'] for e in events if e['kind'] == 'test_start' and e['nodeid'] in remaining]
        errors = [e for e in events if e['kind'] in ('collection_error', 'module_skipped')]
        if worker.get('resource_stop') or worker.get('pause_requested'):
            pass
        elif started:
            node = started[-1]
            reason = worker.get('termination_reason') or f'Worker exited before final result; exit status={returncode}'
            self.record(worker, node, 'timeout' if worker.get('timed_out') else 'crashed', reason, started=True)
        elif remaining and (errors or not any(e['kind']=='test_start' for e in events)):
            status = ('module_skipped' if errors and all(e['kind'] == 'module_skipped' for e in errors)
                      else 'collection_error' if errors else 'worker_error' if any(e['kind'] == 'collection_finished' for e in events) else 'collection_error')
            reason = '\n'.join(e.get('detail', '') for e in errors) or worker.get('termination_reason') or 'Worker exited before any assigned test started; see attempt log'
            # A file-level collection/initialization failure is explicitly not
            # a test-body attempt or a successful skip.
            for node in remaining:
                self.record(worker, node, status, reason, started=False)
        self.db.execute('UPDATE execution_attempts SET finished=COALESCE(finished,?),returncode=COALESCE(returncode,?),capture_state=? WHERE attempt=?',
                        (time.time(), returncode, 'interrupted' if interrupted else 'exited', worker['attempt']))
        self.db.commit()
        done = self.done(worker['module'], worker['allowed_nodeids'])
        return [n for n in worker['allowed_nodeids'] if n not in done]

    def capture_state(self, attempt, state):
        self.db.execute('UPDATE execution_attempts SET capture_state=? WHERE attempt=?', (state, attempt))
        self.db.commit()

    def bindings(self, worker):
        for path in Path(worker['shards']).glob('*.sqlite'):
            with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as raw:
                if not raw.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='registry'").fetchone():continue
                for aid,state in raw.execute('SELECT api_id,state FROM registry'):
                    if not self.db.execute('SELECT 1 FROM apis WHERE api_id=?',(aid,)).fetchone():
                        raise ValueError('API binding escaped the frozen catalog')
                    self.db.execute('INSERT OR IGNORE INTO api_binding_states VALUES (?,?,?)',(aid,state,worker['attempt']))
        self.db.commit()

    def summary(self):
        total = self.db.execute('SELECT count(*) FROM tests').fetchone()[0]
        counts = dict(self.db.execute('SELECT status,count(*) FROM execution_results GROUP BY status'))
        return {'planned': total, 'terminal_results': sum(counts.values()),
                'pending': total-sum(counts.values()), 'outcomes': counts,
                'tests_started': self.db.execute('SELECT count(*) FROM execution_results WHERE test_started=1').fetchone()[0]}
