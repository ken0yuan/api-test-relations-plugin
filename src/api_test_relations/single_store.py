"""Append verified captures to the single database without copying the baseline."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time
from pathlib import PurePosixPath

from .cli import read
from .retention import _idle, _inventory
from .storage import connect, merge


def upgrade_observation_schema(db):
    """Transactional /1 -> /2 upgrade; caller owns the write transaction."""
    schema=json.loads(db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()[0])
    if schema=='api_test_relations_single_database/1':
        db.execute('''CREATE TABLE observations_v2(
            api_key INTEGER REFERENCES apis(id),test_key INTEGER REFERENCES tests(id),
            phase_key INTEGER REFERENCES phases(id),alias_group_key INTEGER REFERENCES alias_groups(id),
            PRIMARY KEY(api_key,test_key,phase_key,alias_group_key))''')
        db.execute('INSERT INTO observations_v2 SELECT * FROM observations')
        db.execute('DROP TABLE observations')
        db.execute('ALTER TABLE observations_v2 RENAME TO observations')
        db.execute("UPDATE metadata SET value=? WHERE key='schema'",(json.dumps('api_test_relations_single_database/2'),))
    elif schema!='api_test_relations_single_database/2':
        raise ValueError('Unsupported destination schema')
    db.execute('''CREATE TABLE IF NOT EXISTS observation_evidence(
        api_key INTEGER,test_key INTEGER,phase_key INTEGER,alias_group_key INTEGER,
        site TEXT,source_capture TEXT,source_shard TEXT,
        PRIMARY KEY(api_key,test_key,phase_key,alias_group_key,site,source_capture,source_shard),
        FOREIGN KEY(api_key,test_key,phase_key,alias_group_key)
            REFERENCES observations(api_key,test_key,phase_key,alias_group_key))''')


def completed(db, module):
    return {r[0] for r in db.execute('''SELECT t.nodeid FROM tests t JOIN modules m ON m.id=t.module_id
        WHERE m.module=? AND t.capture_state='complete' ''', (module,))}


def observations(db, test_id):
    return set(db.execute('''SELECT a.api_id,p.name,g.api_ids_json FROM observations o
        JOIN apis a ON a.id=o.api_key JOIN phases p ON p.id=o.phase_key
        JOIN alias_groups g ON g.id=o.alias_group_key WHERE o.test_key=?''', (test_id,)))


def _is_module_collection_gap(node, phase, module_file):
    """Accept a file-scoped collection gap only for this module's source file."""
    if phase != 'collection' or not node or not module_file:
        return False
    def normalized(path):
        value = PurePosixPath(str(path).replace('\\', '/'))
        parts = value.parts[1:] if value.parts[:1] == ('pytorch',) else value.parts
        return PurePosixPath(*parts)
    return normalized(node) == normalized(module_file)


def prepare_merge_indexes(merged):
    """Index disposable merge rows by owner without changing evidence or SQL.

    The existing primary keys start with job. Import checks also query by nodeid
    alone; without these indexes each test scans the entire module repeatedly.
    Do not use executescript here: the caller controls the transaction boundary.
    """
    indexes = ('merge_relations_by_nodeid', 'merge_phases_by_nodeid')
    merged.execute('CREATE INDEX IF NOT EXISTS '+indexes[0]+' ON relations(nodeid)')
    merged.execute('CREATE INDEX IF NOT EXISTS '+indexes[1]+' ON phases(nodeid)')
    return indexes


def verify_raw(db, source, retained):
    for file in sorted((source / 'shards').glob('*.sqlite')):
        with sqlite3.connect(file.as_uri() + '?mode=ro', uri=True) as raw:
            if raw.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
                raise ValueError('Invalid raw SQLite: ' + file.name)
            tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'meta' not in tables:
                if any(raw.execute('SELECT 1 FROM ' + t + ' LIMIT 1').fetchone()
                       for t in ('tests', 'relations', 'phases') if t in tables):
                    raise ValueError('Nonempty shard without ownership metadata')
                continue
            child = dict(raw.execute('SELECT key,value FROM meta')).get('role') == 'child'
            for table in ('tests', 'phases', 'relations', 'gaps'):
                if table not in tables or (table == 'phases' and child):
                    continue
                columns = [r[1] for r in raw.execute('PRAGMA table_info(' + table + ')')]
                for row in raw.execute('SELECT ' + ','.join(columns) + ' FROM ' + table):
                    node = row[0]
                    if table in ('phases', 'relations') and node in retained:
                        raise ValueError('Completed test was executed again: ' + node)
                    if table == 'tests' and (not node.startswith('pytorch/') or row[1] != node.split('::')[0]):
                        raise ValueError('Invalid test/source identity')
                    if table == 'relations':
                        aliases = json.loads(row[3])
                        if not isinstance(aliases, list) or row[2] not in aliases or any(
                            not db.execute('SELECT 1 FROM apis WHERE api_id=?', (aid,)).fetchone() for aid in aliases):
                            raise ValueError('Invalid API alias group')
                    where = ' AND '.join(c + ' IS ?' for c in columns)
                    if not db.execute('SELECT 1 FROM ' + table + ' WHERE job=? AND ' + where, ('pytest', *row)).fetchone():
                        raise ValueError('Raw merge conflict: ' + table)
                    if table=='relations' and not db.execute(
                        'SELECT 1 FROM relation_sources WHERE job=? AND '+where+' AND shard=?',
                        ('pytest',*row,file.name)).fetchone():
                        raise ValueError('Raw evidence source missing')


def import_merged(db, merged, module, attempt, version, receipt, *, allow_reexecution_of=()):
    """Caller holds BEGIN IMMEDIATE; failure rolls back the whole module."""
    upgrade_observation_schema(db)
    module_id = db.execute('SELECT id FROM modules WHERE module=?', (module,)).fetchone()[0]
    module_file = db.execute('SELECT file FROM modules WHERE id=?', (module_id,)).fetchone()[0]
    retained = completed(db, module)
    raw_tests = dict(merged.execute('SELECT nodeid,file FROM tests'))
    phase_rows = list(merged.execute('SELECT nodeid,phase,state FROM phases'))
    touched = {r[0] for r in phase_rows} | {r[0] for r in merged.execute('SELECT DISTINCT nodeid FROM relations')}
    allowed=set(allow_reexecution_of)
    if not allowed<=touched:raise ValueError('Authorized reexecution is not present in this capture')
    if (touched & retained)-allowed: raise ValueError('Completed instance cannot be replaced')
    prior_positive={};history={}
    for node in allowed:
        row=db.execute('SELECT * FROM tests WHERE module_id=? AND nodeid=?',(module_id,node)).fetchone()
        if row:
            prior_positive[node]=observations(db,row[0])
            history[node]={'test_record':list(row),'observations':[list(r) for r in sorted(prior_positive[node])],
                           'phases':[list(r) for r in db.execute('SELECT p.name,tp.state FROM test_phases tp JOIN phases p ON p.id=tp.phase_id WHERE tp.test_id=? ORDER BY p.name',(row[0],))]}
    api_ids = dict(db.execute('SELECT api_id,id FROM apis'))
    for node, file in raw_tests.items():
        prior = db.execute('SELECT id,file,capture_state FROM tests WHERE module_id=? AND nodeid=?', (module_id, node)).fetchone()
        if prior and prior[1] != file: raise ValueError('Source path changed for existing test')
        if not prior:
            db.execute('INSERT INTO tests(module_id,nodeid,file,capture_state,source_version,source_attempt) VALUES (?,?,?,?,?,?)',
                       (module_id, node, file, 'unobserved', version, attempt))
    ids = dict(db.execute('SELECT nodeid,id FROM tests WHERE module_id=?', (module_id,)))
    for node in touched:
        fresh = set(merged.execute('SELECT api_id,phase,aliases FROM relations WHERE nodeid=?', (node,)))
        # No silent loss of observations saved during earlier interrupted attempts.
        if node not in allowed and not observations(db, ids[node]) <= fresh:
            raise ValueError('New attempt does not contain all prior interrupted observations: ' + node)
        # Keep earlier positive evidence across interrupted attempts. The subset
        # check above forbids shrinking it; new groups/sites are appended below.
        db.execute('DELETE FROM test_phases WHERE test_id=?', (ids[node],))
    for node, phase, state in phase_rows:
        db.execute('INSERT OR IGNORE INTO phases(name) VALUES (?)', (phase,))
        pid = db.execute('SELECT id FROM phases WHERE name=?', (phase,)).fetchone()[0]
        db.execute('INSERT INTO test_phases VALUES (?,?,?)', (ids[node], pid, state))
    for node, phase, api, aliases in merged.execute('SELECT nodeid,phase,api_id,aliases FROM relations'):
        db.execute('INSERT OR IGNORE INTO phases(name) VALUES (?)', (phase,))
        db.execute('INSERT OR IGNORE INTO alias_groups(api_ids_json) VALUES (?)', (aliases,))
        pid = db.execute('SELECT id FROM phases WHERE name=?', (phase,)).fetchone()[0]
        gid = db.execute('SELECT id FROM alias_groups WHERE api_ids_json=?', (aliases,)).fetchone()[0]
        db.execute('INSERT OR IGNORE INTO observations VALUES (?,?,?,?)', (api_ids[api], ids[node], pid, gid))
    evidence=[]
    for node,phase,api,aliases,site,shard in merged.execute('SELECT nodeid,phase,api_id,aliases,site,shard FROM relation_sources'):
        pid=db.execute('SELECT id FROM phases WHERE name=?',(phase,)).fetchone()[0]
        gid=db.execute('SELECT id FROM alias_groups WHERE api_ids_json=?',(aliases,)).fetchone()[0]
        evidence.append((api_ids[api],ids[node],pid,gid,site,receipt.get('source',attempt),shard))
    db.executemany('INSERT OR IGNORE INTO observation_evidence VALUES (?,?,?,?,?,?,?)',evidence)
    for row in evidence:
        if not db.execute('SELECT 1 FROM observation_evidence WHERE '+ ' AND '.join(c+' IS ?' for c in
            ('api_key','test_key','phase_key','alias_group_key','site','source_capture','source_shard')),row).fetchone():
            raise ValueError('Destination evidence mismatch')
    for node in touched:
        stages = dict(merged.execute('SELECT phase,state FROM phases WHERE nodeid=?', (node,)))
        complete = stages.get('setup') == 'closed' and stages.get('teardown') == 'closed' and stages.get('call', 'closed') == 'closed'
        if node in retained and not complete:raise ValueError('Incomplete reexecution cannot replace completed instance')
        db.execute('UPDATE tests SET capture_state=?,source_version=?,source_attempt=? WHERE id=?',
                   ('complete' if complete else 'incomplete', version, attempt, ids[node]))
        expected = set(merged.execute('SELECT api_id,phase,aliases FROM relations WHERE nodeid=?', (node,)))
        if node in allowed:expected|=prior_positive.get(node,set())
        if observations(db, ids[node]) != expected: raise ValueError('Destination observation mismatch')
        actual_phases = set(db.execute('SELECT p.name,tp.state FROM test_phases tp JOIN phases p ON p.id=tp.phase_id WHERE tp.test_id=?', (ids[node],)))
        if actual_phases != set(stages.items()): raise ValueError('Destination phase mismatch')
    for node, phase, reason in merged.execute('SELECT nodeid,phase,reason FROM gaps'):
        tid = ids.get(node) if node else None
        if node and tid is None and not _is_module_collection_gap(node, phase, module_file):
            raise ValueError('Gap owner missing')
        if not db.execute('SELECT 1 FROM capture_gaps WHERE module_id=? AND test_id IS ? AND phase=? AND reason=?',
                          (module_id, tid, phase, reason)).fetchone():
            db.execute('INSERT INTO capture_gaps VALUES (?,?,?,?)', (module_id, tid, phase, reason))
    if not retained <= completed(db, module): raise ValueError('Completed source set changed')
    receipt.update(tests_touched=len(touched), retained_unchanged=len(retained),
                   relations=merged.execute('SELECT count(*) FROM relations').fetchone()[0])
    if allowed:receipt.update(authorized_reexecutions=sorted(allowed),reexecution_prior=history,
                             reexecution_semantics='union of prior and current positive evidence; source_capture attribution retained')


def append_capture(source, database, module, *, interrupted=False, boundary_gaps=(), interrupted_nodes=(), allow_reexecution_of=()):
    reexecution_nodes=set(allow_reexecution_of)
    if any(type(node) is not str or not node.startswith('pytorch/') or '::' not in node for node in reexecution_nodes):
        raise ValueError('Reexecution requires exact pytorch-prefixed test identities')
    source, database = Path(source).absolute(), Path(database).resolve()
    if source.is_symlink(): raise ValueError('Symlink capture directory')
    source = source.resolve()
    if database.is_relative_to(source): raise ValueError('Database inside disposable capture')
    manifest = read(source / 'manifest.json')
    if manifest.get('schema') != 'api_test_relations_plugin/1': raise ValueError('Not a plugin run')
    if Path(manifest['repo']).resolve().is_relative_to(source): raise ValueError('Capture contains source repository')
    allowed = {'manifest.json','api-catalog.json','test-sources.json','environment.json','progress.json',
               'relations.sqlite','relations.sqlite-wal','relations.sqlite-shm','runtime','shards',
               'resume-selection.json','capture-error.json','.retention.lock'}
    if {p.name for p in source.iterdir()} - allowed: raise ValueError('Unknown disposable capture contents')
    # DrvFS maps flock on the database itself onto Windows byte-range locks,
    # which can conflict with SQLite even in the same process. Lock a zero-byte sidecar.
    with Path(str(database)+'.append.lock').open('a') as db_lock, (source / '.retention.lock').open('a') as capture_lock:
        fcntl.flock(db_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(capture_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _idle(source, manifest)
        before = _inventory(source)
        with tempfile.TemporaryDirectory(prefix='.merge-', dir=source.parent) as temporary:
            merged = connect(Path(temporary))
            db = sqlite3.connect(database, timeout=60)
            try:
                db.execute('PRAGMA foreign_keys=ON'); db.execute('PRAGMA synchronous=FULL')
                schema = json.loads(db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()[0])
                if schema not in ('api_test_relations_single_database/1','api_test_relations_single_database/2'): raise ValueError('Unsupported destination schema')
                catalog = [(a['api_id'],a['qualified_name'],a['language'],a['kind']) for a in read(source / 'api-catalog.json')['apis']]
                if set(catalog) != set(db.execute('SELECT api_id,name,language,kind FROM apis')): raise ValueError('Frozen API catalog changed')
                merged.executemany('INSERT INTO apis VALUES (?,?,?,?)', catalog)
                merged.execute('INSERT INTO jobs VALUES (?,?,?,?)', ('pytest','pytorch/','settling',str(source)))
                finished = merge(merged, 'pytest', source / 'shards', final=True)
                merge_indexes = prepare_merge_indexes(merged)
                verify_raw(merged, source, completed(db, module)-reexecution_nodes)
                for reason in boundary_gaps:
                    merged.execute('INSERT OR IGNORE INTO gaps VALUES (?,?,?,?)', ('pytest','','session',reason))
                if interrupted:
                    if reexecution_nodes & completed(db,module):
                        raise ValueError('Incomplete reexecution cannot replace completed instance')
                    # Closing finally blocks must not turn interrupted work into completion.
                    merged.execute("UPDATE phases SET state='interrupted' WHERE state!='closed'")
                    for node in interrupted_nodes:
                        merged.execute("UPDATE phases SET state='interrupted' WHERE nodeid=? AND phase='call'", (node,))
                        merged.execute('INSERT OR IGNORE INTO gaps VALUES (?,?,?,?)', ('pytest',node,'call','runner_interrupted_phase'))
                merged.commit()
                receipt = {'source':str(source),'module':module,'source_hashes':before,'manifest':manifest,
                           'verified_at':time.time(),'cleanup':'pending','proves_zero_omissions':False,
                           'merge_indexes':list(merge_indexes),
                           'selection':read(source/'resume-selection.json') if (source/'resume-selection.json').exists() else None,
                           'boundaries':list(merged.execute('SELECT nodeid,phase,reason FROM gaps'))}
                receipt['process_evidence'] = {table:list(merged.execute('SELECT * FROM '+table))
                    for table in ('processes','process_details','process_tasks')}
                receipt['metadata'] = {name:read(source/name) for name in ('test-sources.json','environment.json','capture-error.json')
                    if (source/name).exists()}
                db.execute('CREATE TABLE IF NOT EXISTS capture_runs(attempt TEXT PRIMARY KEY,module TEXT,detail TEXT NOT NULL)')
                db.commit(); db.execute('BEGIN IMMEDIATE')
                prior = db.execute('SELECT detail FROM capture_runs WHERE attempt=?', (str(source),)).fetchone()
                if prior: raise ValueError('Attempt already committed; inspect receipt before cleanup retry')
                if reexecution_nodes:import_merged(db, merged, module, source.name, manifest.get('package_version'), receipt,allow_reexecution_of=reexecution_nodes)
                else:import_merged(db, merged, module, source.name, manifest.get('package_version'), receipt)
                collection_gap = merged.execute("SELECT 1 FROM gaps WHERE phase='collection' OR reason LIKE 'collection_unavailable:%' LIMIT 1").fetchone()
                state = 'interrupted' if interrupted else ('captured' if finished and not collection_gap else 'capture_incomplete')
                db.execute('UPDATE modules SET capture_state=? WHERE module=?', (state,module))
                if db.execute('PRAGMA foreign_key_check').fetchall(): raise ValueError('Foreign key failure')
                if db.execute('PRAGMA integrity_check').fetchall() != [('ok',)]: raise ValueError('Integrity failure')
                if _inventory(source) != before: raise ValueError('Source changed during merge')
                _idle(source,manifest)
                receipt['checks'] = ['raw_mapping_and_owner_match','destination_mapping_and_phases_equal',
                                     'prior_positive_subset','completed_instances_preserved','foreign_key_check','integrity_check']
                db.execute('INSERT INTO capture_runs VALUES (?,?,?)', (str(source),module,json.dumps(receipt)))
                db.commit()
                if db.execute('PRAGMA quick_check').fetchall() != [('ok',)]: raise ValueError('Committed database failed check')
                with database.open('rb') as stream: os.fsync(stream.fileno())
                _idle(source,manifest)
                if _inventory(source) != before: raise ValueError('Source changed after commit; preserve for review')
                try:
                    shutil.rmtree(source); receipt['cleanup']='complete'
                except OSError as exc:
                    receipt.update(cleanup='incomplete',cleanup_error=repr(exc))
                db.execute('UPDATE capture_runs SET detail=? WHERE attempt=?', (json.dumps(receipt),str(source))); db.commit()
                return {'state':state,'cleanup':receipt['cleanup'],'relations':receipt['relations'],
                        'tests_touched':receipt['tests_touched'],'checks':receipt['checks']}
            finally:
                db.close(); merged.close()
