"""Post-exit, lossless merge/verification with automatic raw-output disposal."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

from .cli import digest, read
from .storage import active_children, connect, merge


def _idle(root, manifest):
    pids = {int(p.name.split('-')[0].split('.')[0])
            for p in (root / 'shards').glob('*.sqlite')
            if p.name.split('-')[0].split('.')[0].isdigit()}
    if manifest.get('capture_pid'):
        pids.add(int(manifest['capture_pid']))
    for section in ('launches', 'receipts', 'checkpoints'):
        for path in (root / 'shards' / section).glob('*.json'):
            data = read(path)
            if data.get('pid'):
                pids.add(int(data['pid']))
    live = {p for p in pids if Path('/proc', str(p)).exists()}
    live |= active_children(root / 'shards')
    if live:
        raise RuntimeError('Capture producers must exit before compacting: ' + str(sorted(live)))


def _inventory(root):
    result = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symlinks are not allowed in disposable capture output')
        if path.is_file() and path.name != '.retention.lock' and not path.name.endswith('-shm'):
            result[str(path.relative_to(root))] = digest(path)
    return result


def _raw_check(db, root):
    """Independent comparison; reject conflicting shard rows, never silently drop them."""
    for path in sorted((root / 'shards').glob('*.sqlite')):
        with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as raw:
            if raw.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
                raise ValueError('Raw shard integrity failure: ' + path.name)
            tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            meta = dict(raw.execute('SELECT key,value FROM meta')) if 'meta' in tables else {}
            for table in ('tests', 'relations', 'gaps', 'phases'):
                if table not in tables or (table == 'phases' and meta.get('role') == 'child'):
                    continue
                columns = [r[1] for r in raw.execute('PRAGMA table_info(' + table + ')')]
                for row in raw.execute('SELECT * FROM ' + table):
                    if table == 'tests' and (not row[0].startswith('pytorch/') or row[1] != row[0].split('::')[0]):
                        raise ValueError('Invalid test identity/source path')
                    if table == 'relations':
                        aliases = json.loads(row[3])
                        if not isinstance(aliases, list) or row[2] not in aliases or any(
                            not db.execute('SELECT 1 FROM apis WHERE api_id=?', (a,)).fetchone() for a in aliases):
                            raise ValueError('Invalid API alias group')
                    where = ' AND '.join('"' + col + '" IS ?' for col in columns)
                    if not db.execute('SELECT 1 FROM ' + table + ' WHERE job=? AND ' + where,
                                      ('pytest', *row)).fetchone():
                        raise ValueError('Raw/merged mismatch or conflicting shard rows: ' + table)


def _pack(merged, target, receipt):
    """Intern repeated strings while retaining the original SQL read interface."""
    with sqlite3.connect(target) as db:
        db.execute('PRAGMA synchronous=FULL')
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('ATTACH DATABASE ? AS source', (str(merged),))
        tables = list(db.execute("SELECT name,sql FROM source.sqlite_master WHERE type='table'"))
        for name, ddl in tables:
            if name == 'relations':
                continue
            db.execute(ddl)
            db.execute('INSERT INTO "' + name + '" SELECT * FROM source."' + name + '"')
        db.execute('CREATE TABLE relation_strings(id INTEGER PRIMARY KEY,value TEXT NOT NULL UNIQUE)')
        columns = ('job', 'nodeid', 'phase', 'api_id', 'aliases', 'site')
        for col in columns:
            db.execute('INSERT OR IGNORE INTO relation_strings(value) SELECT ' + col + ' FROM source.relations')
        db.execute('CREATE TABLE relation_keys(' + ','.join(
            col + ' INTEGER NOT NULL REFERENCES relation_strings(id)' for col in columns) +
            ', PRIMARY KEY(job,nodeid,phase,api_id)) WITHOUT ROWID')
        joins = ' '.join('JOIN relation_strings s' + str(i) + ' ON s' + str(i) + '.value=r.' + col
                         for i, col in enumerate(columns))
        db.execute('INSERT INTO relation_keys SELECT ' + ','.join('s' + str(i) + '.id' for i in range(6)) +
                   ' FROM source.relations r ' + joins)
        db.execute('CREATE INDEX relation_keys_api ON relation_keys(api_id,nodeid)')
        db.execute('CREATE VIEW relations AS SELECT ' + ','.join('s' + str(i) + '.value AS ' + col
                   for i, col in enumerate(columns)) + ' FROM relation_keys r ' +
                   ' '.join('JOIN relation_strings s' + str(i) + ' ON s' + str(i) + '.id=r.' + col
                            for i, col in enumerate(columns)))
        for name, _ in tables:
            for left, right in (('main', 'source'), ('source', 'main')):
                if db.execute(f'SELECT * FROM {left}."{name}" EXCEPT SELECT * FROM {right}."{name}" LIMIT 1').fetchone():
                    raise ValueError('Compact database differs from merged data: ' + name)
        if db.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('Foreign key check failed')
        if db.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
            raise ValueError('Database integrity check failed')
        receipt['checks'] = ['raw_rows_match_merge', 'all_tables_bidirectional_except',
                             'foreign_key_check', 'integrity_check']
        receipt['counts'] = {name: db.execute('SELECT count(*) FROM ' + name).fetchone()[0]
                             for name in ('apis', 'tests', 'relations', 'phases', 'gaps')}
        db.execute('CREATE TABLE retention_receipt(id INTEGER PRIMARY KEY,detail TEXT NOT NULL)')
        db.execute('INSERT INTO retention_receipt VALUES (1,?)', (json.dumps(receipt),))
        db.commit()
    # Reopen committed file: verification must survive closing the writer.
    with sqlite3.connect(target.as_uri() + '?mode=ro', uri=True) as db:
        if db.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
            raise ValueError('Committed database integrity check failed')
    with target.open('rb') as stream:
        os.fsync(stream.fileno())


def compact(source, database):
    """Produce ONE new database, then delete this run only after scripted checks."""
    if os.name != 'posix':
        raise RuntimeError('Post-exit compaction requires the Linux capture host')
    if source.is_symlink():
        raise ValueError('Source must not be a symlink')
    source, database = source.resolve(), database.resolve()
    if database.exists() or database.is_relative_to(source):
        raise ValueError('Database must be a new file outside the disposable run directory')
    manifest = read(source / 'manifest.json')
    if manifest.get('schema') != 'api_test_relations_plugin/1':
        raise ValueError('Not a plugin capture directory')
    repo = Path(manifest['repo']).resolve()
    if repo == source or repo.is_relative_to(source):
        raise ValueError('Refusing to delete a source repository')
    allowed = {'manifest.json', 'api-catalog.json', 'test-sources.json', 'environment.json',
               'progress.json', 'relations.sqlite', 'relations.sqlite-wal', 'relations.sqlite-shm',
               'runtime', 'shards', '.retention.lock'}
    if set(p.name for p in source.iterdir()) - allowed:
        raise ValueError('Unknown files in capture directory; preserve and inspect them')
    database.parent.mkdir(parents=True, exist_ok=True)
    with (source / '.retention.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _idle(source, manifest)
        before = _inventory(source)
        receipt = {'schema': 'verified_retention/1', 'source': str(source), 'manifest': manifest,
                   'source_hashes': before, 'verified_at': time.time(),
                   'cleanup': 'pending', 'proves_zero_omissions': False}
        receipt['metadata'] = {name: read(source / name) for name in
                               ('test-sources.json', 'environment.json', 'progress.json')
                               if (source / name).exists()}
        with tempfile.TemporaryDirectory(prefix='.relations-verify-', dir=database.parent) as folder:
            folder = Path(folder)
            db = connect(folder)
            try:
                catalog = read(source / 'api-catalog.json')
                db.executemany('INSERT INTO apis VALUES (?,?,?,?)', [(a['api_id'], a['qualified_name'],
                               a['language'], a['kind']) for a in catalog['apis']])
                db.execute('INSERT INTO jobs VALUES (?,?,?,?)', ('pytest', 'pytorch/', 'settling', str(source)))
                finished = merge(db, 'pytest', source / 'shards', final=True)
                db.execute('UPDATE jobs SET state=?', ('capture_finished' if finished else 'capture_incomplete',))
                db.commit()
                _raw_check(db, source)
            finally:
                db.close()
            packed = folder / 'compact.sqlite'
            _pack(folder / 'relations.sqlite', packed, receipt)
            _idle(source, manifest)
            if _inventory(source) != before:
                raise RuntimeError('Capture files changed during verification; preserved original')
            # Exclusive atomic publication: never overwrite an existing database.
            os.link(packed, database)
            fd = os.open(database.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            try:
                shutil.rmtree(source)
                receipt['cleanup'] = 'complete'
            except OSError as error:
                receipt['cleanup'] = 'incomplete'
                receipt['cleanup_error'] = str(error)
            with sqlite3.connect(database) as final:
                final.execute('PRAGMA synchronous=FULL')
                final.execute('UPDATE retention_receipt SET detail=? WHERE id=1', (json.dumps(receipt),))
        return {'database': str(database), 'cleanup': receipt['cleanup'], 'checks': receipt['checks'],
                **receipt['counts'], 'proves_zero_omissions': False}
