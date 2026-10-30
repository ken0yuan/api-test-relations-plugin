"""Additive caller classification; legacy observations are never guessed."""
import json

KINDS=('direct','indirect','unknown')
COLUMNS=('nodeid','phase','api_id','aliases','kind','site')


def create_tables(db):
    db.execute('''CREATE TABLE IF NOT EXISTS call_kinds(
        job TEXT,nodeid TEXT,phase TEXT,api_id TEXT,aliases TEXT,
        kind TEXT CHECK(kind IN ('direct','indirect','unknown')),site TEXT,
        PRIMARY KEY(job,nodeid,phase,api_id,aliases,kind,site))''')
    db.execute('''CREATE TABLE IF NOT EXISTS call_kind_sources(
        job TEXT,nodeid TEXT,phase TEXT,api_id TEXT,aliases TEXT,
        kind TEXT CHECK(kind IN ('direct','indirect','unknown')),site TEXT,shard TEXT,
        PRIMARY KEY(job,nodeid,phase,api_id,aliases,kind,site,shard))''')


def raw_rows(raw):
    tables={r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    meta=dict(raw.execute('SELECT key,value FROM meta')) if 'meta' in tables else {}
    declared=meta.get('call_classification')
    if declared not in (None,'caller_test_function/1'):
        raise ValueError('Unsupported call classification')
    if 'call_kinds' not in tables:
        if declared:raise ValueError('Missing declared call classification')
        return []
    columns=tuple(r[1] for r in raw.execute('PRAGMA table_info(call_kinds)'))
    if columns!=COLUMNS:raise ValueError('Invalid call classification columns')
    rows=list(raw.execute('SELECT * FROM call_kinds'))
    if rows and not declared:raise ValueError('Unversioned call classification')
    base=set(raw.execute('SELECT nodeid,phase,api_id,aliases FROM relations'))
    classified=set()
    for node,phase,api,aliases,kind,site in rows:
        if kind not in KINDS or (node,phase,api,aliases) not in base:
            raise ValueError('Invalid call classification owner or kind')
        point=json.loads(site)
        if not isinstance(point,list) or len(point)!=2 or not isinstance(point[0],str) or not isinstance(point[1],int):
            raise ValueError('Invalid classified call site')
        classified.add((node,phase,api,aliases))
    if declared and classified!=base:raise ValueError('Missing classified observation')
    return rows


def merge_rows(db,raw,job,shard):
    rows=raw_rows(raw)
    db.executemany('INSERT OR IGNORE INTO call_kinds VALUES (?,?,?,?,?,?,?)',[(job,*r) for r in rows])
    db.executemany('INSERT OR IGNORE INTO call_kind_sources VALUES (?,?,?,?,?,?,?,?)',[(job,*r,shard) for r in rows])


def verify_rows(db,raw,job,shard):
    rows=raw_rows(raw)
    expected=set(rows)
    actual=set(db.execute('SELECT '+','.join(COLUMNS)+' FROM call_kind_sources WHERE job=? AND shard=?',(job,shard)))
    if expected!=actual:raise ValueError('Raw classified evidence source mismatch')
    for row in rows:
        where=' AND '.join(c+' IS ?' for c in COLUMNS)
        if not db.execute('SELECT 1 FROM call_kinds WHERE job=? AND '+where,(job,*row)).fetchone():
            raise ValueError('Raw classified evidence mismatch')


def upgrade_single(db):
    db.execute('''CREATE TABLE IF NOT EXISTS observation_call_kinds(
        api_key INTEGER,test_key INTEGER,phase_key INTEGER,alias_group_key INTEGER,
        kind TEXT CHECK(kind IN ('direct','indirect','unknown')),site TEXT,
        source_capture TEXT,source_shard TEXT,
        PRIMARY KEY(api_key,test_key,phase_key,alias_group_key,kind,site,source_capture,source_shard),
        FOREIGN KEY(api_key,test_key,phase_key,alias_group_key)
            REFERENCES observations(api_key,test_key,phase_key,alias_group_key))''')


def import_single(db,merged,api_ids,test_ids,source):
    rows=[]
    for node,phase,api,aliases,kind,site,shard in merged.execute(
            'SELECT nodeid,phase,api_id,aliases,kind,site,shard FROM call_kind_sources'):
        pid=db.execute('SELECT id FROM phases WHERE name=?',(phase,)).fetchone()[0]
        gid=db.execute('SELECT id FROM alias_groups WHERE api_ids_json=?',(aliases,)).fetchone()[0]
        rows.append((api_ids[api],test_ids[node],pid,gid,kind,site,source,shard))
    db.executemany('INSERT OR IGNORE INTO observation_call_kinds VALUES (?,?,?,?,?,?,?,?)',rows)
    actual=set(db.execute('SELECT * FROM observation_call_kinds WHERE source_capture=?',(source,)))
    if actual!=set(rows):raise ValueError('Destination classified evidence mismatch')


def single_observations(db,test_id):
    return set(db.execute('''SELECT a.api_id,p.name,g.api_ids_json,k.kind
        FROM observation_call_kinds k JOIN apis a ON a.id=k.api_key
        JOIN phases p ON p.id=k.phase_key JOIN alias_groups g ON g.id=k.alias_group_key
        WHERE k.test_key=?''',(test_id,)))


def query_single(db,args):
    ids=[r[0] for r in db.execute('SELECT api_id FROM apis WHERE name=? OR api_id=?',(args.api,args.api))]
    selected=getattr(args,'call_kind','all')
    tests={}
    tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for aid in ids:
        base=list(db.execute('''SELECT o.api_key,o.test_key,o.phase_key,o.alias_group_key,
            t.nodeid,t.file,p.name,g.api_ids_json,t.source_version,t.source_attempt
            FROM observations o JOIN apis a ON a.id=o.api_key JOIN tests t ON t.id=o.test_key
            JOIN phases p ON p.id=o.phase_key JOIN alias_groups g ON g.id=o.alias_group_key
            WHERE a.api_id=?''',(aid,)))
        for ak,tk,pk,gk,node,file,phase,aliases,version,attempt in base:
            records=list(db.execute('''SELECT kind,site,source_capture,source_shard
                FROM observation_call_kinds WHERE api_key=? AND test_key=? AND phase_key=? AND alias_group_key=?''',
                (ak,tk,pk,gk))) if 'observation_call_kinds' in tables else []
            legacy=list(db.execute('''SELECT site,source_capture,source_shard FROM observation_evidence
                WHERE api_key=? AND test_key=? AND phase_key=? AND alias_group_key=?''',(ak,tk,pk,gk))) if 'observation_evidence' in tables else []
            classified_sources={(r[2],r[3]) for r in records}
            records.extend(('unknown',site,source,shard) for site,source,shard in legacy
                           if (source,shard) not in classified_sources)
            if not records:records=[('unknown',None,None,None)]
            for kind,site,source,shard in records:
                if selected!='all' and kind!=selected:continue
                entry=tests.setdefault(node,{'nodeid':node,'file':file,'evidence':[],
                    'evidence_origin':{'version':version,'attempt':attempt},'capture_gaps':[
                        r[0] for r in db.execute('SELECT reason FROM capture_gaps WHERE test_id=?',(tk,))]})
                entry['evidence'].append({'api_id':aid,'phase':phase,'call_kind':kind,
                    'same_callable_aliases':json.loads(aliases),'first_site':json.loads(site) if site else None,
                    'source_capture':source,'source_shards':[shard] if shard else []})
    return {'api':args.api,'catalog_matches':ids,'tests':list(tests.values()),
            'classified_tests':list(tests.values()),'call_kind_filter':selected,'unobserved_means_no_test':False}


def classified_query_rows(db,aid):
    tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    if 'call_kind_sources' in tables:
        for job,node,phase,aliases,kind,site,shard in db.execute(
                'SELECT job,nodeid,phase,aliases,kind,site,shard FROM call_kind_sources WHERE api_id=?',(aid,)):
            yield job,node,phase,aliases,kind,site,[shard]
    # Old records have no recoverable direct/indirect classification. Even a
    # legacy first site that looks like a test body cannot reconstruct later calls.
    for job,node,phase,aliases,site in db.execute(
            'SELECT job,nodeid,phase,aliases,site FROM relations WHERE api_id=?',(aid,)):
        sources=[r[0] for r in db.execute(
            'SELECT shard FROM relation_sources WHERE job=? AND nodeid=? AND phase=? AND api_id=? AND aliases=? AND site=?',
            (job,node,phase,aid,aliases,site))] if 'relation_sources' in tables else []
        for shard in sources or [None]:
            classified='call_kind_sources' in tables and db.execute(
                'SELECT 1 FROM call_kind_sources WHERE job=? AND nodeid=? AND phase=? AND api_id=? AND aliases=? AND shard IS ?',
                (job,node,phase,aid,aliases,shard)).fetchone()
            if not classified:yield job,node,phase,aliases,'unknown',site,[] if shard is None else [shard]
