"""Owner lookups stay indexed without changing aliases, provenance or rollback."""
import json
import sqlite3
import pytest
from api_test_relations import single_store
from api_test_relations.storage import connect
from test_alias_evidence import divergent_capture,merge_capture
from test_single_store import baseline,add_phases
from test_retention import capture


def logical_rows(db):
    return {table:list(db.execute('SELECT * FROM "'+table+'" ORDER BY '+','.join(
        '"'+row[1]+'"' for row in db.execute('PRAGMA table_info("'+table+'")'))))
        for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")}


def test_index_changes_query_cost_not_rows_or_owner_scope(tmp_path):
    merged=connect(tmp_path)
    nodes=['pytorch/test/test_dense.py::TestDense::test_case['+str(i)+']' for i in range(1000)]
    for job in ('pytest','second_job'):
        merged.executemany('INSERT INTO tests VALUES (?,?,?)',[(job,n,n.split('::')[0]) for n in nodes])
        merged.executemany('INSERT INTO phases VALUES (?,?,?,?)',[(job,n,'call','closed') for n in nodes])
        merged.executemany('INSERT INTO relations VALUES (?,?,?,?,?,?)',[(job,n,'call','a','["a"]',json.dumps(['test_dense.py',i+1])) for i,n in enumerate(nodes)])
    merged.commit();before=logical_rows(merged)
    def workload():
        ticks=[0]
        def progress():ticks[0]+=1;return 0
        merged.set_progress_handler(progress,100)
        try:result=[merged.execute('SELECT api_id,phase,aliases FROM relations WHERE nodeid=?',(n,)).fetchall() for n in nodes[::5]]
        finally:merged.set_progress_handler(None,0)
        return result,ticks[0]
    expected,slow=workload();single_store.prepare_merge_indexes(merged)
    actual,fast=workload()
    assert actual==expected and all(len(rows)==2 for rows in actual)
    assert fast<slow/10
    assert logical_rows(merged)==before
    for table in ('relations','phases'):
        plan=merged.execute('EXPLAIN QUERY PLAN SELECT * FROM '+table+' WHERE nodeid=?',(nodes[0],)).fetchall()
        assert 'SEARCH' in str(plan) and 'nodeid=?' in str(plan)
    single_store.prepare_merge_indexes(merged)  # Safe for an already indexed rescue.
    assert logical_rows(merged)==before
    merged.close()


def test_indexed_import_has_identical_full_baseline_relations_owners_and_sources(tmp_path):
    root=divergent_capture(tmp_path/'raw',duplicate=True)
    merged=merge_capture(root,tmp_path/'merged');before=logical_rows(merged)
    results=[]
    for name,indexed in (('reference',False),('indexed',True)):
        folder=tmp_path/name;folder.mkdir();target=baseline(folder)
        with sqlite3.connect(target) as db:
            db.execute("INSERT INTO apis VALUES (2,'b','torch.random.add','python','function')");db.commit()
            if indexed:single_store.prepare_merge_indexes(merged)
            receipt={'source':'one-identical-logical-capture'}
            single_store.import_merged(db,merged,'one','same-attempt','same-version',receipt)
            assert db.execute('PRAGMA foreign_key_check').fetchall()==[]
            results.append(logical_rows(db))
    assert results[0]==results[1]
    assert logical_rows(merged)==before
    merged.close()


def test_append_prepares_indexes_and_records_them_before_disposal(tmp_path,monkeypatch):
    target=baseline(tmp_path);root=capture(tmp_path);add_phases(root)
    original=single_store.import_merged;called=[]
    def checked(db,merged,*args):
        plan=merged.execute('EXPLAIN QUERY PLAN SELECT api_id,phase,aliases FROM relations WHERE nodeid=?',('probe',)).fetchall()
        assert 'merge_relations_by_nodeid' in str(plan);called.append(True)
        return original(db,merged,*args)
    monkeypatch.setattr(single_store,'import_merged',checked)
    result=single_store.append_capture(root,target,'one')
    assert called and result['cleanup']=='complete' and not root.exists()
    with sqlite3.connect(target) as db:
        receipt=json.loads(db.execute('SELECT detail FROM capture_runs').fetchone()[0])
        assert receipt['merge_indexes']==['merge_relations_by_nodeid','merge_phases_by_nodeid']


def test_index_failure_preserves_formal_database_and_raw_capture(tmp_path,monkeypatch):
    target=baseline(tmp_path);root=capture(tmp_path);add_phases(root)
    with sqlite3.connect(target) as db:before=logical_rows(db)
    def fail(merged):
        merged.execute('CREATE INDEX test_partial_index ON relations(nodeid)')
        raise sqlite3.OperationalError('injected index creation failure')
    monkeypatch.setattr(single_store,'prepare_merge_indexes',fail)
    with pytest.raises(sqlite3.OperationalError,match='injected index'):
        single_store.append_capture(root,target,'one')
    assert (root/'shards/2147483000.sqlite').exists()
    with sqlite3.connect(target) as db:assert logical_rows(db)==before
