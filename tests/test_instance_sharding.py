import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from instance_sharding import (partition_nodes, validate_plan, remaining_nodes, retry_state,
                               dispatch_allowed, positive_workers, validate_capture)
import retained_selection
import run_instance_shards


@pytest.mark.parametrize('limit',[1,2,5,11])
def test_partition_exact_disjoint_full_identity_coverage(limit):
    nodes=['pytorch/test/a.py::test_op[param::'+str(i)+']' for i in range(101)]
    shards=partition_nodes('a',nodes,limit)
    assert len(shards)==limit
    assert {n for s in shards for n in s['nodeids']}==set(nodes)
    assert sum(len(s['nodeids']) for s in shards)==len(nodes)
    assert max(map(lambda s:len(s['nodeids']),shards))-min(map(lambda s:len(s['nodeids']),shards))<=1
    assert shards==partition_nodes('a',list(reversed(nodes)),limit)


def test_partition_rejects_duplicate_and_nonpositive_and_handles_empty():
    with pytest.raises(ValueError):partition_nodes('a',['x','x'],2)
    with pytest.raises(ValueError):positive_workers(0)
    assert partition_nodes('a',[],5)==[]


@pytest.mark.parametrize('defect',['overlap','foreign','excluded','missing'])
def test_plan_rejects_unsafe_assignment(defect):
    nodes=['pytorch/test/a.py::test_a','pytorch/test/a.py::test_b']
    p={'remaining':{'a':nodes},'shards':partition_nodes('a',nodes,2)}
    if defect=='overlap':p['shards'][1]['nodeids']=p['shards'][0]['nodeids']
    if defect=='foreign':p['shards'][0]['nodeids']=['pytorch/test/xpu/test_a.py::test_a']
    if defect=='missing':p['shards'].pop()
    with pytest.raises(ValueError):validate_plan(p,{'a':{'file':'pytorch/test/a.py'}},lambda r:'xpu' if defect=='excluded' else None)


def test_remaining_uses_registered_complete_state_and_refuses_unknown(tmp_path):
    with sqlite3.connect(tmp_path/'db') as db:
        db.executescript("CREATE TABLE modules(id,module); CREATE TABLE tests(module_id,nodeid,capture_state); INSERT INTO modules VALUES(1,'a'); INSERT INTO tests VALUES(1,'done','complete'),(1,'left','incomplete');")
        assert remaining_nodes(db,'a',['done','left'])==['left']
        with pytest.raises(ValueError):remaining_nodes(db,'a',['unknown'])


def test_progress_continuation_and_no_progress_diagnosis():
    assert retry_state(['a','b'],['b'])=='pending'
    assert retry_state(['a'],[])=='captured'
    assert retry_state(['a'],['a'])=='continuation_needs_diagnosis'
    assert retry_state(['a'],['a'],True)=='pending'
    with pytest.raises(ValueError):retry_state(['a'],['a','b'])


def test_dispatch_parameter_ceiling_resource_headroom_gpu_and_stagger():
    resource={'memory_available_mib':10000,'gpu_free_mib':5000,'host_c_free_gib':20,'database_volume_free_gib':20}
    assert dispatch_allowed(resource,4,5,5,30)
    assert not dispatch_allowed(resource,5,5,5,30)
    assert not dispatch_allowed(resource,2,5,2,30)
    assert not dispatch_allowed(resource,0,5,5,29)
    assert not dispatch_allowed({**resource,'gpu_free_mib':None},0,5,5,30)
    assert not dispatch_allowed({**resource,'memory_available_mib':4000},0,5,5,30,3000)


def test_selection_separates_completed_from_other_shards_and_guards_final_collection(tmp_path,monkeypatch):
    repo=tmp_path/'repo';source=repo/'test/a.py';source.parent.mkdir(parents=True);source.touch();out=tmp_path/'out';out.mkdir()
    nodes=['pytorch/test/a.py::test_'+n for n in ['done','a','b']]
    selection=tmp_path/'selection.json';selection.write_text(json.dumps({'job':'a','shard_id':'one','retained':[nodes[0]],'allowed':[nodes[1]]}))
    monkeypatch.setenv('RELATION_RESUME_SELECTION',str(selection))
    cfg=SimpleNamespace(getoption=lambda name:str(repo if name=='--api-relations-repo' else out),hook=SimpleNamespace(pytest_deselected=lambda **kw:None))
    items=[SimpleNamespace(path=source,nodeid='test/a.py::test_'+n) for n in ['done','a','b']]
    retained_selection.pytest_collection_modifyitems(None,cfg,items)
    assert len(items)==1 and items[0].nodeid.endswith('test_a')
    receipt=json.loads((out/'resume-selection.json').read_text())
    assert receipt['selected']==[nodes[1]] and receipt['deselected']==[nodes[0]] and receipt['unassigned']==[nodes[2]]
    retained_selection.pytest_collection_finish(SimpleNamespace(config=cfg,items=items))
    with pytest.raises(pytest.UsageError):retained_selection.pytest_collection_finish(SimpleNamespace(config=cfg,items=[]))


def test_raw_shard_owner_escape_and_observer_failure_rejected(tmp_path):
    out=tmp_path/'out';sh=out/'shards';sh.mkdir(parents=True);node='pytorch/test/a.py::test_a'
    (out/'resume-selection.json').write_text(json.dumps({'selected':[node],'deselected':[]}))
    path=sh/'one.sqlite'
    with sqlite3.connect(path) as db:
        db.executescript('CREATE TABLE phases(nodeid,phase,state); CREATE TABLE gaps(nodeid,phase,reason);')
        db.execute('INSERT INTO phases VALUES (?,?,?)',(node,'call','closed'))
    worker={'output':str(out),'shards':str(sh),'allowed_nodeids':[node]}
    validate_capture(worker)
    with sqlite3.connect(path) as db:db.execute("INSERT INTO gaps VALUES (?, 'call','collector_exception:bad')",(node,))
    with pytest.raises(ValueError,match='Observer'):validate_capture(worker)
    with sqlite3.connect(path) as db:db.execute('DELETE FROM gaps');db.execute("UPDATE phases SET nodeid='foreign'")
    with pytest.raises(ValueError,match='owner'):validate_capture(worker)


def test_entered_nodes_ignores_reused_child_lifecycle(tmp_path):
    for name,role,node,state in [('parent','root','parent-open','entered'),('child','child','parent-closed','entered')]:
        with sqlite3.connect(tmp_path/(name+'.sqlite')) as db:
            db.executescript('CREATE TABLE meta(key,value); CREATE TABLE phases(nodeid,phase,state);')
            db.execute('INSERT INTO meta VALUES (?,?)',('role',role));db.execute('INSERT INTO phases VALUES (?,?,?)',(node,'call',state))
    assert run_instance_shards.entered_nodes({'shards':str(tmp_path)})==['parent-open']


def test_private_raw_archive_is_durable_and_original_is_untouched(tmp_path,monkeypatch):
    source=tmp_path/'attempt';source.mkdir();(source/'evidence').write_bytes(b'original evidence')
    (tmp_path/'preserved').mkdir();monkeypatch.setattr(run_instance_shards,'attempt_processes',lambda w:{})
    synced=[];monkeypatch.setattr(run_instance_shards.os,'fsync',lambda fd:synced.append(fd))
    worker={'output':str(source),'attempt':'one','log':str(tmp_path/'log')}
    archive=Path(run_instance_shards.preserve_attempt(tmp_path,worker))
    assert synced and (archive/'evidence').read_bytes()==(source/'evidence').read_bytes()
    assert json.loads((archive.parent/'provenance.json').read_text())['scratch_retained']
