import json
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace,FunctionType

import pytest
from api_test_relations import classification,retention,single_store
from api_test_relations.cli import main,read
from api_test_relations.storage import connect,merge
from relation_context import describe_test,TEST_FUNCTION
from relation_collector import Collector,Registry
from test_plugin import launch
from test_retention import capture
from test_single_store import baseline,add_phases


@pytest.fixture(scope='module')
def classified_capture(tmp_path_factory):
    folder=tmp_path_factory.mktemp('classified');repo=folder/'repo'
    (repo/'torch').mkdir(parents=True);(repo/'test').mkdir()
    (repo/'torch/__init__.py').write_text(
        'def alpha(): return 1\nalias=alpha\ndef fused(): return alpha()\nnative=len\n',encoding='utf-8')
    (repo/'sitecustomize.py').write_text(
        'import sys\nfrom importlib.machinery import BuiltinImporter,FrozenImporter,PathFinder\n'
        'sys.meta_path[:]=[BuiltinImporter,FrozenImporter,PathFinder]\n',encoding='utf-8')
    (repo/'test/test_classified.py').write_text('''
import functools,multiprocessing,subprocess,sys,threading
from concurrent.futures import ProcessPoolExecutor
import pytest,torch

def helper(): return torch.alpha()

def test_direct(): torch.alpha()

@pytest.mark.parametrize('op',[torch.alpha])
def test_dynamic(op): op()

def test_helper(): helper()

def test_fused(): torch.fused()

def test_indirect_then_direct():
    helper()
    torch.alpha()

def test_direct_then_indirect():
    torch.alpha()
    helper()

def test_alias(): torch.alias()

def test_native(): torch.native([1])

def test_lambda():
    worker=lambda:torch.alpha()
    worker()

def test_thread():
    t=threading.Thread(target=helper);t.start();t.join()

def test_child():
    subprocess.run([sys.executable,'-c','import torch;torch.alpha()'],check=True)

@pytest.fixture(scope='module')
def shared_pool():
    with ProcessPoolExecutor(1,mp_context=multiprocessing.get_context('spawn')) as pool:
        yield pool

@pytest.mark.parametrize('number',[1,2])
def test_pool(shared_pool,number): shared_pool.submit(helper).result()

def test_self_in_pool_a(shared_pool):
    if shared_pool is None:
        torch.alpha()
        return
    shared_pool.submit(test_self_in_pool_a,None).result()

def test_self_in_pool_b(shared_pool):
    if shared_pool is None:
        torch.alpha()
        return
    shared_pool.submit(test_self_in_pool_b,None).result()

def test_body_in_child():
    if multiprocessing.current_process().name!='MainProcess':
        torch.alpha()
        return
    context=multiprocessing.get_context('spawn')
    child=context.Process(target=test_body_in_child)
    child.start();child.join(20)
    assert child.exitcode==0

def decorate(function):
    @functools.wraps(function)
    def wrapper():
        torch.alpha()
        return function()
    return wrapper

@decorate
def test_decorated(): torch.alpha()

class TestClass:
    def test_method(self): torch.alpha()
''',encoding='utf-8')
    catalog=folder/'catalog.json'
    catalog.write_text(json.dumps({'apis':[{'api_id':name,'qualified_name':'torch.'+name,
        'language':'python','kind':'function'} for name in ('alpha','alias','fused','native')]}),encoding='utf-8')
    out=folder/'capture'
    launch(repo,'--api-relations-out',str(out),'--api-relations-repo',str(repo),
           '--api-relations-catalog',str(catalog))
    return repo,out


def kinds(out,name,api='alpha'):
    with sqlite3.connect(out/'relations.sqlite') as db:
        return {r[0] for r in db.execute(
            'SELECT DISTINCT kind FROM call_kinds WHERE nodeid LIKE ? AND api_id=? AND phase="call"',
            ('%::'+name,api))}


@pytest.mark.parametrize('test,expected',[
    ('test_direct',{'direct'}),('test_dynamic[alpha]',{'direct'}),
    ('test_helper',{'indirect'}),('test_fused',{'indirect'}),
    ('test_indirect_then_direct',{'direct','indirect'}),
    ('test_direct_then_indirect',{'direct','indirect'}),
    ('test_alias',{'direct'}),('test_lambda',{'indirect'}),
    ('test_thread',{'indirect'}),('test_child',{'indirect'}),
    ('test_pool[1]',{'indirect'}),('test_pool[2]',{'indirect'}),
    ('test_self_in_pool_a',{'direct'}),('test_self_in_pool_b',{'direct'}),
    ('test_body_in_child',{'direct'}),('test_decorated',{'direct','indirect'}),
    ('test_method',{'direct'}),
])
def test_actual_caller_classification(classified_capture,test,expected):
    _,out=classified_capture
    assert kinds(out,test)==expected


def test_fused_entry_native_entry_and_alias_groups(classified_capture):
    _,out=classified_capture
    assert kinds(out,'test_fused','fused')=={'direct'}
    # len is also genuinely called by pytest; its exact alias must retain both
    # test-body and framework calls instead of filtering the latter away.
    assert 'direct' in kinds(out,'test_native','native')
    assert kinds(out,'test_alias','alias')=={'direct'}
    with sqlite3.connect(out/'relations.sqlite') as db:
        assert db.execute('SELECT count(*) FROM gaps').fetchone()[0]==0
        assert all(json.loads(row[0])==['alias','alpha'] for row in db.execute(
            'SELECT aliases FROM call_kinds WHERE api_id="alpha"'))


def test_no_duplicate_or_cross_task_classification(classified_capture):
    _,out=classified_capture
    with sqlite3.connect(out/'relations.sqlite') as db:
        rows=list(db.execute('SELECT nodeid,kind,site FROM call_kinds WHERE api_id="alpha" AND nodeid LIKE "%test_indirect_then_direct"'))
        assert len(rows)==2 and len({r[2] for r in rows})==2
        tasks=list(db.execute('SELECT nodeid,state FROM process_tasks'))
        assert {node.rsplit('::',1)[-1] for node,state in tasks}=={
            'test_pool[1]','test_pool[2]','test_self_in_pool_a','test_self_in_pool_b'}
        assert all(state=='finished' for node,state in tasks)


def test_direct_filter_and_compact_round_trip(classified_capture,tmp_path,capsys):
    _,out=classified_capture
    main(['query','--out',str(out),'--api','torch.alpha','--call-kind','direct'])
    before=json.loads(capsys.readouterr().out)
    assert before['tests'] and all(e['call_kind']=='direct' for t in before['tests'] for e in t['evidence'])
    assert not any(t['nodeid'].endswith('test_helper') for t in before['tests'])
    copied=tmp_path/'copy';shutil.copytree(out,copied)
    target=tmp_path/'compact.sqlite';retention.compact(copied,target)
    main(['query','--out',str(target),'--api','torch.alpha','--call-kind','direct'])
    after=json.loads(capsys.readouterr().out)
    strip=lambda r:{(t['nodeid'],e['phase'],tuple(e['same_callable_aliases']),tuple(e['first_site']))
                    for t in r['tests'] for e in t['evidence']}
    assert strip(before)==strip(after)


def classified_legacy_fixture(tmp_path):
    root=capture(tmp_path);add_phases(root)
    with sqlite3.connect(root/'shards/2147483001-late.sqlite') as raw:
        raw.execute('CREATE TABLE call_kinds(nodeid TEXT,phase TEXT,api_id TEXT,aliases TEXT,kind TEXT,site TEXT)')
        node,phase,api,aliases,site=raw.execute('SELECT * FROM relations').fetchone()
        raw.executemany('INSERT INTO call_kinds VALUES (?,?,?,?,?,?)',[
            (node,phase,api,aliases,'direct',site),
            (node,phase,api,aliases,'indirect',json.dumps(['pytorch/test/helper.py',4]))])
        raw.execute('INSERT INTO meta VALUES (?,?)',('call_classification','caller_test_function/1'))
    return root


def test_classification_survives_single_database_append(tmp_path,capsys):
    target=baseline(tmp_path);root=classified_legacy_fixture(tmp_path)
    result=single_store.append_capture(root,target,'one')
    assert result['cleanup']=='complete'
    with sqlite3.connect(target) as db:
        assert {r[0] for r in db.execute('SELECT kind FROM observation_call_kinds')}=={'direct','indirect'}
        assert db.execute('PRAGMA foreign_key_check').fetchall()==[]
    for kind in ('direct','indirect'):
        main(['query','--out',str(target),'--api','torch.add','--call-kind',kind])
        report=json.loads(capsys.readouterr().out)
        assert len(report['tests'])==1
        assert {e['call_kind'] for e in report['tests'][0]['evidence']}=={kind}


@pytest.mark.parametrize('defect',['missing','orphan','kind'])
def test_invalid_classification_blocks_cleanup(tmp_path,defect):
    root=classified_legacy_fixture(tmp_path)
    with sqlite3.connect(root/'shards/2147483001-late.sqlite') as raw:
        if defect=='missing':raw.execute('DELETE FROM call_kinds')
        elif defect=='orphan':raw.execute("UPDATE call_kinds SET nodeid='pytorch/test/other.py::test_other'")
        else:raw.execute("UPDATE call_kinds SET kind='invented'")
    with pytest.raises(ValueError):retention.compact(root,tmp_path/'compact.sqlite')
    assert root.exists()


def test_legacy_is_unknown_and_never_guessed_direct(tmp_path,capsys):
    root=capture(tmp_path);target=tmp_path/'compact.sqlite';retention.compact(root,target)
    main(['query','--out',str(target),'--api','torch.add','--call-kind','direct'])
    assert json.loads(capsys.readouterr().out)['tests']==[]
    main(['query','--out',str(target),'--api','torch.add','--call-kind','unknown'])
    assert len(json.loads(capsys.readouterr().out)['tests'])==1


def test_identical_code_foreign_function_is_not_test_body():
    def api():pass
    def body():api()
    foreign=FunctionType(body.__code__,body.__globals__,'foreign',closure=body.__closure__)
    root=SimpleNamespace(api=api)
    registry=Registry({'apis':[{'api_id':'api','qualified_name':'torch.api','language':'python','kind':'function'}]},root)
    kinds=[];collector=Collector(registry,lambda *a:None,lambda *a:None,lambda owner,ids,site,kind:kinds.append(kind))
    collector.install()
    try:
        token=collector.enter(('test_body','call'),describe_test(body))
        foreign();body();collector.leave(token)
    finally:collector.close()
    assert kinds==['indirect','direct']


def test_unknown_identity_is_not_silently_direct():
    from relation_context import call_kind
    import inspect
    token=TEST_FUNCTION.set(None)
    try:assert call_kind(inspect.currentframe(),lambda frame:None)=='unknown'
    finally:TEST_FUNCTION.reset(token)


def test_doctor_preserves_safe_path_environment(tmp_path):
    import os,subprocess,sys
    from api_test_relations.cli import PACKAGE
    env={**os.environ,'PYTHONSAFEPATH':'1','PYTHONPATH':str(PACKAGE.parent)}
    result=subprocess.run([sys.executable,'-m','api_test_relations','doctor'],cwd=tmp_path,
        env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout)['native_adapter']=='built_and_imported'
