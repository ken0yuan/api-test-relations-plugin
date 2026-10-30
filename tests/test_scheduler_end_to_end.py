"""Real supervisors/workers with synthetic device probes; no CUDA hardware claim."""
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import pytest
from api_test_relations.cli import PACKAGE

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from prepare_instance_run import prepare


def fixture(tmp_path, sources, inventories, devices=1, **config):
    repo=tmp_path/'repo';(repo/'test').mkdir(parents=True)
    (repo/'torch').mkdir();(repo/'torch/__init__.py').write_text('def add(a,b):return a+b\n',encoding='utf-8')
    (repo/'sitecustomize.py').write_text('import sys\nfrom importlib.machinery import BuiltinImporter,FrozenImporter,PathFinder\nsys.meta_path[:]=[BuiltinImporter,FrozenImporter,PathFinder]\n',encoding='utf-8')
    for file,source in sources.items():(repo/file).write_text(source,encoding='utf-8')
    catalog=tmp_path/'catalog.json'
    catalog.write_text(json.dumps({'apis':[{'api_id':'python:torch.add','qualified_name':'torch.add','language':'python','kind':'function'}]}),encoding='utf-8')
    inventory=tmp_path/'inventory.jsonl'
    inventory.write_text(''.join(json.dumps({'file':file,'name':name})+'\n' for file,names in inventories.items() for name in names),encoding='utf-8')
    probe=[sys.executable,'-c','import json;print(json.dumps('+repr({'devices':[{'id':str(i),'free_mib':8000} for i in range(devices)]})+'))']
    cfg={'backend':'cuda','repo':str(repo),'python':sys.executable,'plugin_root':str(PACKAGE.parent),
         'package_site':'','catalog':str(catalog),'device_probe_command':probe,
         'device_lock_dir':str(tmp_path/'device-locks'),'case_timeout_seconds':60,
         'collection_timeout_seconds':60,'module_timeout_seconds':180,'child_wait_seconds':0,
         'poll_interval_seconds':.05,'worker_start_reserve_mib':64,**config}
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(cfg),encoding='utf-8')
    root=tmp_path/'run';prepare(root,config_path,[str(inventory)])
    return root


def launch(root,resume=False):
    env={k:v for k,v in os.environ.items() if not k.startswith('API_RELATION_')}
    env.pop('CUDA_VISIBLE_DEVICES',None);env.pop('ASCEND_RT_VISIBLE_DEVICES',None)
    env['PYTHONPATH']=str(PACKAGE.parent);env['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
    log=(root/('resume-supervisor.log' if resume else 'supervisor.log')).open('wb')
    command=[sys.executable,str(SCRIPTS/'run_instance_shards.py'),'--out',str(root)]
    if resume:command.append('--resume')
    child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    log.close()
    return child


def wait(child,root):
    try:code=child.wait(timeout=150)
    except subprocess.TimeoutExpired:
        child.terminate();child.wait(timeout=30);pytest.fail((root/'supervisor.log').read_text(encoding='utf-8')[-5000:])
    assert code==0,(root/'supervisor.log').read_text(encoding='utf-8')[-6000:]
    progress=json.loads((root/'progress.json').read_text(encoding='utf-8'))
    assert progress['state']=='collection_finished',[(p.name,p.read_text(encoding='utf-8')) for p in (root/'receipts').glob('*.json')]
    assert progress['execution']['pending']==0
    return progress


def test_all_eight_cards_run_concurrently_then_exclusive_uses_all(tmp_path):
    markers=tmp_path/'markers';markers.mkdir()
    source='''import json,os,time
from pathlib import Path
import pytest,torch
@pytest.mark.parametrize("i",range(8))
def test_parallel(i):
    root=Path(os.environ['SCHEDULER_MARKERS'])
    ids=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    assert len(ids)==1
    (root/(str(i)+'.json')).write_text(json.dumps(dict(pid=os.getpid(),device=ids[0],at=time.time())))
    deadline=time.monotonic()+45
    while len(list(root.glob('*.json')))<8 and time.monotonic()<deadline:time.sleep(.05)
    assert len(list(root.glob('*.json')))==8
    assert torch.add(2,3)==5
    (root/(str(i)+'.done')).write_text(str(time.time()))
'''
    distributed='''import os,torch
from pathlib import Path
def test_all():
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==8
    assert len(list(Path(os.environ['SCHEDULER_MARKERS']).glob('*.done')))==8
    assert torch.add(1,2)==3
'''
    root=fixture(tmp_path,{'test/a.py':source,'test/dist.py':distributed},
                 {'test/a.py':[f'test_parallel[{i}]' for i in range(8)],'test/dist.py':['test_all']},
                 devices=8,cases_per_shard=1,distributed_patterns=['test/dist.py'],
                 environment={'SCHEDULER_MARKERS':str(markers)})
    progress=wait(launch(root),root)
    assert progress['execution']['outcomes']=={'passed':9}
    records=[json.loads(p.read_text(encoding='utf-8')) for p in markers.glob('*.json')]
    assert len({r['pid'] for r in records})==8
    assert {r['device'] for r in records}=={str(i) for i in range(8)}
    with sqlite3.connect(root/'relations.sqlite') as db:
        assignments=[json.loads(r[0]) for r in db.execute('SELECT devices FROM execution_attempts')]
        assert sum(len(ids)==1 for ids in assignments)==8 and sum(len(ids)==8 for ids in assignments)==1
        assert db.execute('SELECT count(*) FROM observations').fetchone()[0]==9


def test_assertion_skip_native_crash_and_remaining_case_are_all_recorded(tmp_path):
    source='''import os,time,pytest,torch
def test_01_pass():assert torch.add(1,2)==3
def test_02_fail():
    torch.add(1,2)
    assert False,"intentional assertion"
def test_03_skip():pytest.skip("explicit hardware condition")
def test_04_crash():
    torch.add(1,2)
    time.sleep(1.2)
    os._exit(7)
def test_05_after():assert torch.add(2,3)==5
'''
    names=['test_01_pass','test_02_fail','test_03_skip','test_04_crash','test_05_after']
    root=fixture(tmp_path,{'test/a.py':source},{'test/a.py':names},cases_per_shard=100)
    progress=wait(launch(root),root)
    assert progress['execution']['outcomes']=={'crashed':1,'failed':1,'passed':2,'skipped':1}
    with sqlite3.connect(root/'relations.sqlite') as db:
        assert db.execute('SELECT count(*) FROM execution_attempts').fetchone()[0]==2
        assert db.execute("SELECT count(*) FROM tests WHERE capture_state='incomplete'").fetchone()[0]==1
        assert db.execute("SELECT count(*) FROM observations o JOIN tests t ON t.id=o.test_key WHERE t.nodeid LIKE '%test_04_crash'").fetchone()[0]==1
        before=list(db.execute('SELECT * FROM execution_results'))
    wait(launch(root,True),root)
    with sqlite3.connect(root/'relations.sqlite') as db:
        assert list(db.execute('SELECT * FROM execution_results'))==before
        assert db.execute('SELECT count(*) FROM execution_attempts').fetchone()[0]==2


def test_collection_error_does_not_discard_the_rest_of_the_queue(tmp_path):
    root=fixture(tmp_path,{'test/a.py':'raise ImportError("intentional unavailable dependency")\n',
                          'test/b.py':'import torch\ndef test_good():assert torch.add(1,2)==3\n'},
                 {'test/a.py':['test_a','test_b'],'test/b.py':['test_good']})
    progress=wait(launch(root),root)
    assert progress['execution']['outcomes']=={'collection_error':2,'passed':1}


def test_supervisor_restart_adopts_the_original_worker(tmp_path):
    marker=tmp_path/'entered';gate=tmp_path/'gate'
    source='''import os,time,torch
from pathlib import Path
def test_wait():
    Path(os.environ['ENTERED']).write_text(str(os.getpid()))
    deadline=time.monotonic()+60
    while not Path(os.environ['GATE']).exists() and time.monotonic()<deadline:time.sleep(.05)
    assert Path(os.environ['GATE']).exists()
    assert torch.add(1,2)==3
'''
    root=fixture(tmp_path,{'test/a.py':source},{'test/a.py':['test_wait']},
                 environment={'ENTERED':str(marker),'GATE':str(gate)})
    parent=launch(root)
    deadline=time.monotonic()+60
    while not marker.exists() and time.monotonic()<deadline:
        if parent.poll() is not None:pytest.fail((root/'supervisor.log').read_text(encoding='utf-8'))
        time.sleep(.05)
    assert marker.exists()
    original_pid=marker.read_text(encoding='utf-8')
    os.kill(parent.pid,signal.SIGKILL);parent.wait(timeout=10)
    resumed=launch(root,True)
    try:
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            progress=json.loads((root/'progress.json').read_text(encoding='utf-8'))
            supervisor=json.loads((root/'supervisor.json').read_text(encoding='utf-8'))
            if supervisor['pid']==resumed.pid and progress['active_workers']==1:break
            if resumed.poll() is not None:pytest.fail((root/'resume-supervisor.log').read_text(encoding='utf-8'))
            time.sleep(.05)
        gate.touch()
        wait(resumed,root)
    finally:
        gate.touch()
        if resumed.poll() is None:resumed.terminate();resumed.wait(timeout=30)
    assert marker.read_text(encoding='utf-8')==original_pid
    with sqlite3.connect(root/'relations.sqlite') as db:
        assert db.execute('SELECT count(*) FROM execution_attempts').fetchone()[0]==1


@pytest.mark.parametrize('block_alarm',[False,True])
def test_timeout_records_current_case_and_runs_the_tail(tmp_path,block_alarm):
    source='''import signal,time,torch
def test_01_timeout():
    if BLOCK:signal.signal(signal.SIGALRM,signal.SIG_IGN)
    time.sleep(15)
def test_02_tail():assert torch.add(1,2)==3
'''.replace('if BLOCK:', 'if '+str(block_alarm)+':')
    root=fixture(tmp_path,{'test/a.py':source},{'test/a.py':['test_01_timeout','test_02_tail']},
                 case_timeout_seconds=2,timeout_grace_seconds=0,cases_per_shard=100)
    progress=wait(launch(root),root)
    assert progress['execution']['outcomes']=={'timeout':1,'passed':1}
    with sqlite3.connect(root/'relations.sqlite') as db:
        count=db.execute('SELECT count(*) FROM execution_attempts').fetchone()[0]
        assert count==(2 if block_alarm else 1)
        assert db.execute("SELECT test_started FROM execution_results WHERE status='timeout'").fetchone()==(1,)


def test_missing_identity_does_not_prevent_existing_cases_running(tmp_path):
    root=fixture(tmp_path,{'test/a.py':'import torch\ndef test_good():assert torch.add(1,2)==3\n'},
                 {'test/a.py':['test_missing','test_good']},cases_per_shard=100)
    progress=wait(launch(root),root)
    assert progress['execution']['outcomes']=={'not_collected':1,'passed':1}
    with sqlite3.connect(root/'relations.sqlite') as db:
        assert db.execute("SELECT test_started FROM execution_results WHERE status='not_collected'").fetchone()==(0,)


def test_restart_after_database_commit_before_receipt_never_appends_twice(tmp_path):
    root=fixture(tmp_path,{'test/a.py':'import torch\ndef test_good():assert torch.add(1,2)==3\n'},
                 {'test/a.py':['test_good']})
    code='''import os,sys
from pathlib import Path
import run_instance_shards as runner
original=runner.append_capture
def append_then_die(*args,**kwargs):
    original(*args,**kwargs)
    os._exit(99)
runner.append_capture=append_then_die
runner.run(Path(sys.argv[1]))
'''
    env={**os.environ,'PYTHONPATH':str(SCRIPTS)+os.pathsep+str(PACKAGE.parent)}
    dead=subprocess.run([sys.executable,'-c',code,str(root)],env=env,capture_output=True,timeout=60)
    assert dead.returncode==99,dead.stderr
    assert not list((root/'receipts').glob('*.json'))
    wait(launch(root,True),root)
    with sqlite3.connect(root/'relations.sqlite') as db:
        assert db.execute('SELECT count(*) FROM execution_attempts').fetchone()[0]==1
        assert db.execute('SELECT count(*) FROM capture_runs').fetchone()[0]==1
        assert db.execute('SELECT count(*) FROM observations').fetchone()[0]==1
        assert db.execute('SELECT capture_state FROM execution_attempts').fetchone()==('recovered_committed',)


@pytest.mark.parametrize('include_compile',[True,False])
def test_api_list_is_exact_and_compile_exclusion_is_explicit(tmp_path,include_compile):
    inventory=tmp_path/'apis.jsonl'
    inventory.write_text(''.join(json.dumps({'api':n})+'\n' for n in ['torch.compile','torch.add','view']),encoding='utf-8')
    root=fixture(tmp_path,{'test/a.py':'import torch\ndef test_compile():assert torch.compile(lambda:torch.add(1,2))()==3\n'},
                 {'test/a.py':['test_compile']},api_inventory=str(inventory),no_default_excludes=include_compile)
    torch_source=tmp_path/'repo/torch/__init__.py'
    torch_source.write_text(torch_source.read_text(encoding='utf-8')+'def compile(f):return f\n',encoding='utf-8')
    wait(launch(root),root)
    with sqlite3.connect(root/'relations.sqlite') as db:
        assert {r[0] for r in db.execute('SELECT name FROM apis')}=={'torch.compile','torch.add','view'}
        observed={r[0] for r in db.execute('SELECT DISTINCT a.name FROM observations o JOIN apis a ON a.id=o.api_key')}
        assert observed==({'torch.compile','torch.add'} if include_compile else {'torch.add'})
        assert db.execute("SELECT state FROM api_binding_states WHERE api_id='python:view'").fetchone()==('unresolved_inventory_name',)
