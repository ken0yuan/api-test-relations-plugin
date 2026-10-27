"""Durability boundaries: interrupted phases, concurrent owners and failed commits."""
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import pytest
from api_test_relations.cli import PACKAGE
from relation_plugin import RelationPlugin


def bare_plugin(tmp_path):
    p=object.__new__(RelationPlugin)
    p.pid=os.getpid();p.repo=tmp_path;p.paths={};p.pending={};p.gaps=set();p.processes={}
    p.lock=threading.RLock();p.registry=SimpleNamespace(status={'a':'bound'});p._saved_registry=None
    p.db=sqlite3.connect(tmp_path/'raw.sqlite',check_same_thread=False)
    p.db.executescript('''CREATE TABLE relations(nodeid,phase,api_id,aliases,site,PRIMARY KEY(nodeid,phase,api_id,aliases));
CREATE TABLE gaps(nodeid,phase,reason,PRIMARY KEY(nodeid,phase,reason));
CREATE TABLE registry(api_id PRIMARY KEY,state);
CREATE TABLE processes(token PRIMARY KEY,nodeid,phase,pid,state);''')
    return p


def test_commit_failure_retains_original_batch_and_first_site(tmp_path):
    p=bare_plugin(tmp_path);real=p.db
    class FailingCommit:
        def __getattr__(self,key):return getattr(real,key)
        def commit(self):raise sqlite3.OperationalError('injected disk failure')
    p.record(('first[param]','call'),('a','alias'),(str(tmp_path/'original.py'),7))
    p.db=FailingCommit()
    with pytest.raises(sqlite3.OperationalError):p.flush()
    assert len(p.pending)==2
    assert real.execute('SELECT count(*) FROM relations').fetchone()[0]==0
    p.record(('first[param]','call'),('a','alias'),(str(tmp_path/'later.py'),8))
    p.db=real;p.flush();p.flush()
    assert real.execute('SELECT count(*) FROM relations').fetchone()[0]==2
    assert json.loads(real.execute('SELECT site FROM relations LIMIT 1').fetchone()[0])==['pytorch/original.py',7]
    assert not p.pending


def test_concurrent_producers_and_commits_preserve_each_owner(tmp_path):
    p=bare_plugin(tmp_path)
    def producer(i):
        for k in range(100):p.record((f'test[{i}]','call'),(f'a{k}',),('test.py',k))
    threads=[threading.Thread(target=producer,args=(i,)) for i in range(4)]
    for t in threads:t.start()
    while any(t.is_alive() for t in threads):p.flush()
    for t in threads:t.join()
    p.flush()
    assert set(p.db.execute('SELECT nodeid,phase,api_id FROM relations'))=={
        (f'test[{i}]','call',f'a{k}') for i in range(4) for k in range(100)}


@pytest.mark.parametrize('mode',['root','spawn_child','fork_pool','barrier'])
def test_live_checkpoint_survives_sigkill_before_phase_end(tmp_path,mode):
    if os.name!='posix':pytest.skip('process-group SIGKILL validation is POSIX')
    repo=tmp_path/'repo';repo.mkdir();(repo/'torch').mkdir();(repo/'test').mkdir()
    (repo/'torch/__init__.py').write_text('def alpha():pass\ndef beta():pass\nalias=alpha\n')
    (repo/'sitecustomize.py').write_text('import sys\nfrom importlib.machinery import BuiltinImporter,FrozenImporter,PathFinder\nsys.meta_path[:]=[BuiltinImporter,FrozenImporter,PathFinder]\n')
    (repo/'taskmod.py').write_text('import torch,time\ndef task():\n torch.alpha()\n time.sleep(90)\n')
    body={
        'root':'torch.alpha();time.sleep(90)',
        'barrier':"torch.alpha()\n while not __import__('pathlib').Path('continue').exists():time.sleep(.01)\n torch.beta()\n __import__('pathlib').Path('beta_done').touch()\n time.sleep(90)",
        'spawn_child':"subprocess.run([sys.executable,'-c','import torch,time;torch.alpha();time.sleep(90)'])",
        'fork_pool':"\n with ProcessPoolExecutor(1,mp_context=multiprocessing.get_context('fork')) as pool:\n  pool.submit(task).result()"
    }[mode]
    (repo/'test/test_long.py').write_text('import torch,time,subprocess,sys,multiprocessing\nfrom concurrent.futures import ProcessPoolExecutor\nfrom taskmod import task\ndef test_long():\n '+body+'\n')
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps({'apis':[
        dict(api_id='python:torch.'+n,qualified_name='torch.'+n,language='python',kind='function') for n in ('alpha','alias','beta')]}))
    out=tmp_path/'capture'
    env={k:v for k,v in os.environ.items() if not k.startswith('API_RELATION_')}
    env.update(PYTHONPATH=os.pathsep.join([str(repo),str(PACKAGE.parent)]),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    command=[sys.executable,'-m','pytest','-p','api_test_relations.pytest_plugin','-q','test',
             '--api-relations-out',str(out),'--api-relations-repo',str(repo),'--api-relations-catalog',str(catalog)]
    with (tmp_path/'run.log').open('w') as log:
        child=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=log,start_new_session=True)
        try:
            deadline=time.monotonic()+35;rows=set()
            while time.monotonic()<deadline:
                rows=set()
                for file in (out/'shards').glob('*.sqlite'):
                    with sqlite3.connect(file.as_uri()+'?mode=ro',uri=True) as db:
                        if db.execute("SELECT 1 FROM sqlite_master WHERE name='relations'").fetchone():
                            rows.update(db.execute('SELECT nodeid,phase,api_id FROM relations'))
                if len(rows)>=2:break
                assert child.poll() is None,(tmp_path/'run.log').read_text()
                time.sleep(.1)
            expected={('pytorch/test/test_long.py::test_long','call','python:torch.'+n) for n in ('alpha','alias')}
            assert rows==expected,(rows,(tmp_path/'run.log').read_text())
            assert child.poll() is None
            if mode=='barrier':
                # The acknowledgement follows a committed batch; new record
                # callbacks must block until the scheduler terminates the process.
                request=out/'shards/checkpoint-request.json'
                request.write_text(json.dumps({'token':'test-final-commit'}))
                state=out/'shards/checkpoints'/(str(child.pid)+'.json')
                until=time.monotonic()+10
                while time.monotonic()<until:
                    status=json.loads(state.read_text())
                    if status.get('request')=='test-final-commit' and status['state']=='quiesced':break
                    time.sleep(.05)
                else:pytest.fail('no final-commit acknowledgement')
                (repo/'continue').touch();time.sleep(.2)
                assert not (repo/'beta_done').exists()
        finally:
            os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        durable=set()
        for file in (out/'shards').glob('*.sqlite'):
            with sqlite3.connect(file) as db:
                assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
                durable.update(db.execute('SELECT nodeid,phase,api_id FROM relations'))
        assert durable==expected
        with sqlite3.connect(out/'shards'/(str(child.pid)+'.sqlite')) as db:
            assert db.execute("SELECT state FROM phases WHERE phase='call'").fetchone()[0]=='entered'
