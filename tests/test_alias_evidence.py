"""0.1.8 regression: real process-local aliases are evidence, not conflicts."""
from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import sqlite3
import pytest
from api_test_relations.cli import read, save
from api_test_relations.storage import connect, merge
from api_test_relations import retention, single_store
from test_retention import capture
from test_single_store import baseline, add_phases
from test_plugin import launch


def divergent_capture(tmp_path, reverse=False, duplicate=False):
    root=capture(tmp_path);add_phases(root)
    cat=read(root/'api-catalog.json')
    cat['apis'].append(dict(api_id='b',qualified_name='torch.random.add',language='python',kind='function'))
    save(root/'api-catalog.json',cat)
    node='pytorch/test/test_one.py::test_one[param]'
    parent=root/'shards/2147483000.sqlite';child=root/'shards/2147483001-late.sqlite'
    with sqlite3.connect(parent) as db:
        db.execute('INSERT INTO relations VALUES (?,?,?,?,?)',(node,'call','a','["a"]','["parent.py", 1]'))
    with sqlite3.connect(child) as db:
        db.execute('UPDATE relations SET aliases=?,site=?',('["a", "b"]','["child.py", 2]'))
        db.execute('INSERT INTO relations VALUES (?,?,?,?,?)',(node,'call','b','["a", "b"]','["child.py", 2]'))
    if duplicate:shutil.copyfile(child,root/'shards/2147483002-duplicate.sqlite')
    if reverse:
        tmp=parent.with_suffix('.swap');parent.rename(tmp);child.rename(parent);tmp.rename(child)
    return root


def merge_capture(root,target):
    target.mkdir();db=connect(target)
    db.executemany('INSERT INTO apis VALUES (?,?,?,?)',[(a['api_id'],a['qualified_name'],a['language'],a['kind']) for a in read(root/'api-catalog.json')['apis']])
    merge(db,'pytest',root/'shards',final=True)
    return db


@pytest.mark.parametrize('reverse',[False,True])
def test_aliases_order_repeat_and_duplicate_shards(tmp_path,reverse):
    root=divergent_capture(tmp_path,reverse,duplicate=True)
    db=merge_capture(root,tmp_path/'merged')
    try:
        expected={('a','["a"]','["parent.py", 1]'),('a','["a", "b"]','["child.py", 2]'),('b','["a", "b"]','["child.py", 2]')}
        assert set(db.execute('SELECT api_id,aliases,site FROM relations'))==expected
        assert db.execute('SELECT count(*) FROM relation_sources').fetchone()[0]==5
        merge(db,'pytest',root/'shards',final=True)
        assert set(db.execute('SELECT api_id,aliases,site FROM relations'))==expected
        assert db.execute('SELECT count(*) FROM relation_sources').fetchone()[0]==5
        retention._raw_check(db,root)
    finally:db.close()


def test_compact_preserves_groups_sites_and_process_sources(tmp_path):
    root=divergent_capture(tmp_path,duplicate=True);target=tmp_path/'packed.sqlite'
    retention.compact(root,target)
    with sqlite3.connect(target) as db:
        assert db.execute('SELECT count(*) FROM relations').fetchone()[0]==3
        assert db.execute('SELECT count(*) FROM relation_sources').fetchone()[0]==5
        assert set(db.execute("SELECT aliases FROM relations WHERE api_id='a'"))=={('["a"]',),('["a", "b"]',)}
        assert db.execute('PRAGMA foreign_key_check').fetchall()==[]


def test_append_migrates_old_key_and_keeps_all_evidence(tmp_path):
    target=baseline(tmp_path);root=divergent_capture(tmp_path)
    with sqlite3.connect(target) as db:db.execute("INSERT INTO apis VALUES (2,'b','torch.random.add','python','function')")
    single_store.append_capture(root,target,'one')
    with sqlite3.connect(target) as db:
        assert json.loads(db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()[0]).endswith('/2')
        assert db.execute('SELECT count(*) FROM observations').fetchone()[0]==3
        assert db.execute('SELECT count(*) FROM observation_evidence').fetchone()[0]==3
        assert db.execute('PRAGMA foreign_key_check').fetchall()==[]
        assert db.execute('SELECT capture_state FROM tests WHERE id=1').fetchone()==('complete',)


def test_schema_migration_rolls_back_on_append_failure(tmp_path,monkeypatch):
    target=baseline(tmp_path);root=divergent_capture(tmp_path)
    with sqlite3.connect(target) as db:db.execute("INSERT INTO apis VALUES (2,'b','torch.random.add','python','function')")
    original=single_store.import_merged
    def fail(*args):original(*args);raise RuntimeError('post-migration failure')
    monkeypatch.setattr(single_store,'import_merged',fail)
    with pytest.raises(RuntimeError,match='post-migration'):
        single_store.append_capture(root,target,'one')
    assert root.exists()
    with sqlite3.connect(target) as db:
        assert json.loads(db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()[0]).endswith('/1')
        assert sum(r[5]>0 for r in db.execute('PRAGMA table_info(observations)'))==3


def test_existing_summary_migration_rebuilds_from_raw(tmp_path):
    root=divergent_capture(tmp_path);folder=tmp_path/'old';folder.mkdir()
    with sqlite3.connect(folder/'relations.sqlite') as db:
        db.execute('CREATE TABLE relations(job TEXT,nodeid TEXT,phase TEXT,api_id TEXT,aliases TEXT,site TEXT,PRIMARY KEY(job,nodeid,phase,api_id))')
        db.execute('INSERT INTO relations VALUES (?,?,?,?,?,?)',('pytest','pytorch/test/test_one.py::test_one[param]','call','a','["a"]','["parent.py", 1]'))
    db=connect(folder)
    db.executemany('INSERT INTO apis VALUES (?,?,?,?)',[('a','torch.add','python','function'),('b','torch.random.add','python','function')])
    merge(db,'pytest',root/'shards',final=True)
    retention._raw_check(db,root)
    assert db.execute('SELECT count(*) FROM relations').fetchone()[0]==3
    db.close()


def test_summary_schema_migration_failure_is_atomic(tmp_path,monkeypatch):
    from api_test_relations import storage
    raw_connect=sqlite3.connect
    with raw_connect(tmp_path/'relations.sqlite') as db:
        db.execute('CREATE TABLE relations(job,nodeid,phase,api_id,aliases,site,PRIMARY KEY(job,nodeid,phase,api_id))')
        db.execute("INSERT INTO relations VALUES ('j','n','call','a','[\"a\"]','[]')")
    class FailingConnection:
        def __init__(self,*a,**kw):self.real=raw_connect(*a,**kw)
        def __getattr__(self,key):return getattr(self.real,key)
        def __enter__(self):self.real.__enter__();return self
        def __exit__(self,*args):return self.real.__exit__(*args)
        def execute(self,sql,*args):
            if sql=='DROP TABLE relations':raise RuntimeError('injected migration failure')
            return self.real.execute(sql,*args)
    monkeypatch.setattr(storage.sqlite3,'connect',FailingConnection)
    with pytest.raises(RuntimeError,match='migration failure'):storage.connect(tmp_path)
    with raw_connect(tmp_path/'relations.sqlite') as db:
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='relations_v018'").fetchone()
        assert db.execute('SELECT count(*) FROM relations').fetchone()[0]==1
        assert sum(r[5]>0 for r in db.execute('PRAGMA table_info(relations)'))==4


def test_missing_source_provenance_is_rejected(tmp_path):
    root=divergent_capture(tmp_path);db=merge_capture(root,tmp_path/'merged')
    db.execute('DELETE FROM relation_sources');db.commit()
    with pytest.raises(ValueError,match='source missing'):retention._raw_check(db,root)
    with pytest.raises(ValueError,match='source missing'):single_store.verify_raw(db,root,set())
    db.close()


def test_equal_alias_group_with_different_sites_retains_both(tmp_path):
    root=divergent_capture(tmp_path)
    with sqlite3.connect(root/'shards/2147483000.sqlite') as db:
        db.execute('UPDATE relations SET aliases=?',('["a", "b"]',))
    db=merge_capture(root,tmp_path/'merged')
    assert set(db.execute("SELECT site FROM relations WHERE api_id='a'"))=={('["parent.py", 1]',),('["child.py", 2]',)}
    retention._raw_check(db,root);db.close()


def real_capture(parent):
    repo=parent/'repo';(repo/'test').mkdir(parents=True);(repo/'torch').mkdir()
    (repo/'torch/random.py').write_text('def manual_seed(seed): return seed\n')
    (repo/'torch/__init__.py').write_text('from . import random\nmanual_seed=random.manual_seed\n')
    (repo/'sitecustomize.py').write_text('import sys\nfrom importlib.machinery import BuiltinImporter,FrozenImporter,PathFinder\nsys.meta_path[:]=[BuiltinImporter,FrozenImporter,PathFinder]\n')
    (repo/'test/test_alias.py').write_text('''import subprocess,sys,torch
def test_parent_and_child():
    original=torch.manual_seed
    torch.manual_seed=lambda seed: seed
    torch.manual_seed(1)
    subprocess.run([sys.executable,'-c','import torch;torch.manual_seed(2)'],check=True)
    torch.manual_seed=original
    torch.manual_seed(3)
''')
    catalog=parent/'catalog.json'
    catalog.write_text(json.dumps({'apis':[dict(api_id='python:'+n,qualified_name=n,language='python',kind='function') for n in ('torch.manual_seed','torch.random.manual_seed')]}))
    out=parent/'capture'
    launch(repo,'--api-relations-out',str(out),'--api-relations-repo',str(repo),'--api-relations-catalog',str(catalog))
    with sqlite3.connect(out/'relations.sqlite') as db:
        groups={tuple(json.loads(r[0])) for r in db.execute("SELECT aliases FROM relations WHERE api_id='python:torch.manual_seed' AND phase='call'")}
        assert groups=={('python:torch.manual_seed',),('python:torch.manual_seed','python:torch.random.manual_seed')}
        assert db.execute('SELECT count(*) FROM relation_sources').fetchone()[0]>=4
        assert db.execute('SELECT count(*) FROM gaps').fetchone()[0]==0
        retention._raw_check(db,out)
    return out


def test_four_independent_pytest_workers_with_real_children(tmp_path):
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs=list(pool.map(real_capture,[tmp_path/str(i) for i in range(4)]))
    assert len(outputs)==4
    for i,out in enumerate(outputs):
        result=retention.compact(out,tmp_path/f'worker-{i}.sqlite')
        assert result['relations']>=3
