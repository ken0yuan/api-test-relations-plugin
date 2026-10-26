import json
from pathlib import Path
import sqlite3
import pytest
from api_test_relations import single_store
from api_test_relations.cli import read, save
from test_retention import capture

SCHEMA = '''
CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
CREATE TABLE apis(id INTEGER PRIMARY KEY,api_id TEXT UNIQUE,name TEXT,language TEXT,kind TEXT);
CREATE TABLE modules(id INTEGER PRIMARY KEY,module TEXT UNIQUE,file TEXT,capture_state TEXT);
CREATE TABLE tests(id INTEGER PRIMARY KEY,module_id INTEGER REFERENCES modules(id),nodeid TEXT,file TEXT,capture_state TEXT,source_version TEXT,source_attempt TEXT,UNIQUE(module_id,nodeid));
CREATE TABLE phases(id INTEGER PRIMARY KEY,name TEXT UNIQUE);
CREATE TABLE alias_groups(id INTEGER PRIMARY KEY,api_ids_json TEXT UNIQUE);
CREATE TABLE observations(api_key INTEGER REFERENCES apis(id),test_key INTEGER REFERENCES tests(id),phase_key INTEGER REFERENCES phases(id),alias_group_key INTEGER REFERENCES alias_groups(id),PRIMARY KEY(api_key,test_key,phase_key));
CREATE TABLE test_phases(test_id INTEGER REFERENCES tests(id),phase_id INTEGER REFERENCES phases(id),state TEXT,PRIMARY KEY(test_id,phase_id));
CREATE TABLE capture_gaps(module_id INTEGER REFERENCES modules(id),test_id INTEGER REFERENCES tests(id),phase TEXT,reason TEXT);
'''


def baseline(tmp_path):
    target=tmp_path/'single.sqlite'
    with sqlite3.connect(target) as db:
        db.executescript(SCHEMA)
        db.execute('INSERT INTO metadata VALUES (?,?)',('schema',json.dumps('api_test_relations_single_database/1')))
        db.execute("INSERT INTO apis VALUES (1,'a','torch.add','python','function')")
        db.execute("INSERT INTO modules VALUES (1,'one','pytorch/test/test_one.py','pending')")
        db.execute("INSERT INTO tests VALUES (1,1,'pytorch/test/test_one.py::test_retained[zero]','pytorch/test/test_one.py','complete','original','old')")
        db.execute("INSERT INTO capture_gaps VALUES (1,1,'call','historical_gap')")
    return target


def add_phases(root):
    with sqlite3.connect(root/'shards/2147483000.sqlite') as db:
        db.execute('CREATE TABLE phases(nodeid TEXT,phase TEXT,state TEXT)')
        db.executemany('INSERT INTO phases VALUES (?,?,?)',
            [('pytorch/test/test_one.py::test_one[param]',p,'closed') for p in ('setup','call','teardown')])


def test_append_prunes_and_preserves_zero_api_complete_and_gaps(tmp_path):
    target=baseline(tmp_path); root=capture(tmp_path); add_phases(root)
    result=single_store.append_capture(root,target,'one')
    assert result['cleanup']=='complete' and not root.exists()
    with sqlite3.connect(target) as db:
        assert single_store.completed(db,'one')=={'pytorch/test/test_one.py::test_retained[zero]','pytorch/test/test_one.py::test_one[param]'}
        assert db.execute('SELECT source_version,source_attempt FROM tests WHERE id=1').fetchone()==('original','old')
        assert db.execute('SELECT count(*) FROM observations').fetchone()[0]==1
        assert {r[0] for r in db.execute('SELECT reason FROM capture_gaps')}=={'historical_gap','raw_gap_must_remain'}
        assert not db.execute('PRAGMA foreign_key_check').fetchall()


@pytest.mark.parametrize('defect',['completed','lost_prior','transaction'])
def test_failed_append_preserves_baseline_and_raw(tmp_path,monkeypatch,defect):
    target=baseline(tmp_path); root=capture(tmp_path); add_phases(root)
    with sqlite3.connect(target) as db:
        db.execute('INSERT INTO tests VALUES (2,1,?,?,?,?,?)',('pytorch/test/test_one.py::test_one[param]','pytorch/test/test_one.py','complete' if defect=='completed' else 'incomplete','old','old'))
        if defect=='lost_prior':
            db.execute("INSERT INTO phases VALUES (1,'setup')")
            db.execute('INSERT INTO alias_groups VALUES (1,?)',('["a"]',))
            db.execute('INSERT INTO observations VALUES (1,2,1,1)')
        before=list(db.execute('SELECT * FROM tests'))
    if defect=='transaction':
        original=single_store.import_merged
        def fail(*args):
            original(*args)
            raise RuntimeError('Injected transaction failure')
        monkeypatch.setattr(single_store,'import_merged',fail)
    with pytest.raises((ValueError,RuntimeError)):
        single_store.append_capture(root,target,'one')
    assert (root/'shards/2147483001-late.sqlite').exists()
    with sqlite3.connect(target) as db:
        assert list(db.execute('SELECT * FROM tests'))==before


def test_interruption_never_promotes_finally_closed_instance(tmp_path):
    target=baseline(tmp_path); root=capture(tmp_path); add_phases(root)
    node='pytorch/test/test_one.py::test_one[param]'
    single_store.append_capture(root,target,'one',interrupted=True,interrupted_nodes=[node])
    with sqlite3.connect(target) as db:
        assert node not in single_store.completed(db,'one')
        assert db.execute('SELECT count(*) FROM observations').fetchone()[0]==1


def test_four_separate_captures_serially_merge_to_one_database(tmp_path):
    target=baseline(tmp_path)
    for i in range(4):
        parent=tmp_path/str(i);parent.mkdir()
        root=capture(parent);add_phases(root)
        with sqlite3.connect(target) as db:
            db.execute('INSERT INTO modules(module,file,capture_state) VALUES (?,?,?)',(str(i),'pytorch/test/test_one.py','pending'))
        single_store.append_capture(root,target,str(i))
    with sqlite3.connect(target) as db:
        assert db.execute('SELECT count(*) FROM observations').fetchone()[0]==4
        assert db.execute('SELECT count(*) FROM capture_runs').fetchone()[0]==4
        assert len({r[0] for r in db.execute('SELECT test_key FROM observations')})==4


def test_four_real_processes_deselect_retained_then_append_and_cleanup(tmp_path):
    import os,subprocess,sys
    from api_test_relations.cli import PACKAGE
    target=baseline(tmp_path)
    repo=tmp_path/'repo';repo.mkdir();(repo/'torch').mkdir()
    (repo/'torch/__init__.py').write_text('def add(): pass\n')
    (repo/'sitecustomize.py').write_text('import sys\nfrom importlib.machinery import BuiltinImporter,FrozenImporter,PathFinder\nsys.meta_path[:]=[BuiltinImporter,FrozenImporter,PathFinder]\n')
    scripts=PACKAGE.parents[1]/'scripts'
    catalog=tmp_path/'catalog.json'
    save(catalog,{'apis':[{'api_id':'a','qualified_name':'torch.add','language':'python','kind':'function'}]})
    children=[]
    for i in range(4):
        filename=f'test_worker{i}.py';node=f'pytorch/{filename}::test_one[old]'
        (repo/filename).write_text('import pytest,torch,time\n@pytest.mark.parametrize("value",["old","new"])\ndef test_one(value):\n    assert value != "old", "completed body must not run"\n    time.sleep(.3)\n    torch.add()\n')
        with sqlite3.connect(target) as db:
            db.execute('INSERT INTO modules(module,file,capture_state) VALUES (?,?,?)',(str(i),'pytorch/'+filename,'pending'))
            mid=db.execute('SELECT id FROM modules WHERE module=?',(str(i),)).fetchone()[0]
            db.execute('INSERT INTO tests(module_id,nodeid,file,capture_state,source_version) VALUES (?,?,?,?,?)',(mid,node,'pytorch/'+filename,'complete','original'))
        selection=tmp_path/f'selection{i}.json';save(selection,{'job':str(i),'retained':[node]})
        env={k:v for k,v in os.environ.items() if not k.startswith('API_RELATION_')}
        env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONPATH=os.pathsep.join([str(repo),str(PACKAGE.parent),str(scripts)]),RELATION_RESUME_SELECTION=str(selection))
        output=tmp_path/f'run{i}'
        child=subprocess.Popen([sys.executable,'-m','pytest',filename,'-q','-p','api_test_relations.pytest_plugin','-p','retained_selection',
           '--api-relations-repo',str(repo),'--api-relations-out',str(output),'--api-relations-catalog',str(catalog)],cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        children.append((i,child,output,node))
    for i,child,output,node in children:
        stdout=child.communicate(timeout=60)[0]
        assert child.returncode==0,stdout
        assert read(output/'resume-selection.json')['deselected']==[node]
        single_store.append_capture(output,target,str(i))
        assert not output.exists()
    with sqlite3.connect(target) as db:
        assert db.execute('SELECT count(*) FROM observations').fetchone()[0]==4
        assert all(row[0].endswith('[new]') for row in db.execute('SELECT t.nodeid FROM observations o JOIN tests t ON t.id=o.test_key'))
        assert db.execute("SELECT count(*) FROM tests WHERE source_version='original'").fetchone()[0]==5
