import json
from pathlib import Path
import sqlite3
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from single_scheduler_helpers import save,read,sha
from prune_instance_evidence import prune,CHECKS
from scheduler_state import create_database
from progress_report import report


def prepared(tmp_path):
    root=tmp_path/'run';root.mkdir();database=root/'relations.sqlite'
    with sqlite3.connect(database) as db:db.execute('CREATE TABLE capture_runs(attempt TEXT,module TEXT,detail TEXT)')
    save(root/'manifest.json',{'database':str(database)})
    attempt='example';source=root/'attempts'/attempt;archive=root/'preserved'/attempt
    (archive/'capture').mkdir(parents=True);(archive/'capture/evidence').write_text('positive',encoding='utf-8')
    hashes={'evidence':sha(archive/'capture/evidence')}
    scratch=root/'worker-runtime'/attempt;scratch.mkdir(parents=True);(scratch/'compiled').write_bytes(b'cache')
    logs=root/'logs';logs.mkdir();(logs/'example.log').write_text('diagnostic',encoding='utf-8')
    worker={'attempt':attempt,'output':str(source),'module':'test/a.py','pid':-1,'start_ticks':'none'}
    proof={'source':str(source),'module':worker['module'],'source_hashes':hashes,'checks':sorted(CHECKS)}
    with sqlite3.connect(database) as db:db.execute('INSERT INTO capture_runs VALUES (?,?,?)',(str(source),worker['module'],json.dumps(proof)))
    save(archive/'provenance.json',{'source':str(source),'worker':worker,'source_hashes':hashes})
    save(root/'receipts/example.json',{'state':'captured','worker':worker})
    return root,archive,scratch,source


def test_only_committed_verified_archive_and_private_cache_are_removed(tmp_path):
    root,archive,scratch,_=prepared(tmp_path)
    formal=(root/'relations.sqlite').read_bytes()
    dry=prune(root);assert len(dry['candidates'])==1 and not dry['removed']
    assert archive.exists() and scratch.exists()
    actual=prune(root,True)
    assert not actual['errors'] and len(actual['removed'])==1 and actual['bytes_freed']>0
    assert not archive.exists() and not scratch.exists()
    assert (root/'logs/example.log').read_text(encoding='utf-8')=='diagnostic'
    assert (root/'relations.sqlite').read_bytes()==formal
    assert read(root/'retention-receipts/example.json')['state']=='removed'
    assert not prune(root,True)['removed']


def test_uses_the_append_inventory_contract_for_sqlite_sidecars(tmp_path):
    root,archive,scratch,_=prepared(tmp_path)
    (archive/'capture/relations.sqlite-wal').write_bytes(b'')
    (archive/'capture/relations.sqlite-shm').write_bytes(b'ephemeral SQLite shared-memory metadata')
    provenance=read(archive/'provenance.json')
    provenance['source_hashes']={p.name:sha(p) for p in (archive/'capture').iterdir() if p.is_file()}
    save(archive/'provenance.json',provenance)
    result=prune(root,True)
    assert not result['errors'] and len(result['removed'])==1


@pytest.mark.parametrize('condition',['tampered','uncommitted','blocked','active','original','symlink'])
def test_unverified_or_live_evidence_is_never_removed(tmp_path,monkeypatch,condition):
    root,archive,scratch,source=prepared(tmp_path)
    if condition=='tampered':(archive/'capture/evidence').write_text('changed',encoding='utf-8')
    elif condition=='uncommitted':
        with sqlite3.connect(root/'relations.sqlite') as db:db.execute('DELETE FROM capture_runs')
    elif condition=='blocked':
        receipt=read(root/'receipts/example.json');receipt['state']='verification_blocked';save(root/'receipts/example.json',receipt)
    elif condition=='active':monkeypatch.setattr('prune_instance_evidence.attempt_processes',lambda worker:{123:{}})
    elif condition=='original':source.mkdir(parents=True)
    elif condition=='symlink':
        outside=tmp_path/'outside';outside.write_text('keep',encoding='utf-8')
        (archive/'capture/link').symlink_to(outside)
    result=prune(root,True)
    assert not result['removed'] and (result['retained'] or result['errors'])
    assert archive.exists() and scratch.exists()
    if condition=='symlink':assert outside.read_text(encoding='utf-8')=='keep'


def test_periodic_progress_counts_unique_api_test_pairs_and_deltas(tmp_path):
    root=tmp_path/'run';root.mkdir();database=root/'relations.sqlite'
    create_database(database,{'apis':[{'api_id':'python:torch.add','qualified_name':'torch.add','language':'python','kind':'function'}]},
                    [{'module':'test/a.py','file':'pytorch/test/a.py','nodeids':['pytorch/test/a.py::test_a']}])
    with sqlite3.connect(database) as db:
        db.executemany('INSERT INTO phases VALUES (?,?)',[(1,'setup'),(2,'call')]);db.execute("INSERT INTO alias_groups VALUES (1,'[]')")
        db.executemany('INSERT INTO observations VALUES (1,1,?,1)',[(1,),(2,)])
        db.execute('INSERT INTO execution_attempts VALUES (?,?,?,?,?,?,?,?)',('one','test/a.py',json.dumps(['pytorch/test/a.py::test_a']),'["0"]',1,2,0,'captured'))
        db.execute('INSERT INTO execution_events VALUES (?,?,?)',('one',0,json.dumps({'kind':'test_start','nodeid':'pytorch/test/a.py::test_a'})))
        db.execute('INSERT INTO execution_results VALUES (?,?,?,?,?,?,?,?,?)',('test/a.py','pytorch/test/a.py::test_a','passed','',.1,'one',1,1,2))
        db.executemany('INSERT INTO capture_gaps VALUES (1,1,?,?)',[('call','external:a'),('call','external:b'),('call','other')])
    save(root/'manifest.json',{'database':str(database),'api_inventory_provenance':{'records':1}})
    save(root/'supervisor.json',{'pid':-1,'start_ticks':'none'});save(root/'current-processes.json',{'workers':[]})
    save(root/'progress.json',{'state':'collection_finished','updated_at':2})
    first=report(root)
    assert first['relation_pairs']==1 and first['observation_rows']==2
    assert first['tests_started']==1 and first['terminal_results']==1 and first['pending']==0
    assert first['capture_gap_reasons']=={'external':2,'other':1}
    second=report(root)
    assert second['delta']['relation_pairs']==0 and second['delta']['terminal_results']==0
