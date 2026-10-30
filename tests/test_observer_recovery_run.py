import json
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_instance_run import prepare
from prepare_recovery_run import recover
from test_scheduler_end_to_end import fixture,launch,wait


def test_recovery_keeps_verified_results_and_recollects_only_affected_cases(tmp_path):
    marker=tmp_path/'good_runs'
    good='import os,torch\nfrom pathlib import Path\ndef test_good():\n p=Path(os.environ["GOOD_MARKER"]);p.write_text((p.read_text() if p.exists() else "")+"run\\n")\n assert torch.add(1,2)==3\n'
    missing='import os,torch\nif os.environ["SIMULATE_GAP"]=="1":raise ImportError("verified missing test dependency")\ndef test_missing():assert torch.add(1,2)==3\n'
    bad='''import os,time,torch
def test_01_crash():
    assert torch.add(1,2)==3
    if os.environ['SIMULATE_GAP']=='1':
        time.sleep(1.2)
        os._exit(7)
def test_observer(request):
    assert torch.add(1,2)==3
    if os.environ['SIMULATE_GAP']=='1':
        state=request.config._relation_state
        state['observer'].gap(('pytorch/test/b.py::test_observer','call'),'profile_hook_lost_during_phase')
'''
    old=fixture(tmp_path,{'test/a.py':good,'test/a2.py':missing,'test/b.py':bad},
                {'test/a.py':['test_good'],'test/a2.py':['test_missing'],'test/b.py':['test_01_crash','test_observer']},
                environment={'SIMULATE_GAP':'1','GOOD_MARKER':str(marker)})
    parent=launch(old);assert parent.wait(timeout=60)==0
    assert json.loads((old/'progress.json').read_text(encoding='utf-8'))['state']=='paused_for_review'
    config=json.loads((tmp_path/'config.json').read_text(encoding='utf-8'))
    config['environment']['SIMULATE_GAP']='0';config['inventories']=[str(tmp_path/'inventory.jsonl')]
    config_file=tmp_path/'fixed.json';config_file.write_text(json.dumps(config),encoding='utf-8')
    validation_dir=tmp_path/'validation';validation_dir.mkdir();valid=validation_dir/'run'
    prepare(valid,config_file,[]);wait(launch(valid),valid)
    proof=validation_dir/'validation.json';proof.write_text(json.dumps({'state':'collection_finished','blocked':[],
        'execution':{'pending':0},'evidence':str(validation_dir),'gaps':[]}),encoding='utf-8')
    new=tmp_path/'recovered';result=recover(old,new,config_file,proof,True)
    assert result['rescheduled']==3
    with sqlite3.connect(old/'relations.sqlite') as db:
        assert db.execute('SELECT count(*) FROM execution_results').fetchone()[0]==4
        previous_relations=db.execute('SELECT count(*) FROM observations').fetchone()[0]
    with sqlite3.connect(new/'relations.sqlite') as db:
        assert db.execute('SELECT count(*) FROM observations').fetchone()[0]+db.execute('SELECT count(*) FROM recovery_quarantine_observations').fetchone()[0]==previous_relations
        assert db.execute('SELECT count(*) FROM recovery_quarantine_observations').fetchone()[0]>0
        assert db.execute('SELECT count(*) FROM execution_result_history').fetchone()[0]==3
        assert db.execute('SELECT count(*) FROM execution_results').fetchone()[0]==1
    wait(launch(new),new)
    assert marker.read_text(encoding='utf-8').splitlines()==['run','run']
    with sqlite3.connect(new/'relations.sqlite') as db:
        assert db.execute("SELECT status,count(*) FROM execution_results GROUP BY status").fetchall()==[('passed',4)]
        assert db.execute('PRAGMA foreign_key_check').fetchall()==[]
