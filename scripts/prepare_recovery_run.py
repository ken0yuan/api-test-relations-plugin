"""Continue a stopped run in a fresh directory, keeping valid results and history."""
import argparse
import fcntl
import json
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_instance_run import prepare
from single_scheduler_helpers import read,save,sha,live,attempt_processes
from instance_sharding import critical_observer_gaps

RECOVERABLE={'trace_guard_bypassed','profile_bypassed_multiplexer',
             'profile_hook_lost_before_phase','profile_hook_lost_during_phase'}


def recover(previous,out,config,validation,retry_collection_errors=False):
    previous=Path(previous).resolve();out=Path(out).resolve();validation=Path(validation).resolve()
    old=read(previous/'manifest.json');proof=read(validation)
    if proof['state']!='collection_finished' or proof.get('blocked') or proof['execution']['pending']:
        raise ValueError('Fix validation did not finish without a blocked capture')
    if any('hook_lost' in r or 'bypassed' in r for r,n in proof.get('gaps',[])):
        raise ValueError('Fix validation still reports observer loss')
    if live(read(previous/'supervisor.json')):raise ValueError('Previous supervisor is still live')
    with (previous/'supervisor.lock').open('a') as own,Path(old['database']+'.scheduler.lock').open('a') as scheduler:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(scheduler,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for path in (previous/'launches').glob('*.json'):
            if attempt_processes(read(path)):raise ValueError('A previous producer is still live')
        for path,digest in old['frozen_files'].items():
            if sha(path)!=digest:raise ValueError('Previous frozen evidence/source changed')
        reviewed=[];retry=set()
        with sqlite3.connect((Path(proof['evidence'])/'run/relations.sqlite').as_uri()+'?mode=ro',uri=True) as valid:
            for path in (previous/'receipts').glob('*.json'):
                receipt=read(path)
                if receipt.get('state')!='verification_blocked':continue
                worker=receipt['worker'];reasons=set()
                for shard in Path(worker['shards']).glob('*.sqlite'):
                    with sqlite3.connect(shard.as_uri()+'?mode=ro',uri=True) as raw:
                        if raw.execute("SELECT 1 FROM sqlite_master WHERE name='gaps'").fetchone():
                            reasons.update(r[0] for r in critical_observer_gaps(raw))
                if not reasons or not reasons<=RECOVERABLE:raise ValueError('Blocked reason outside validated callback fix')
                for node in worker['allowed_nodeids']:
                    row=valid.execute('SELECT t.capture_state,r.test_started FROM tests t JOIN modules m ON m.id=t.module_id '
                        'JOIN execution_results r ON r.module=m.module AND r.nodeid=t.nodeid WHERE m.module=? AND t.nodeid=?',
                        (worker['module'],node)).fetchone()
                    if row!=('complete',1):raise ValueError('Affected identity lacks a complete successful capture validation: '+node)
                    retry.add((worker['module'],node))
                # A preceding crash in the same shard/module can share the
                # observer defect. Retry it only after this exact identity has
                # a complete capture in the fixed validation run.
                with sqlite3.connect(old['database']) as prior:
                    crashed=[r[0] for r in prior.execute("SELECT nodeid FROM execution_results WHERE module=? AND status='crashed'",(worker['module'],))]
                for node in crashed:
                    row=valid.execute('SELECT t.capture_state,r.test_started FROM tests t JOIN modules m ON m.id=t.module_id '
                        'JOIN execution_results r ON r.module=m.module AND r.nodeid=t.nodeid WHERE m.module=? AND t.nodeid=?',
                        (worker['module'],node)).fetchone()
                    if row==('complete',1):retry.add((worker['module'],node))
                reviewed.append({'receipt':receipt,'receipt_sha256':sha(path),'critical_reasons':sorted(reasons)})
        prepared=prepare(out,config,[]);new=read(out/'manifest.json')
        if old['commit']!=new['commit'] or old['inventory_count']!=new['inventory_count']:
            raise ValueError('Recovery changed source revision or inventory count')
        if {r['module']:r['sha256'] for r in old['tests']}!={r['module']:r['sha256'] for r in new['tests']}:
            raise ValueError('Recovery changed test source files')
        validation_meta=read(Path(proof['evidence'])/'run/manifest.json')
        for path,digest in validation_meta['frozen_files'].items():
            path=Path(path);old_package=Path(validation_meta['plugin_root'])
            if path.is_relative_to(old_package):
                target=Path(new['plugin_root'])/path.relative_to(old_package)
                if sha(target)!=digest:raise ValueError('Recovery collector differs from validated collector')
        with sqlite3.connect(old['database']) as source,sqlite3.connect(new['database']) as destination:
            destination.execute('ATTACH DATABASE ? AS previous',(old['database'],))
            checks=[('apis','api_id,name,language,kind'),('tests','nodeid,file')]
            for table,columns in checks:
                for left,right in [('main','previous'),('previous','main')]:
                    if destination.execute('SELECT '+columns+' FROM '+left+'.'+table+' EXCEPT SELECT '+columns+' FROM '+right+'.'+table+' LIMIT 1').fetchone():
                        raise ValueError('Recovery changes registered identities')
            destination.execute('DETACH DATABASE previous')
            source.backup(destination)
            if destination.execute('PRAGMA integrity_check').fetchall()!=[('ok',)]:raise ValueError('Copied database integrity failure')
            if retry_collection_errors:
                retry.update(destination.execute("SELECT module,nodeid FROM execution_results WHERE status='collection_error'"))
            destination.execute('BEGIN IMMEDIATE')
            destination.execute('CREATE TABLE IF NOT EXISTS execution_result_history(module TEXT,nodeid TEXT,status TEXT,reason TEXT,duration REAL,attempt TEXT,test_started INTEGER,call_reported INTEGER,finished REAL,retry_reason TEXT,previous_run TEXT)')
            quarantine_counts={}
            for table,owner in [('observation_evidence','test_key'),('observations','test_key'),('test_phases','test_id'),('capture_gaps','test_id')]:
                destination.execute('CREATE TABLE IF NOT EXISTS recovery_quarantine_'+table+' AS SELECT *,CAST(NULL AS TEXT) AS previous_run FROM '+table+' WHERE 0')
                quarantine_counts[table]=0
            for module,node in sorted(retry):
                state=destination.execute('SELECT t.id,t.capture_state FROM tests t JOIN modules m ON m.id=t.module_id WHERE m.module=? AND t.nodeid=?',(module,node)).fetchone()
                if not state or state[1]=='complete':raise ValueError('Cannot retry or replace completed capture')
                tid=state[0]
                for table,owner in [('observation_evidence','test_key'),('observations','test_key'),('test_phases','test_id'),('capture_gaps','test_id')]:
                    count=destination.execute('SELECT count(*) FROM '+table+' WHERE '+owner+'=?',(tid,)).fetchone()[0]
                    destination.execute('INSERT INTO recovery_quarantine_'+table+' SELECT *,? FROM '+table+' WHERE '+owner+'=?',(str(previous),tid))
                    destination.execute('DELETE FROM '+table+' WHERE '+owner+'=?',(tid,))
                    quarantine_counts[table]+=count
                destination.execute("UPDATE tests SET capture_state='unobserved',source_version=NULL,source_attempt=NULL WHERE id=?",(tid,))
                destination.execute('INSERT INTO execution_result_history SELECT *,?,? FROM execution_results WHERE module=? AND nodeid=?',
                                    ('recollect_after_verified_observer_fix',str(previous),module,node))
                destination.execute('DELETE FROM execution_results WHERE module=? AND nodeid=?',(module,node))
            for reviewed_attempt in reviewed:
                attempt=reviewed_attempt['receipt']['worker']['attempt']
                destination.execute("UPDATE execution_attempts SET capture_state='superseded_observer_failure' WHERE attempt=?",(attempt,))
            detail={'previous_run':str(previous),'previous_database':old['database'],'previous_manifest_sha256':sha(previous/'manifest.json'),
                    'validation':str(validation),'validation_sha256':sha(validation),'reviewed':reviewed,
                    'rescheduled':sorted(retry),'quarantined_counts':quarantine_counts,'new_plugin_root':new['plugin_root'],'created_at':time.time()}
            destination.execute('CREATE TABLE IF NOT EXISTS recovery_reviews(id INTEGER PRIMARY KEY,detail TEXT)')
            destination.execute('INSERT INTO recovery_reviews(detail) VALUES (?)',(json.dumps(detail),));destination.commit()
            for table in ('observations','observation_evidence','capture_runs'):
                before=source.execute('SELECT count(*) FROM '+table).fetchone()[0]
                after=destination.execute('SELECT count(*) FROM '+table).fetchone()[0]
                if before!=after+quarantine_counts.get(table,0):raise ValueError('Existing capture/evidence count changed')
        save(out/'recovery-review.json',detail)
        return {**prepared,'previous_run':str(previous),'rescheduled':len(retry),'blocked_attempts_reviewed':len(reviewed)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('previous','out','config','validation'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--retry-collection-errors',action='store_true');args=parser.parse_args()
    print(json.dumps(recover(args.previous,args.out,args.config,args.validation,args.retry_collection_errors),ensure_ascii=False))
