"""Consistent progress counts, verified cleanup and deltas for a prepared run."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from single_scheduler_helpers import live, read, save
from prune_instance_evidence import prune,size


def report(root,cleanup=False):
    root=Path(root).resolve();meta=read(root/'manifest.json')
    cleanup_result=prune(root,True) if cleanup else None
    with sqlite3.connect(Path(meta['database']).as_uri()+'?mode=ro',uri=True,timeout=60) as db:
        db.execute('BEGIN')
        outcomes=dict(db.execute('SELECT status,count(*) FROM execution_results GROUP BY status'))
        planned=db.execute('SELECT count(*) FROM tests').fetchone()[0]
        terminal=sum(outcomes.values())
        started=db.execute("SELECT count(DISTINCT json_extract(data,'$.nodeid')) FROM execution_events WHERE json_extract(data,'$.kind')='test_start'").fetchone()[0]
        dispatched=db.execute('SELECT count(DISTINCT value) FROM execution_attempts,json_each(allowed)').fetchone()[0]
        # A phase/alias group is evidence, not another API-test pair.
        relations=db.execute('SELECT count(*) FROM (SELECT DISTINCT api_key,test_key FROM observations)').fetchone()[0]
        observation_rows=db.execute('SELECT count(*) FROM observations').fetchone()[0]
        apis_observed=db.execute('SELECT count(DISTINCT api_key) FROM observations').fetchone()[0]
        gaps=db.execute('SELECT count(*) FROM capture_gaps').fetchone()[0]
        gap_reasons=dict(db.execute("SELECT CASE WHEN instr(reason,':')>0 THEN substr(reason,1,instr(reason,':')-1) ELSE reason END,count(*) FROM capture_gaps GROUP BY 1 ORDER BY count(*) DESC LIMIT 8"))
        signalled=db.execute('SELECT count(*) FROM execution_attempts WHERE returncode<0').fetchone()[0]
        unavailable={'not_collected','collection_error','worker_error','module_skipped'}
        missing_reasons=[{'status':s,'count':n,'reason':r[:1000]} for s,r,n in db.execute(
            "SELECT status,reason,count(*) FROM execution_results WHERE status IN ('not_collected','collection_error','worker_error','module_skipped') GROUP BY status,reason ORDER BY count(*) DESC LIMIT 8")]
        crashes=[{'status':s,'count':n,'reason':r[:1000]} for s,r,n in db.execute(
            "SELECT status,reason,count(*) FROM execution_results WHERE status IN ('crashed','timeout') GROUP BY status,reason ORDER BY count(*) DESC LIMIT 5")]
        capture_states=dict(db.execute('SELECT capture_state,count(*) FROM execution_attempts GROUP BY capture_state'))
    supervisor=read(root/'supervisor.json');workers=read(root/'current-processes.json')['workers']
    alive=[w for w in workers if live(w)]
    progress=read(root/'progress.json')
    now=time.time();stats={'planned':planned,'dispatched':dispatched,'tests_started':started,'terminal_results':terminal,
        'pending':planned-terminal,'outcomes':outcomes,'crashed':outcomes.get('crashed',0),'timeout':outcomes.get('timeout',0),
        'signalled_attempts':signalled,'unavailable':{s:outcomes.get(s,0) for s in sorted(unavailable)},
        'unavailable_reasons':missing_reasons,'abnormal_reasons':crashes,'relation_pairs':relations,
        'observation_rows':observation_rows,'apis_planned':meta.get('api_inventory_provenance',{}).get('records'),
        'apis_observed':apis_observed,'capture_gaps':gaps,'capture_gap_reasons':gap_reasons,'capture_attempt_states':capture_states,
        'supervisor_alive':live(supervisor),'supervisor_pid':supervisor['pid'],'active_workers':len(alive),
        'device_leases':{w['attempt']:w['devices'] for w in alive},'state':progress['state'],
        'progress_updated_at':progress['updated_at'],'reported_at':now,
        'reported_china':datetime.datetime.fromtimestamp(now,datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'database_bytes':Path(meta['database']).stat().st_size,'preserved_bytes':size(root/'preserved'),
        'cache_bytes':size(root/'worker-runtime'),'log_bytes':size(root/'logs'),
        'disk_free_bytes':shutil.disk_usage(root).free,'run':str(root),'database':meta['database'],
        'cleanup':cleanup_result,'relation_count_definition':'distinct API-test pairs; phases and alias groups counted separately only in observation_rows'}
    reports=root/'periodic-reports';reports.mkdir(exist_ok=True)
    previous=read(reports/'latest.json') if (reports/'latest.json').exists() else None
    if previous:
        stats['delta']={key:stats[key]-previous[key] for key in ['terminal_results','tests_started','relation_pairs','observation_rows','apis_observed','crashed','timeout']}
        stats['since']=previous['reported_at']
    save(reports/(str(time.time_ns())+'.json'),stats);save(reports/'latest.json',stats)
    return stats


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prune',action='store_true');args=parser.parse_args()
    print(json.dumps(report(args.out,args.prune),ensure_ascii=False))
