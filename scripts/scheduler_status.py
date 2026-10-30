"""Read test verdicts and capture completeness without changing a run."""
import argparse
import json
from pathlib import Path
import sqlite3


def status(root, export=None):
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    with sqlite3.connect(Path(manifest['database']).as_uri()+'?mode=ro', uri=True) as db:
        planned = db.execute('SELECT count(*) FROM tests').fetchone()[0]
        outcomes = dict(db.execute('SELECT status,count(*) FROM execution_results GROUP BY status'))
        result = {'planned': planned, 'terminal_results': sum(outcomes.values()),
                  'pending': planned-sum(outcomes.values()), 'outcomes': outcomes,
                  'capture_complete': db.execute("SELECT count(*) FROM tests WHERE capture_state='complete'").fetchone()[0],
                  'capture_attempt_states': dict(db.execute('SELECT capture_state,count(*) FROM execution_attempts GROUP BY capture_state')),
                  'capture_gaps': db.execute('SELECT count(*) FROM capture_gaps').fetchone()[0]}
        result['apis_planned']=db.execute('SELECT count(*) FROM apis').fetchone()[0]
        result['apis_observed']=db.execute('SELECT count(DISTINCT api_key) FROM observations').fetchone()[0]
        result['api_binding_states']=dict(db.execute('SELECT state,count(*) FROM api_binding_states GROUP BY state'))
        if export:
            with Path(export).open('w',encoding='utf-8') as output:
                columns = ['module','nodeid','status','reason','duration','attempt','test_started','call_reported','finished']
                for row in db.execute('SELECT '+','.join(columns)+' FROM execution_results ORDER BY module,nodeid'):
                    output.write(json.dumps(dict(zip(columns,row)),ensure_ascii=False)+'\n')
    if (root/'progress.json').exists():
        progress = json.loads((root/'progress.json').read_text(encoding='utf-8'))
        result.update(state=progress['state'], active_workers=progress['active_workers'],
                      devices=progress.get('devices'), device_leases=progress.get('device_leases'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--results-jsonl',type=Path)
    args = parser.parse_args()
    print(json.dumps(status(args.out,args.results_jsonl),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
