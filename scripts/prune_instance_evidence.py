"""Remove only stopped, verified captures already committed to the formal DB."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from single_scheduler_helpers import attempt_processes, cleanup_temp_alias, read, save, sha
from api_test_relations.retention import _inventory

STATES={'captured','capture_incomplete','interrupted','recovered_committed'}
CHECKS={'raw_mapping_and_owner_match','destination_mapping_and_phases_equal',
        'prior_positive_subset','completed_instances_preserved','foreign_key_check','integrity_check'}


def scoped(root,path):
    path=Path(path)
    if path.is_symlink() or not path.resolve().is_relative_to(root) or path.resolve()==root:
        raise ValueError('Cleanup path escapes its run: '+str(path))
    return path


def files(path):
    result={}
    for item in path.rglob('*'):
        if item.is_symlink():raise ValueError('Symlink in preserved capture')
        if item.is_file():result[str(item.relative_to(path))]=sha(item)
    return result


def size(path):
    total=0
    for item in path.rglob('*'):
        try:
            if item.is_file() and not item.is_symlink():total+=item.stat().st_size
        except FileNotFoundError:pass  # Live compiler caches may rename temporary files.
    return total


def prune(root,apply=False):
    root=Path(root).resolve();manifest=read(root/'manifest.json')
    database=Path(manifest['database']).resolve()
    if not database.is_relative_to(root):raise ValueError('Formal database must belong to this run')
    audit=root/'retention-receipts';audit.mkdir(exist_ok=True)
    result={'apply':apply,'removed':[],'candidates':[],'retained':[],'errors':[],'bytes_freed':0}
    with (root/'evidence-prune.lock').open('a') as lock,sqlite3.connect(database.as_uri()+'?mode=ro',uri=True,timeout=30) as db:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if db.execute('PRAGMA quick_check').fetchall()!=[('ok',)]:raise ValueError('Formal database quick_check failed')
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='capture_runs'").fetchone():return result
        committed={source:json.loads(detail) for source,detail in db.execute('SELECT attempt,detail FROM capture_runs')}
        for receipt_path in sorted((root/'receipts').glob('*.json')):
            attempt=receipt_path.stem
            try:
                receipt=read(receipt_path);worker=receipt.get('worker',{})
                archive=root/'preserved'/attempt;scratch=root/'worker-runtime'/attempt
                marker=audit/(attempt+'.json')
                if marker.exists() and read(marker).get('state')=='removed':continue
                if receipt.get('state') not in STATES:
                    result['retained'].append({'attempt':attempt,'reason':receipt.get('state','unverified')});continue
                if not worker or worker.get('attempt')!=attempt:raise ValueError('Attempt identity mismatch')
                source=scoped(root,root/'attempts'/attempt)
                if worker['output']!=str(source):raise ValueError('Source capture ownership mismatch')
                proof=committed.get(str(source))
                if proof is None:
                    fresh=db.execute('SELECT detail FROM capture_runs WHERE attempt=?',(str(source),)).fetchone()
                    if fresh:proof=json.loads(fresh[0])
                if not proof or proof['module']!=worker['module'] or not CHECKS<=set(proof.get('checks',[])):
                    raise ValueError('No complete committed verification receipt')
                if source.exists():
                    result['retained'].append({'attempt':attempt,'reason':'original capture cleanup not yet complete'});continue
                if attempt_processes(worker):
                    result['retained'].append({'attempt':attempt,'reason':'producer still alive'});continue
                scoped(root,archive);scoped(root,scratch)
                if not archive.exists():
                    if not marker.exists():raise ValueError('Preserved evidence missing without a retention receipt')
                else:
                    provenance=read(archive/'provenance.json')
                    if provenance['source']!=str(source) or provenance['worker']['attempt']!=attempt:
                        raise ValueError('Preserved archive ownership mismatch')
                    actual=files(archive/'capture')
                    if actual!=provenance['source_hashes']:raise ValueError('Preserved archive changed')
                    if _inventory(archive/'capture')!=proof['source_hashes']:
                        raise ValueError('Archive differs from committed source hashes')
                amount=size(archive)+size(scratch)
                entry={'attempt':attempt,'bytes':amount,'archive':str(archive),'scratch':str(scratch)}
                if apply:
                    save(marker,{**entry,'state':'verified_for_removal','at':time.time(),
                                 'source_capture':str(source),'source_hashes':proof['source_hashes'],
                                 'checks':proof['checks'],'formal_database':str(database),
                                 'logs_and_execution_events_retained':True})
                    # Recheck producers immediately before deleting derived files.
                    if attempt_processes(worker):raise ValueError('Producer appeared during cleanup')
                    cleanup_temp_alias(worker,scratch)
                    if archive.exists():shutil.rmtree(archive)
                    if scratch.exists():shutil.rmtree(scratch)
                    save(marker,{**read(marker),'state':'removed','removed_at':time.time()})
                    result['bytes_freed']+=amount
                result['removed' if apply else 'candidates'].append(entry)
            except (OSError,ValueError,KeyError,sqlite3.Error) as exc:
                result['errors'].append({'attempt':attempt,'reason':repr(exc)})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();print(json.dumps(prune(args.out,args.apply),ensure_ascii=False,indent=2))
