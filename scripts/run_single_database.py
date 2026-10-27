"""Single writer, at most four CUDA module workers, verified per-attempt disposal.

This runner consumes an explicitly prepared manifest; it never recreates snapshots.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import time

from single_scheduler_helpers import (read, save, sha, proc, live, resources, can_launch,
                                     launch_spec, attempt_processes, terminate_attempt, cleanup_temp_alias)
from api_test_relations.single_store import completed, append_capture


class AdoptedProcess:
    """Observe a previously launched worker without inventing an exit status."""
    def __init__(self, worker):
        self.worker = worker

    def poll(self):
        return None if live(self.worker) else 0

    def wait(self, timeout=None):
        deadline = time.monotonic() + (timeout if timeout is not None else 10)
        while live(self.worker):
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.worker['command'], timeout)
            time.sleep(.1)
        return 0  # Only process liveness is used; this is not a test outcome.


def source_unavailable_reason(row):
    if row['runner'] == 'doctests':
        return None
    if row['runner'] != 'pytest':
        return 'runner_unavailable:' + str(row['runner'])
    if not row.get('source') or not row.get('sha256'):
        return 'source_unavailable:missing_frozen_python_source'
    if not Path(row['source']).is_file() or sha(row['source']) != row['sha256']:
        return 'source_unavailable:frozen_source_changed_or_missing'
    return None


def record_unavailable(meta, root, row, reason):
    """Retain a module boundary without claiming an unexecuted module was captured."""
    state = reason.partition(':')[0]
    with sqlite3.connect(meta['database'], timeout=60) as db:
        db.execute('PRAGMA foreign_keys=ON'); db.execute('PRAGMA synchronous=FULL')
        mid = db.execute('SELECT id FROM modules WHERE module=?', (row['module'],)).fetchone()[0]
        db.execute('UPDATE modules SET capture_state=? WHERE id=?', (state, mid))
        if not db.execute('SELECT 1 FROM capture_gaps WHERE module_id=? AND test_id IS NULL AND phase=? AND reason=?',
                          (mid, 'collection', reason)).fetchone():
            db.execute('INSERT INTO capture_gaps VALUES (?,?,?,?)', (mid, None, 'collection', reason))
    with Path(meta['database']).open('rb') as stream:
        os.fsync(stream.fileno())
    token = hashlib.sha256(row['module'].encode()).hexdigest()[:16]
    save(root/'receipts'/('unavailable-'+token+'.json'), {
        'module':row['module'], 'state':state, 'reason':reason, 'source':row.get('source'),
        'runner':row['runner'], 'executed':False, 'at':time.time()})
    return state


def open_database(path):
    return sqlite3.connect(Path(path).as_uri() + '?mode=ro', uri=True, timeout=60)


def cleanup_scratch(root, worker):
    """Only the private scratch belonging to an exited, verified attempt."""
    if attempt_processes(worker): raise RuntimeError('Attempt still has live processes')
    base = (root / 'worker-runtime').resolve()
    target = base / worker['attempt']
    if target.is_symlink() or target.resolve().parent != base: raise ValueError('Invalid scratch boundary')
    cleanup_temp_alias(worker, target)
    if target.exists(): shutil.rmtree(target)


def entered_nodes(worker):
    result = set()
    for path in Path(worker['shards']).glob('*.sqlite'):
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='phases'").fetchone():
                result.update(r[0] for r in db.execute("SELECT nodeid FROM phases WHERE state!='closed'"))
    return sorted(result)


def run(root):
    meta = read(root / 'manifest.json')
    lock = (root / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # A shared baseline lock prevents runners in different directories using the same DB.
    baseline_lock = Path(meta['database'] + '.scheduler.lock').open('a')
    fcntl.flock(baseline_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'supervisor.json').exists(): raise RuntimeError('Use explicit reconciliation after any prior start')
    for path, digest in meta['frozen_files'].items():
        if sha(path) != digest: raise RuntimeError('Frozen implementation changed: ' + path)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=meta['repo'],text=True).strip() != meta['commit']:
        raise RuntimeError('PyTorch source changed')
    queue = read(root / 'queue.json')
    workers, handles = {}, {}
    for original in meta.get('adopt_workers', []):
        worker = dict(original)
        job = worker['module']
        if job in workers or len(workers) >= meta['worker_limit']:
            raise RuntimeError('Duplicate/excess adopted workers')
        if not Path(worker['output']).is_dir():
            raise RuntimeError('Adopted raw output missing')
        workers[job] = worker
        handles[job] = AdoptedProcess(worker)
        queue[job] = 'collecting'
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    supervisor = {**proc(os.getpid()), 'started_at':time.time(),
                  'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    save(root/'supervisor.json',supervisor)
    last_launch = max((w['started_at'] for w in workers.values()), default=0)
    last_publish = 0
    counts = dict(meta['starting_counts'])
    try:
        while True:
            now = time.time(); changed = False
            for job, worker in list(workers.items()):
                child = handles[job]; child.poll()
                expired = now-worker['started_at'] >= meta['module_timeout_seconds']
                low_disk = shutil.disk_usage('/mnt/c').free < 2*1024**3
                if stopped or expired or low_disk:
                    worker['interrupted_nodes'] = entered_nodes(worker)
                    worker['terminated_pids'] = terminate_attempt(worker)
                    worker['interrupted'] = True
                    child.wait(timeout=10)
                if live(worker): continue
                worker.setdefault('parent_exited_at',now)
                remaining = attempt_processes(worker)
                if remaining and not worker.get('interrupted') and now-worker['parent_exited_at'] < 120: continue
                if remaining:
                    worker['interrupted_nodes'] = entered_nodes(worker)
                    worker['terminated_pids'] = terminate_attempt(worker); worker['interrupted'] = True
                if attempt_processes(worker): raise RuntimeError('Cannot merge while attempt process remains alive')
                try:
                    boundary = ['checkpoint_barrier_missing:'+str(pid) for pid in worker.get('checkpoint_barrier',{}).get('missing',[])]
                    result = append_capture(Path(worker['output']), Path(meta['database']), job,
                        interrupted=worker.get('interrupted',False), boundary_gaps=boundary,
                        interrupted_nodes=worker.get('interrupted_nodes',()))
                    if result['cleanup'] == 'complete':
                        owner_root = Path(worker.get('runtime_root', root))
                        cleanup_scratch(owner_root,worker)
                        Path(worker['log']).unlink(missing_ok=True)
                        (owner_root/'selections'/(worker['attempt']+'.json')).unlink(missing_ok=True)
                    queue[job] = result['state']
                    save(root/'receipts'/(worker['attempt']+'.json'), {**result,'module':job,'attempt':worker['attempt'],'finished_at':time.time()})
                    with open_database(meta['database']) as db:
                        counts = {k:db.execute(sql).fetchone()[0] for k,sql in {
                            'complete_tests':"SELECT count(*) FROM tests WHERE capture_state='complete'",
                            'tests':'SELECT count(*) FROM tests','observations':'SELECT count(*) FROM observations',
                            'capture_gaps':'SELECT count(*) FROM capture_gaps'}.items()}
                except Exception as exc:
                    queue[job] = 'verification_blocked'
                    save(root/'receipts'/(worker['attempt']+'.json'), {'module':job,'state':'verification_blocked',
                         'error':repr(exc),'worker':worker,'raw_preserved':True,'at':time.time()})
                    # Halt dispatch: unresolved evidence must not accumulate unattended.
                    stopped = True
                workers.pop(job); handles.pop(job); changed = True
            resource = resources()
            resource['host_c_free_gib'] = shutil.disk_usage('/mnt/c').free/1024**3
            resource['database_volume_free_gib'] = shutil.disk_usage(Path(meta['database']).parent).free/1024**3
            if not stopped and len(workers)<meta['worker_limit'] and time.time()-last_launch>=30 and can_launch(resource) and min(resource['host_c_free_gib'],resource['database_volume_free_gib'])>=10:
                row = next((r for r in meta['tests'] if queue.get(r['module'])=='pending'),None)
                if row:
                    job = row['module']
                    unavailable = source_unavailable_reason(row)
                    if unavailable:
                        queue[job]=record_unavailable(meta, root, row, unavailable); changed=True
                        with open_database(meta['database']) as db:
                            counts['capture_gaps'] = db.execute('SELECT count(*) FROM capture_gaps').fetchone()[0]
                    else:
                        attempt = hashlib.sha256(job.encode()).hexdigest()[:16]+'-'+str(time.time_ns())
                        with open_database(meta['database']) as db: retained=sorted(completed(db,job))
                        save(root/'selections'/(attempt+'.json'),{'job':job,'retained':retained})
                        cmd,env,output=launch_spec(meta,root,row,attempt)
                        log=root/'logs'/(attempt+'.log'); log.parent.mkdir(exist_ok=True)
                        with log.open('wb') as stream:
                            child=subprocess.Popen(cmd,cwd=meta['repo'],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                        actual=proc(child.pid)
                        if not actual: raise RuntimeError('Worker vanished before tracking')
                        worker={**actual,'command':cmd,'module':job,'file':row['file'],'attempt':attempt,
                                'output':str(output),'shards':str(output/'shards'),'log':str(log),
                                'started_at':time.time(),'excluded_complete_instances':len(retained),
                                'temp_alias':env.get('RELATION_TEMP_ALIAS')}
                        workers[job]=worker; handles[job]=child; queue[job]='collecting'; last_launch=time.time(); changed=True
            finished = not workers and (stopped or 'pending' not in queue.values())
            if changed or finished or time.time()-last_publish>=15:
                state='paused_for_review' if stopped else ('collection_finished' if finished else ('collecting' if workers else 'waiting_resources'))
                progress={'state':state,'updated_at':time.time(),'worker_limit':meta['worker_limit'],
                    'active_workers':len(workers),'current_modules':list(workers),'queue':{s:list(queue.values()).count(s) for s in set(queue.values())},
                    'resources':resource,'counts':counts,'counts_exclude_inflight':True,'database':meta['database']}
                save(root/'queue.json',queue)
                for target in (root,Path(meta['publish_root'])):
                    save(target/'progress.json',progress)
                    save(target/'current-processes.json',{'updated_at':time.time(),'workers':list(workers.values())})
                last_publish=time.time()
            if finished: break
            time.sleep(2)
    except BaseException as exc:
        save(root/'scheduler-error.json',{'reason':repr(exc),'workers':list(workers.values()),'at':time.time()})
        raise
    finally:
        baseline_lock.close(); lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out.resolve())
