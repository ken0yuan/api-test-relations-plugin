"""Resume the scheduler's own process failure without rerunning its workers."""
import hashlib
import importlib.util
import sqlite3
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('adoption_scheduler', SCRIPTS / 'run_single_database.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_special_runner_has_a_boundary_without_none_path_or_execution(tmp_path):
    row = {'module':'special', 'runner':'upstream_special_runner', 'source':None, 'sha256':None}
    reason = runner.source_unavailable_reason(row)
    assert reason == 'runner_unavailable:upstream_special_runner'
    target = tmp_path / 'single.sqlite'
    with sqlite3.connect(target) as db:
        db.executescript('CREATE TABLE modules(id INTEGER PRIMARY KEY,module TEXT,capture_state TEXT);'
            'CREATE TABLE tests(id INTEGER PRIMARY KEY,capture_state TEXT,source_version TEXT);'
            'CREATE TABLE capture_gaps(module_id INTEGER REFERENCES modules(id),test_id INTEGER,phase TEXT,reason TEXT);'
            "INSERT INTO modules VALUES(1,'special','pending'); INSERT INTO tests VALUES(1,'complete','0.1.7');")
    for _ in range(2):
        assert runner.record_unavailable({'database':str(target)}, tmp_path, row, reason) == 'runner_unavailable'
    with sqlite3.connect(target) as db:
        assert db.execute('SELECT count(*) FROM capture_gaps').fetchone()[0] == 1
        assert db.execute('SELECT capture_state,source_version FROM tests').fetchall() == [('complete','0.1.7')]


def test_frozen_pytest_source_and_doctest_policy(tmp_path):
    path = tmp_path / 'test_a.py'; path.write_text('def test_a(): pass\n')
    row = {'runner':'pytest','source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    assert runner.source_unavailable_reason(row) is None
    path.write_text('changed')
    assert runner.source_unavailable_reason(row).startswith('source_unavailable:')
    assert runner.source_unavailable_reason({'runner':'doctests','source':None}) is None


def test_adoption_observes_existing_identity_without_starting_process(monkeypatch):
    state = {'live':True}
    worker = {'pid':123,'start_ticks':'4','command':['original-worker']}
    monkeypatch.setattr(runner, 'live', lambda w: w is worker and state['live'])
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a,**kw: (_ for _ in ()).throw(AssertionError('must not relaunch')))
    handle = runner.AdoptedProcess(worker)
    assert handle.poll() is None
    state['live'] = False
    assert handle.poll() == 0
    assert handle.wait(timeout=.1) == 0
