import json
import os
import sqlite3
import pytest
from api_test_relations import retention
from api_test_relations.cli import main, read, save
from test_settle import fixture


def capture(tmp_path):
    root = fixture(tmp_path / 'run')
    save(root / 'manifest.json', {**read(root / 'manifest.json'),
         'schema': 'api_test_relations_plugin/1', 'repo': str(tmp_path / 'repo')})
    return root


def test_verified_cleanup_keeps_late_child_gap_and_parameter_identity(tmp_path, capsys):
    root = capture(tmp_path)
    target = tmp_path / 'relations.sqlite'
    result = retention.compact(root, target)
    assert result['cleanup'] == 'complete' and result['relations'] == 1
    assert not root.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ['relations.sqlite']
    with sqlite3.connect(target) as db:
        assert db.execute('PRAGMA integrity_check').fetchall() == [('ok',)]
        assert not db.execute('PRAGMA foreign_key_check').fetchall()
        row = db.execute('SELECT nodeid,api_id,aliases,site FROM relations').fetchone()
        assert row == ('pytorch/test/test_one.py::test_one[param]', 'a', '["a"]', '["pytorch/test/test_one.py", 9]')
        assert db.execute('SELECT reason FROM gaps').fetchone() == ('raw_gap_must_remain',)
        receipt = json.loads(db.execute('SELECT detail FROM retention_receipt').fetchone()[0])
        assert receipt['cleanup'] == 'complete'
        assert receipt['proves_zero_omissions'] is False
    main(['query', '--out', str(target), '--api', 'torch.add'])
    assert json.loads(capsys.readouterr().out)['tests'][0]['nodeid'].endswith('[param]')


@pytest.mark.parametrize('defect', ['unknown_api', 'missing_owner', 'conflict', 'bad_path', 'live', 'unknown_file', 'symlink'])
def test_invalid_capture_never_deleted(tmp_path, defect):
    root = capture(tmp_path)
    child = root / 'shards/2147483001-late.sqlite'
    with sqlite3.connect(child) as db:
        if defect == 'unknown_api': db.execute("UPDATE relations SET api_id='unknown'")
        if defect == 'missing_owner': db.execute("UPDATE relations SET nodeid='pytorch/test/missing.py::test_x'")
        if defect == 'bad_path': db.execute("UPDATE tests SET file='wrong.py'")
    if defect == 'conflict':
        with sqlite3.connect(root / 'shards/2147483000.sqlite') as db:
            db.execute('INSERT INTO relations VALUES (?,?,?,?,?)',
                       ('pytorch/test/test_one.py::test_one[param]', 'call', 'a', '[]', '[]'))
    if defect == 'live': save(root / 'manifest.json', {**read(root / 'manifest.json'), 'capture_pid': os.getpid()})
    if defect == 'unknown_file': (root / 'personal.txt').write_text('keep me')
    if defect == 'symlink': (root / 'runtime').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises((ValueError, RuntimeError)):
        retention.compact(root, tmp_path / 'result.sqlite')
    assert child.exists() and not (tmp_path / 'result.sqlite').exists()


@pytest.mark.parametrize('failure', ['pack', 'late_change'])
def test_verification_failure_preserves_raw(tmp_path, monkeypatch, failure):
    root = capture(tmp_path)
    original = retention._pack
    def injected(*args):
        if failure == 'pack': raise sqlite3.OperationalError('simulated storage failure')
        original(*args)
        (root / 'progress.json').write_text('{}')
    monkeypatch.setattr(retention, '_pack', injected)
    with pytest.raises((RuntimeError, sqlite3.OperationalError)):
        retention.compact(root, tmp_path / 'result.sqlite')
    assert (root / 'shards/2147483001-late.sqlite').exists()
    assert not (tmp_path / 'result.sqlite').exists()


def test_destination_protection(tmp_path):
    root = capture(tmp_path)
    existing = tmp_path / 'existing.sqlite'
    existing.write_bytes(b'keep')
    for target in (existing, root / 'nested.sqlite'):
        with pytest.raises(ValueError): retention.compact(root, target)
    assert existing.read_bytes() == b'keep'
    assert root.exists()


def test_compact_accepts_an_exited_unreaped_owner_and_keeps_relations(tmp_path, zombie_process):
    root = capture(tmp_path)
    with zombie_process(lambda pid: None) as pid:
        save(root / 'manifest.json', {**read(root / 'manifest.json'), 'capture_pid': pid})
        target = tmp_path / 'relations.sqlite'
        result = retention.compact(root, target)
        assert result['cleanup'] == 'complete' and result['relations'] == 1
        with sqlite3.connect(target) as db:
            assert db.execute('SELECT reason FROM gaps').fetchone() == ('raw_gap_must_remain',)
        assert not root.exists()


def test_cleanup_failure_keeps_verified_database_and_reports_partial_cleanup(tmp_path, monkeypatch):
    root = capture(tmp_path)
    original = retention.shutil.rmtree
    def fail_source(path, *args, **kwargs):
        if path == root: raise OSError('simulated permission error')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(retention.shutil, 'rmtree', fail_source)
    result = retention.compact(root, tmp_path / 'result.sqlite')
    assert result['cleanup'] == 'incomplete' and root.exists()
    with sqlite3.connect(tmp_path / 'result.sqlite') as db:
        assert db.execute('SELECT count(*) FROM relations').fetchone()[0] == 1
        assert json.loads(db.execute('SELECT detail FROM retention_receipt').fetchone()[0])['cleanup'] == 'incomplete'
