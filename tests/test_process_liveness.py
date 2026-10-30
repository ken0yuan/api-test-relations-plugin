from pathlib import Path

import pytest

from api_test_relations import process_liveness
from api_test_relations.cli import save
from api_test_relations.storage import active_children


def fake_proc(tmp_path, monkeypatch, state='S', start_ticks='123'):
    root = tmp_path / 'proc'
    path = root / '42' / 'stat'
    path.parent.mkdir(parents=True)
    fields = [state] + ['0'] * 19
    fields[19] = start_ticks
    # comm can contain spaces and closing parentheses.
    path.write_text('42 (capture ) worker) ' + ' '.join(fields), encoding='utf-8')
    monkeypatch.setattr(process_liveness, '_PROC', root)
    return path


@pytest.mark.parametrize('state,expected', [('R', True), ('S', True), ('D', True), ('T', True), ('Z', False)])
def test_only_exited_producers_allow_post_exit_operations(tmp_path, monkeypatch, state, expected):
    fake_proc(tmp_path, monkeypatch, state=state)
    assert process_liveness.process_may_be_running(42) is expected
    assert process_liveness.process_may_be_running(43) is False


def test_pid_reuse_requires_recorded_start_time(tmp_path, monkeypatch):
    fake_proc(tmp_path, monkeypatch)
    assert process_liveness.process_may_be_running(42, '123')
    assert not process_liveness.process_may_be_running(42, '122')
    assert process_liveness.process_may_be_running(42)


@pytest.mark.parametrize('failure', ['permission', 'malformed'])
def test_unknown_proc_state_remains_a_possible_writer(tmp_path, monkeypatch, failure):
    path = fake_proc(tmp_path, monkeypatch)
    if failure == 'malformed':
        path.write_text('unreadable process state', encoding='utf-8')
    else:
        def denied(*args, **kwargs):
            raise PermissionError('procfs access denied')
        monkeypatch.setattr(Path, 'read_text', denied)
    assert process_liveness.process_may_be_running(42)


@pytest.mark.parametrize('state,start_ticks,expected', [('S', '123', {42}), ('Z', '123', set()), ('S', '122', set())])
def test_child_drain_uses_the_same_exit_and_pid_identity_rules(tmp_path, monkeypatch, state, start_ticks, expected):
    fake_proc(tmp_path, monkeypatch, state=state)
    folder = tmp_path / 'shards'
    (folder / 'launches').mkdir(parents=True)
    save(folder / 'launches/child.json', {'pid': 42, 'start_ticks': start_ticks})
    assert active_children(folder) == expected


def test_real_unreaped_child_is_exited(zombie_process):
    with zombie_process(lambda pid: None) as pid:
        assert Path('/proc', str(pid)).exists()
        assert not process_liveness.process_may_be_running(pid)
