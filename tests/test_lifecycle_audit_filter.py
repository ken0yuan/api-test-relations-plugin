import pytest
from _relation_identity import lifecycle_audit_filter


def test_native_audit_filter_preserves_all_lifecycle_events_and_arguments():
    observed = []
    callback = lifecycle_audit_filter(lambda event, args: observed.append((event, args)))
    payload = (object(), {'new_reference': []})
    for event in ['sys.setprofile', 'sys.settrace', 'subprocess.Popen',
                  'os.fork', 'os.posix_spawn', 'os.exec', 'os.system']:
        assert callback(event, payload) is None
        assert observed[-1][0] == event and observed[-1][1] is payload
    assert len(observed) == 7
    for event in ['object.__getattr__', 'object.__setattr__', 'import', 'open',
                  'sys.setprofile.extra', 'subprocess.Popen.extra']:
        assert callback(event, payload) is None
    assert len(observed) == 7


def test_lifecycle_audit_rejection_remains_fail_closed():
    def reject(event, args):
        raise RuntimeError('lifecycle change rejected')
    callback = lifecycle_audit_filter(reject)
    assert callback('object.__getattr__', ()) is None
    with pytest.raises(RuntimeError, match='lifecycle change rejected'):
        callback('sys.settrace', (object(),))
    with pytest.raises(TypeError, match='must be callable'):
        lifecycle_audit_filter(None)
