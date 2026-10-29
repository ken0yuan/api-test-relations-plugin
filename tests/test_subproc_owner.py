import pytest

from relation_context import OWNER
from relation_subproc_pool import MAGIC, MAX_CONTEXT, owned, pack, unpack


def context(name='one'):
    return {'token': 'a' * 32, 'nodeid': 'pytorch/test_probe.py::test_' + name,
            'phase': 'call', 'transport': 'subproc_pool_task', 'settings': {}}


def test_transport_preserves_arbitrary_original_payload_bytes():
    payload = bytes(range(256)) * 7
    owner = context()
    decoded, original = unpack(pack(owner, payload))
    assert decoded == owner and original == payload


def test_unowned_legacy_job_bytes_pass_through_unchanged():
    payload = b'\x80\x05legacy-pickled-job\x00\xff'
    assert unpack(payload) == (None, payload)


@pytest.mark.parametrize('data', [MAGIC, MAGIC + b'\x00\x00',
                                MAGIC + (MAX_CONTEXT + 1).to_bytes(4, 'big'),
                                MAGIC + b'\x00\x00\x00\x10{}'])
def test_truncated_or_oversized_owner_envelope_is_rejected(data):
    with pytest.raises(ValueError):
        unpack(data)


def test_invalid_owner_cannot_silently_fall_back_to_pool_creation_test():
    owner = context()
    owner['nodeid'] = None
    with pytest.raises(ValueError):
        unpack(pack(owner, b'payload'))


def test_nested_ownership_restored_after_callable_exception():
    token = OWNER.set(('outside', 'setup'))
    try:
        with owned(context('A')):
            assert OWNER.get() == (context('A')['nodeid'], 'call')
            with pytest.raises(RuntimeError):
                with owned(context('B')):
                    assert OWNER.get() == (context('B')['nodeid'], 'call')
                    raise RuntimeError('workload error')
            assert OWNER.get() == (context('A')['nodeid'], 'call')
        assert OWNER.get() == ('outside', 'setup')
    finally:
        OWNER.reset(token)


def test_unsupported_or_partially_imported_pool_is_never_patched(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from relation_subproc_pool import MODULE, SubprocPoolBridge
    source = tmp_path / 'foreign_pool.py'
    source.write_text('# incompatible upstream implementation\n')
    gaps = []
    class Pool:
        def submit(self): pass
    module = SimpleNamespace(__file__=str(source), __spec__=SimpleNamespace(_initializing=True),
                             SubprocPool=Pool, SubprocMain=Pool)
    monkeypatch.setitem(sys.modules, MODULE, module)
    original = Pool.submit
    bridge = SubprocPoolBridge(SimpleNamespace(plugin=SimpleNamespace(gap=lambda *x:gaps.append(x))))
    bridge.maybe_install()
    assert not gaps and not bridge.rejected and Pool.submit is original
    module.__spec__._initializing = False
    bridge.maybe_install();bridge.maybe_install()
    assert len(gaps) == 1 and 'subproc_pool_adapter_unsupported:' in gaps[0][1]
    assert bridge.rejected and Pool.submit is original


def test_completed_parent_is_not_interrupted_by_compile_child_lifetime(tmp_path):
    import ast
    import sqlite3
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / 'scripts/run_single_database.py'
    tree = ast.parse(source.read_text())
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'entered_nodes')
    namespace = {'Path':Path, 'sqlite3':sqlite3}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), namespace)
    for name,role,rows in [('parent','root',[('complete','call','closed'),('active','call','entered')]),
                           ('child','child',[('complete','call','entered')])]:
        with sqlite3.connect(tmp_path/(name+'.sqlite')) as db:
            db.execute('CREATE TABLE meta(key TEXT,value TEXT)')
            db.execute('INSERT INTO meta VALUES (?,?)',('role',role))
            db.execute('CREATE TABLE phases(nodeid TEXT,phase TEXT,state TEXT)')
            db.executemany('INSERT INTO phases VALUES (?,?,?)',rows)
    assert namespace['entered_nodes']({'shards':str(tmp_path)}) == ['active']
