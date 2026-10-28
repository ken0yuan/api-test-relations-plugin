import hashlib
import marshal
from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from aot_const_format_compat import build_formatter, replace_nested

SOURCE = '''def outer():
    unchanged = lambda x: x + 1
    def format_consts_to_cpp(consts, align_bytes, symbol_prefix):
        consts_size = len(consts)
        asan_attr = "prefix\\n"
        const_cpp = asan_attr
        const_cpp += f"alignas({align_bytes}) extern "
        const_cpp += f"unsigned char {symbol_prefix}_binary_constants_bin_start[{consts_size}] = {{\\t\\n"
        count_bytes = 0
        for c in consts:
            const_cpp += f"{c}, "
            count_bytes = count_bytes + 1
            if count_bytes % 16 == 0:
                const_cpp += "\\t\\n"
        const_cpp += "};\\t\\n"
        const_cpp += f"alignas({align_bytes}) extern unsigned char * {symbol_prefix}_binary_constants_bin_end;\\t\\n"
        return const_cpp, "weights.cpp"
    return format_consts_to_cpp, unchanged
'''


def pair():
    scope = {}
    exec(compile(SOURCE, 'sample.py', 'exec'), scope)
    original = scope['outer']
    renderer, changes = build_formatter(SOURCE, 'sample.py', hashlib.sha256(SOURCE.encode()).hexdigest())
    replacement, found = replace_nested(original.__code__, renderer)
    assert found == 1
    fixed = types.FunctionType(replacement, scope)
    return original, fixed


@pytest.mark.parametrize('payload', [b'', b'\0', bytes(range(256)), bytes(range(256)) * 17,
                                    bytearray(range(256))])
def test_exact_cpp_and_unchanged_code_and_identity(payload):
    old, new = pair()
    before, unchanged_before = old()
    after, unchanged_after = new()
    assert before(payload, 64, '_') == after(payload, 64, '_')
    assert before.__code__.co_firstlineno == after.__code__.co_firstlineno
    assert before.__qualname__ == after.__qualname__
    assert marshal.dumps(unchanged_before.__code__) == marshal.dumps(unchanged_after.__code__)
    identity = id(old)
    old.__code__ = new.__code__
    assert id(old) == identity
    assert old()[0](payload, 64, '_') == after(payload, 64, '_')


def test_fail_closed_on_source_or_shape_change():
    with pytest.raises(ValueError, match='hash mismatch'):
        build_formatter(SOURCE, 'sample.py')
    changed = SOURCE.replace('const_cpp += "};', 'const_cpp -= "};')
    with pytest.raises(ValueError, match='Unexpected constant renderer'):
        build_formatter(changed, 'sample.py', hashlib.sha256(changed.encode()).hexdigest())


def test_missing_renderer_is_not_silently_patched():
    renderer, _ = build_formatter(SOURCE, 'sample.py', hashlib.sha256(SOURCE.encode()).hexdigest())
    code = compile('x = 1', 'other.py', 'exec')
    result, count = replace_nested(code, renderer)
    assert result is code and count == 0


@pytest.mark.parametrize('modules,enabled', [(None, False), ([], False),
                                           (['other_module'], False),
                                           (['inductor/test_aot_inductor'], True)])
def test_scheduler_requires_explicit_module_opt_in(tmp_path, monkeypatch, modules, enabled):
    from single_scheduler_helpers import launch_spec, cleanup_temp_alias
    monkeypatch.delenv('RELATION_AOT_CONST_FORMAT', raising=False)
    repo = tmp_path / 'pytorch'
    source = repo / 'test/inductor/test_aot_inductor.py'
    source.parent.mkdir(parents=True)
    source.touch()
    root = tmp_path / 'run'
    manifest = {'repo': str(repo), 'plugin_root': str(tmp_path / 'plugin'),
                'python': '/usr/bin/python3', 'device_filter': 'cuda',
                'catalog': str(tmp_path / 'catalog.json'), 'cuda_capability': '12.0'}
    if modules is not None:
        manifest['aot_const_format_modules'] = modules
    row = {'source': str(source), 'module': 'inductor/test_aot_inductor', 'runner': 'pytest'}
    cmd, env, output = launch_spec(manifest, root, row, 'attempt')
    try:
        assert ('aot_const_format_compat' in cmd) is enabled
        assert (env.get('RELATION_AOT_CONST_FORMAT') == '1') is enabled
    finally:
        cleanup_temp_alias({'output': str(output), 'temp_alias': env['TMPDIR']},
                           root / 'worker-runtime/attempt')
