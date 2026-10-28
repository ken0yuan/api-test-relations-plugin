"""Explicit, hash-guarded linear AOT constant formatting for observed CPython 3.11.

Only a nested string renderer is replaced. The compiler's callable identity,
all other code, API observation, source files and generated C++ bytes survive.
"""
import ast
import hashlib
import json
import marshal
import os
from pathlib import Path
import sys
import types

SOURCE_SHA256 = 'c7873c46a80a2e72c3ab901aa1e18632f95e9dde7fa9baac98296a19b1a5be45'
REVISION = 'aot-const-format-linear-1'
_installed = None


def code_digest(code):
    return hashlib.sha256(marshal.dumps(code)).hexdigest()


def build_formatter(source, filename, expected=SOURCE_SHA256):
    if hashlib.sha256(source.encode()).hexdigest() != expected:
        raise ValueError('AOT constant formatter source hash mismatch')
    candidates = [n for n in ast.walk(ast.parse(source))
                  if isinstance(n, ast.FunctionDef) and n.name == 'format_consts_to_cpp']
    if len(candidates) != 1:
        raise ValueError('Expected exactly one constant formatter')
    node = candidates[0]
    changed = {'start': 0, 'append': 0, 'return': 0}

    class Linear(ast.NodeTransformer):
        def visit_Assign(self, item):
            if (len(item.targets) == 1 and isinstance(item.targets[0], ast.Name)
                    and item.targets[0].id == 'const_cpp'):
                item.value = ast.copy_location(ast.List(elts=[item.value], ctx=ast.Load()), item.value)
                changed['start'] += 1
            return item

        def visit_AugAssign(self, item):
            if isinstance(item.target, ast.Name) and item.target.id == 'const_cpp':
                if not isinstance(item.op, ast.Add):
                    raise ValueError('Unexpected constant renderer operation')
                # List in-place extension is linear and does not introduce a
                # new profiled C-method call for every constant byte.
                item.value = ast.copy_location(ast.Tuple(elts=[item.value], ctx=ast.Load()), item.value)
                changed['append'] += 1
                return item
            return item

        def visit_Return(self, item):
            if (isinstance(item.value, ast.Tuple) and len(item.value.elts) == 2
                    and isinstance(item.value.elts[0], ast.Name) and item.value.elts[0].id == 'const_cpp'):
                item.value.elts[0] = ast.copy_location(ast.Call(
                    func=ast.Attribute(value=ast.Constant(value=''), attr='join', ctx=ast.Load()),
                    args=[ast.Name(id='const_cpp', ctx=ast.Load())], keywords=[]), item.value.elts[0])
                changed['return'] += 1
            return item

    transformed = Linear().visit(node)
    if changed != {'start': 1, 'append': 6, 'return': 1}:
        raise ValueError('Unexpected constant formatter structure: ' + str(changed))
    module = ast.fix_missing_locations(ast.Module(body=[transformed], type_ignores=[]))
    code = compile(module, filename, 'exec')
    return next(c for c in code.co_consts if isinstance(c, types.CodeType)), changed


def replace_nested(code, replacement):
    found = 0
    parts = []
    for item in code.co_consts:
        if isinstance(item, types.CodeType):
            if item.co_name == 'format_consts_to_cpp':
                if item.co_freevars or replacement.co_freevars or item.co_argcount != replacement.co_argcount:
                    raise ValueError('Unexpected formatter signature or closure')
                item = replacement.replace(co_qualname=item.co_qualname)
                found += 1
            else:
                item, count = replace_nested(item, replacement)
                found += count
        parts.append(item)
    return (code.replace(co_consts=tuple(parts)) if found else code), found


def install():
    global _installed
    if _installed:
        if _installed[0].__code__ is not _installed[2]:
            raise RuntimeError('AOT compiler changed after compatibility installation')
        return _installed[3]
    cache = sys.modules.get('torch._inductor.codecache')
    if cache is None:
        raise RuntimeError('AOT codecache must already be imported by normal collection')
    path = Path(cache.__file__).resolve()
    renderer, changes = build_formatter(path.read_text(), str(path))
    function = cache.AotCodeCompiler.compile
    function = getattr(function, '__func__', function)
    original = function.__code__
    rewritten, count = replace_nested(original, renderer)
    if count != 1:
        raise ValueError('Expected exactly one nested constant formatter in compiler')
    proof = {'revision': REVISION, 'enabled': True, 'source': str(path),
             'source_sha256': SOURCE_SHA256, 'original_code_sha256': code_digest(original),
             'replacement_code_sha256': code_digest(rewritten), 'changes': changes,
             'source_file_modified': False, 'compiler_callable_identity_preserved': True,
             'observation_disabled': False}
    function.__code__ = rewritten
    _installed = (function, original, rewritten, proof)
    return proof


def pytest_collection_finish(session):
    # Preserve normal imports and alias resolution. Patch only after the test
    # module has imported its real compiler, before any test body executes.
    if os.environ.get('RELATION_AOT_CONST_FORMAT') == '1':
        proof = install()
        output = Path(session.config.getoption('--api-relations-out'))
        (output / 'aot-const-format-compat.json').write_text(json.dumps(proof, indent=2))
        environment = output / 'environment.json'
        data = json.loads(environment.read_text())
        data['aot_const_format_compatibility'] = proof
        environment.write_text(json.dumps(data, indent=2))
