"""Shared causal owner, usable before importing torch or the native adapter."""
from contextvars import ContextVar
from functools import lru_cache
import hashlib
import inspect
import os
from pathlib import Path
import sys
OWNER=ContextVar('api_relation_owner',default=None)
TEST_FUNCTION=ContextVar('api_relation_test_function',default=None)


@lru_cache(maxsize=4096)
def code_identity(code):
    return (os.path.realpath(code.co_filename), code.co_firstlineno,
            code.co_qualname, hashlib.sha256(code.co_code).hexdigest())


@lru_cache(maxsize=4096)
def source_digest(filename):
    try:return hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    except OSError:return None


def describe_test(function):
    """Identify the pytest body, excluding decorator helper frames."""
    try:
        function=inspect.unwrap(function)
        function=getattr(function,'__func__',function)
        code=getattr(function,'__code__',None)
    except (AttributeError,TypeError,ValueError):return None
    if code is None:return None
    return {'code':list(code_identity(code)), 'module':function.__module__,
            'qualname':function.__qualname__, 'pid':os.getpid(), 'object_id':id(function),
            'source_sha256':source_digest(code_identity(code)[0])}


def call_kind(caller, frame_function):
    target=TEST_FUNCTION.get()
    if caller is None or not isinstance(target,dict) or not target.get('code'):
        return 'unknown'
    # A spawned interpreter may import the same test without pytest's assertion
    # rewriting. Compare source identity, not rewritten bytecode, across processes.
    if code_identity(caller.f_code)[:3]!=tuple(target['code'][:3]):return 'indirect'
    actual=frame_function(caller)
    if target.get('pid')==os.getpid():
        return 'direct' if id(actual)==target.get('object_id') else 'indirect'
    if not target.get('source_sha256') or source_digest(target['code'][0])!=target['source_sha256']:
        return 'unknown'
    # Process-local pointers are never compared across spawn/fork boundaries.
    value=sys.modules.get(target.get('module'))
    try:
        for part in target.get('qualname','').split('.'):
            value=inspect.getattr_static(value,part)
        value=getattr(value,'__func__',value)
        value=inspect.unwrap(value)
    except (AttributeError,TypeError,ValueError):return 'unknown'
    return 'direct' if actual is value else 'unknown'
