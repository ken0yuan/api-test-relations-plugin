"""Optional descriptor permission must be explicit, local and fail closed."""
import ctypes
import importlib.util
import os
from pathlib import Path
import sys
import types
import pytest

path=Path(__file__).resolve().parents[1]/'src/api_test_relations/runtime/relation_cuda_ipc.py'
spec=importlib.util.spec_from_file_location('cuda_ipc_policy',path)
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv('API_RELATION_CUDA_IPC_VMM','1')
    monkeypatch.delenv('PYTORCH_ALLOC_CONF',raising=False)
    monkeypatch.delenv('PYTORCH_CUDA_ALLOC_CONF',raising=False)
    monkeypatch.setattr(policy.sys,'platform','linux')
    monkeypatch.delitem(sys.modules,'torch',raising=False)


def test_disabled_does_not_touch_allocator_or_permissions(monkeypatch):
    monkeypatch.delenv('API_RELATION_CUDA_IPC_VMM',raising=False)
    monkeypatch.setenv('PYTORCH_ALLOC_CONF','expandable_segments:False')
    monkeypatch.setattr(ctypes,'CDLL',lambda *a,**k:pytest.fail('disabled policy requested permission'))
    assert policy.configure() is None
    assert os.environ['PYTORCH_ALLOC_CONF']=='expandable_segments:False'


def test_only_actual_parent_family_is_authorized(enabled,monkeypatch):
    calls=[]
    monkeypatch.setattr(policy.os,'getppid',lambda:4242)
    monkeypatch.setattr(ctypes,'CDLL',lambda *a,**k:types.SimpleNamespace(prctl=lambda *args:calls.append(args) or 0))
    result=policy.configure()
    assert calls==[(policy.PR_SET_PTRACER,4242,0,0,0)]
    assert result['authorized_parent_pid']==4242
    assert result['global_kernel_policy_changed'] is False
    assert result['legacy_cuda_ipc_repaired'] is False
    assert os.environ['PYTORCH_ALLOC_CONF']=='expandable_segments:True'


def test_permission_failure_preserves_allocator(enabled,monkeypatch):
    monkeypatch.setenv('PYTORCH_ALLOC_CONF','backend:native')
    monkeypatch.setattr(ctypes,'CDLL',lambda *a,**k:types.SimpleNamespace(prctl=lambda *args:-1))
    with pytest.raises(OSError):policy.configure()
    assert os.environ['PYTORCH_ALLOC_CONF']=='backend:native'


@pytest.mark.parametrize('environment',[
    {'PYTORCH_ALLOC_CONF':'expandable_segments:False'},
    {'PYTORCH_ALLOC_CONF':'backend:cudaMallocAsync'},
    {'PYTORCH_ALLOC_CONF':'backend:native','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}])
def test_conflicting_allocator_is_not_silently_changed(environment):
    with pytest.raises(RuntimeError):policy.allocator_config(environment)


def test_initialized_cuda_is_rejected_before_permission(enabled,monkeypatch):
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda:True)))
    monkeypatch.setattr(ctypes,'CDLL',lambda *a,**k:pytest.fail('initialized CUDA requested permission'))
    with pytest.raises(RuntimeError,match='before CUDA initialization'):policy.configure()
