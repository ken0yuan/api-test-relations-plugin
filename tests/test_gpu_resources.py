"""A GPU query failure must be diagnosable and must not grant a launch slot."""
from pathlib import Path
import subprocess
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import single_scheduler_helpers as helper


@pytest.mark.parametrize('text,expected', [('0\n',0), ('3072\n',3072), (' 5120\n',5120)])
def test_valid_gpu_query_keeps_zero_distinct_from_unavailable(monkeypatch,text,expected):
    calls=[]
    def query(command,**kwargs):
        calls.append((command,kwargs));return text
    monkeypatch.setattr(helper.subprocess,'check_output',query)
    value,detail=helper.gpu_memory_sample()
    assert value==expected and detail['state']=='ok' and detail['free_mib']==expected
    assert len(calls)==1 and calls[0][1]['timeout']==5
    assert calls[0][1]['stderr'] is subprocess.PIPE
    assert helper.can_launch({'memory_available_mib':4096,'gpu_free_mib':value})==(expected>=3072)


@pytest.mark.parametrize('text', ['', 'N/A\n', '1\n2\n', '-1', '1.5'])
def test_invalid_query_output_never_becomes_free_memory(monkeypatch,text):
    monkeypatch.setattr(helper.subprocess,'check_output',lambda *a,**k:text)
    value,detail=helper.gpu_memory_sample()
    assert value is None and detail['state']=='invalid_output'
    assert detail['stdout']==text
    assert not helper.can_launch({'memory_available_mib':16000,'gpu_free_mib':value})


@pytest.mark.parametrize('error,state', [
    (subprocess.TimeoutExpired(['nvidia-smi'],5,output=b'partial',stderr=b'delayed'),'timeout'),
    (subprocess.CalledProcessError(9,['nvidia-smi'],output='partial',stderr='NVML unavailable'),'command_failed'),
    (FileNotFoundError(2,'missing nvidia-smi'),'os_error'),
])
def test_failure_kind_and_original_diagnostic_are_preserved(monkeypatch,error,state):
    def fail(*a,**k):raise error
    monkeypatch.setattr(helper.subprocess,'check_output',fail)
    value,detail=helper.gpu_memory_sample()
    assert value is None and detail['state']==state
    assert detail['exception']==type(error).__name__
    assert detail['elapsed_seconds']>=0
    if state=='timeout':assert detail['stdout']=='partial' and detail['stderr']=='delayed'
    if state=='command_failed':assert detail['returncode']==9 and detail['stderr']=='NVML unavailable'
    if state=='os_error':assert detail['errno']==2
    assert not helper.can_launch({'memory_available_mib':16000,'gpu_free_mib':value})


@pytest.mark.parametrize('value,state', [(None,'timeout'), (4096,'ok')])
def test_resource_publication_includes_query_state_without_stale_fallback(monkeypatch,value,state):
    detail={'state':state,'elapsed_seconds':.1,'timeout_seconds':5}
    monkeypatch.setattr(helper,'gpu_memory_sample',lambda:(value,detail))
    result=helper.resources()
    assert result['gpu_free_mib'] is value and result['gpu_query']==detail
    result['memory_available_mib']=4096
    assert helper.can_launch(result)==(value is not None)
