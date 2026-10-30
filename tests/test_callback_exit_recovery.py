"""A failing debugger/profiler disables itself, preserving independent capture."""
import sys
from types import SimpleNamespace
import pytest
from _relation_identity import guard_intact
from test_relation_collector import collecting,catalog


@pytest.mark.parametrize('event',['call','line','opcode','return'])
@pytest.mark.parametrize('error_type',[ValueError,KeyboardInterrupt,SystemExit])
def test_trace_error_propagates_unchanged_and_capture_continues(event,error_type):
    def api():return 1
    def target():
        value=2
        return value
    error=error_type('debugger exit')
    def trace(frame,kind,arg):
        if frame.f_code is target.__code__:
            frame.f_trace_opcodes=True
            if kind==event:raise error
        return trace
    with collecting(SimpleNamespace(api=api),catalog(('api','function'))) as (collector,hits,gaps):
        token=collector.enter(('test','call'))
        sys.settrace(trace)
        with pytest.raises(error_type) as caught:target()
        assert caught.value is error
        assert sys.gettrace() is None
        assert guard_intact()
        api()
        # A new debugger can start normally after the previous one failed.
        seen=[]
        def next_trace(frame,kind,arg):
            if frame.f_code is target.__code__:seen.append(kind)
            return next_trace
        sys.settrace(next_trace);target();sys.settrace(None)
        assert 'call' in seen
        collector.leave(token)
    assert hits==[(('test','call'),('python:torch.api',))]
    assert gaps==[]


@pytest.mark.parametrize('error_type',[ValueError,KeyboardInterrupt,SystemExit])
def test_profile_error_disables_only_the_client(error_type):
    def api():return 1
    def target():return 2
    error=error_type('profiler exit')
    def profile(frame,event,arg):
        if frame.f_code is target.__code__ and event=='call':raise error
    with collecting(SimpleNamespace(api=api),catalog(('api','function'))) as (collector,hits,gaps):
        token=collector.enter(('test','call'))
        sys.setprofile(profile)
        with pytest.raises(error_type) as caught:target()
        assert caught.value is error
        assert sys.getprofile() is None
        assert guard_intact()
        api();collector.leave(token)
    assert hits==[(('test','call'),('python:torch.api',))]
    assert gaps==[]


def test_observer_code_isolation_keeps_workload_identity(monkeypatch):
    from relation_collector import Collector,Registry
    seen=[]
    action=SimpleNamespace(SKIP=object(),DEFAULT=object())
    frame_api=SimpleNamespace(_FrameAction=action,_FrameExecStrategy=lambda *args:args,
                              set_code_exec_strategy=lambda code,strategy:seen.append(code))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(_C=SimpleNamespace(_dynamo=SimpleNamespace(eval_frame=frame_api))))
    def api():return 1
    original=api.__code__
    with collecting(SimpleNamespace(api=api),catalog(('api','function'))) as (collector,hits,gaps):
        token=collector.enter(('test','call'));api();collector.leave(token)
    assert Registry._match.__code__ in seen and Collector._dispatch.__code__ in seen
    assert original not in seen and api.__code__ is original
    assert hits and not gaps
