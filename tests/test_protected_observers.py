"""Independent observers survive native and Python profiler transitions."""
import cProfile
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace

import pytest
from _relation_identity import protected_backend_available
from relation_collector import Collector,Registry

pytestmark=[pytest.mark.paired_plugins,pytest.mark.skipif(not protected_backend_available(),reason='requires CPython callback boundary protection')]


def observed(tmp_path,source,invoke):
    from trace_api_chain.object_capture import Recorder
    from trace_api_chain.replay import rebuild
    import trace_api_chain.replay as replay_module
    name='protected_fixture';path=tmp_path/'api.py';path.write_text(source)
    module=ModuleType(name);module.__file__=str(path);exec(compile(source,str(path),'exec'),vars(module));sys.modules[name]=module
    cat={'apis':[{'api_id':'python:torch.api','qualified_name':'torch.api','language':'python','kind':'function'}]}
    hits=[];gaps=[];collector=Collector(Registry(cat,SimpleNamespace(api=module.api)),lambda owner,ids,site:hits.append((owner,ids)),lambda owner,reason:gaps.append(reason))
    recorder=Recorder(tmp_path/'events.jsonl','test::one',tmp_path/'test.py',namespaces=(name,))
    collector.install();owner=collector.enter(('test::one','call'))
    recorder.begin();assert recorder.protected_subscription is not None
    try:invoke(module)
    finally:
        recorder.end();collector.leave(owner);collector.close();sys.modules.pop(name,None)
    rows=[json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert not gaps and not recorder.gaps
    assert not recorder.stack
    assert sum(r['event']=='call' for r in rows)==sum(r['event']=='return' for r in rows)
    graph=rebuild(rows,[],json.loads(Path(replay_module.__file__).with_name('contract.json').read_text()))
    assert not graph['gaps'] and hits
    return module,rows


def test_cprofile_enable_disable_does_not_replace_independent_observers(tmp_path):
    source='''import cProfile
def leaf():return len([])
def api():
 p=cProfile.Profile()
 p.enable()
 leaf()
 p.disable()
 leaf()
 return p
'''
    saved=[]
    module,rows=observed(tmp_path,source,lambda module:saved.append(module.api()))
    assert len([r for r in rows if r['event']=='call' and r['name']=='leaf'])==2
    assert any(getattr(r.code,'co_name',None)=='leaf' for r in saved[0].getstats())


def test_python_profile_and_trace_clients_can_change_without_losing_capture(tmp_path):
    source='''import sys
events=[]
def client(frame,event,arg):
 if frame.f_code is leaf.__code__ and event=='call':events.append('profile')
def trace(frame,event,arg):
 if frame.f_code is leaf.__code__ and event=='line':events.append('trace')
 return trace
def leaf():return 1
def api():
 sys.setprofile(client)
 assert sys.getprofile() is client
 sys.settrace(trace)
 assert sys.gettrace() is trace
 leaf()
 sys.setprofile(None)
 sys.settrace(None)
 leaf()
'''
    module,rows=observed(tmp_path,source,lambda module:module.api())
    assert module.events.count('profile')==1 and module.events.count('trace')==1
    assert len([r for r in rows if r['event']=='call' and r['name']=='leaf'])==2


def test_rejected_setter_preserves_the_previous_client(tmp_path):
    source='''import sys
reject=False
def client(frame,event,arg):pass
def audit(event,args):
 if reject and event=='sys.setprofile':raise RuntimeError('deny callback change')
sys.addaudithook(audit)
def api():
 global reject
 sys.setprofile(client)
 reject=True
 try:
  try:sys.setprofile(None)
  except RuntimeError:pass
  else:raise AssertionError('audit rejection was ignored')
  assert sys.getprofile() is client
 finally:reject=False
 sys.setprofile(None)
'''
    observed(tmp_path,source,lambda module:module.api())


def test_cprofile_context_and_native_error_preserve_real_returns(tmp_path):
    source="""import cProfile
def leaf():return 1
def api():
 p=cProfile.Profile()
 try:p.enable(unexpected_argument=True)
 except TypeError:pass
 else:raise AssertionError('native exception was lost')
 with p:leaf()
 leaf()
 return p
"""
    saved=[]
    _,rows=observed(tmp_path,source,lambda module:saved.append(module.api()))
    assert len([r for r in rows if r['event']=='call' and r['name']=='leaf'])==2
    assert any(getattr(r.code,'co_name',None)=='leaf' for r in saved[0].getstats())


def test_saved_native_setters_preserve_callable_identity_and_real_returns(tmp_path):
    source="""import sys
raw_profile=sys.setprofile
raw_trace=sys.settrace
def leaf():return 1
def api():
 raw_profile(None)
 raw_trace(None)
 leaf()
"""
    # Save the actual native setters before Collector replaces public interfaces.
    _,rows=observed(tmp_path,source,lambda module:module.api())
    assert len([r for r in rows if r['event']=='call' and r['name']=='leaf'])==1


def test_native_profiler_methods_saved_before_first_install_are_protected(tmp_path):
    import os,subprocess
    source="""import cProfile
p=cProfile.Profile()
saved_enable=p.enable
saved_disable=p.disable
def leaf():return 1
def api():
 saved_enable()
 leaf()
 saved_disable()
 leaf()
 return p
"""
    script="from pathlib import Path;from test_protected_observers import observed;observed(Path("+repr(str(tmp_path))+"),"+repr(source)+",lambda module:module.api())"
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(sys.path))
    result=subprocess.run([sys.executable,'-c',script],env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
