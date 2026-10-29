"""CUDA dispatch excludes XPU files before pytest collection and keeps history."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from single_scheduler_helpers import backend_exclusion_reason,launch_spec
spec=importlib.util.spec_from_file_location('backend_scope_runner',SCRIPTS/'run_single_database.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


@pytest.mark.parametrize('module',[
    'xpu/test_conv','inductor/test_xpu_basic','profiler/test_xpu_scope_profiler',
    'test_xpu','test_xpu_expandable_segments',
])
def test_cuda_scope_rejects_xpu_before_creating_scratch_or_launch(tmp_path,module):
    row={'module':module,'file':'pytorch/test/'+module+'.py'}
    assert backend_exclusion_reason({'device_filter':'cuda'},row)=='excluded_backend_xpu'
    with pytest.raises(ValueError,match='excluded_backend_xpu'):
        launch_spec({'device_filter':'cuda'},tmp_path/'run',row,'attempt')
    assert not (tmp_path/'run').exists()


def test_scope_is_device_specific_and_does_not_match_incidental_name():
    assert backend_exclusion_reason({'device_filter':'xpu'},{'module':'xpu/test_conv'}) is None
    assert backend_exclusion_reason({'device_filter':'cuda'},{'module':'test_xpudata'}) is None
    assert backend_exclusion_reason({'device_filter':'cuda'},{'module':'test_cuda'}) is None
    assert backend_exclusion_reason({'device_filter':'cuda'},{'module':'inductor/test_max_autotune'}) is None
    assert backend_exclusion_reason({'device_filter':'cuda'},{'module':'alias','file':r'pytorch\test\xpu\test_conv.py'})=='excluded_backend_xpu'


def test_exclusion_preserves_prior_state_without_touching_capture_database(tmp_path,monkeypatch):
    queue={'xpu/test_conv':'captured','test_xpu':'capture_incomplete','test_cuda':'pending'}
    meta={'device_filter':'cuda','tests':[{'module':k,'file':'pytorch/test/'+k+'.py'} for k in queue]}
    monkeypatch.setattr(runner.sqlite3,'connect',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('must not modify historical database')))
    records=runner.apply_backend_exclusions(meta,queue,tmp_path)
    assert queue=={'xpu/test_conv':'excluded_backend_xpu','test_xpu':'excluded_backend_xpu','test_cuda':'pending'}
    assert [r['prior_queue_state'] for r in records]==['captured','capture_incomplete']
    report=json.loads((tmp_path/'backend-exclusions.json').read_text())
    assert report['historical_evidence_preserved'] and all(not r['future_collection_allowed'] for r in report['modules'])
