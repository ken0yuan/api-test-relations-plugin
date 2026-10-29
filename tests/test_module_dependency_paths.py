from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from single_scheduler_helpers import launch_spec, cleanup_temp_alias


@pytest.mark.parametrize('module,expected', [('onnx/exporter/test_api', True), ('test_ops', False)])
def test_dependency_paths_are_scoped_to_the_selected_module(tmp_path, module, expected):
    repo=tmp_path/'pytorch';source=repo/'test'/Path(module+'.py');source.parent.mkdir(parents=True);source.touch()
    root=tmp_path/'run';extra=str(tmp_path/'onnx-deps')
    manifest={'repo':str(repo),'plugin_root':str(tmp_path/'plugin'),'python':'/usr/bin/python3',
              'device_filter':'cuda','catalog':str(tmp_path/'catalog.json'),'cuda_capability':'12.0',
              'dependency_paths':[str(tmp_path/'original-deps')],
              'module_dependency_paths':{'onnx/exporter/test_api':[extra]}}
    cmd,env,output=launch_spec(manifest,root,{'module':module,'source':str(source),'runner':'pytest'},'attempt')
    try:
        paths=env['PYTHONPATH'].split(':')
        assert (extra in paths) is expected
        assert str(tmp_path/'original-deps') in paths
        assert str(source.parent) in paths
        assert cmd[3]==str(source)
    finally:
        cleanup_temp_alias({'output':str(output),'temp_alias':env['TMPDIR']},root/'worker-runtime/attempt')
