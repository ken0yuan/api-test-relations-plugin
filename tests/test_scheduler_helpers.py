from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from single_scheduler_helpers import launch_spec


def test_launch_environment_includes_per_file_sibling_imports(tmp_path):
    repo=tmp_path/'pytorch'; source=repo/'test/nn/attention/test_fa3.py'
    source.parent.mkdir(parents=True); source.touch()
    root=tmp_path/'run'
    manifest={'repo':str(repo),'plugin_root':str(tmp_path/'plugin'),
              'python':'/usr/bin/python3','device_filter':'cuda','catalog':str(tmp_path/'catalog.json'),
              'dependency_paths':[],'cuda_capability':'12.0'}
    row={'source':str(source),'module':'nn/attention/test_fa3','runner':'pytest'}
    _,env,_=launch_spec(manifest,root,row,'attempt')
    assert str(source.parent) in env['PYTHONPATH'].split(':')
