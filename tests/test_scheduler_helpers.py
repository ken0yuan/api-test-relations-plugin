from pathlib import Path
import json
import os
import socket
import subprocess
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from single_scheduler_helpers import launch_spec, short_temp_alias, cleanup_temp_alias


def test_launch_environment_includes_per_file_sibling_imports(tmp_path):
    repo=tmp_path/'pytorch'; source=repo/'test/nn/attention/test_fa3.py'
    source.parent.mkdir(parents=True); source.touch()
    root=tmp_path/'run'
    manifest={'repo':str(repo),'plugin_root':str(tmp_path/'plugin'),
              'python':'/usr/bin/python3','device_filter':'cuda','catalog':str(tmp_path/'catalog.json'),
              'dependency_paths':[],'cuda_capability':'12.0'}
    row={'source':str(source),'module':'nn/attention/test_fa3','runner':'pytest'}
    _,env,output=launch_spec(manifest,root,row,'attempt')
    try:
        assert str(source.parent) in env['PYTHONPATH'].split(':')
        assert Path(env['TMPDIR']).is_symlink()
        assert Path(env['TMPDIR']).resolve()==(root/'worker-runtime/attempt/tmp').resolve()
    finally:cleanup_temp_alias({'output':str(output),'temp_alias':env['TMPDIR']},root/'worker-runtime/attempt')


def test_long_run_temp_has_short_working_unix_listener_and_private_target(tmp_path):
    scratch=tmp_path/('long-run-'+'x'*100)/'worker-runtime'/('attempt-'+'y'*60)
    output=tmp_path/'capture'
    alias=short_temp_alias(output,scratch/'tmp')
    worker={'output':str(output),'temp_alias':str(alias)}
    listener=alias/'pymp-12345678'/'listener-12345678'
    listener.parent.mkdir()
    assert len(os.fsencode(str(listener))) < 108
    try:
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:sock.bind(str(listener))
        assert (scratch/'tmp/pymp-12345678/listener-12345678').exists()
    finally:
        listener.unlink(missing_ok=True)
        cleanup_temp_alias(worker,scratch)
    assert not alias.is_symlink()
    assert (scratch/'tmp').is_dir()


def test_temp_cleanup_refuses_changed_alias_and_preserves_unrelated_content(tmp_path):
    output=tmp_path/'capture';scratch=tmp_path/'scratch';alias=short_temp_alias(output,scratch/'tmp')
    unrelated=tmp_path/'unrelated';unrelated.mkdir();marker=unrelated/'keep';marker.write_text('keep')
    alias.unlink();alias.symlink_to(unrelated,target_is_directory=True)
    try:
        with pytest.raises(ValueError,match='target changed'):
            cleanup_temp_alias({'output':str(output),'temp_alias':str(alias)},scratch)
        assert marker.read_text()=='keep'
    finally:alias.unlink()


def test_resource_sharer_fd_roundtrip_with_long_physical_scratch(tmp_path):
    scratch=tmp_path/('long-run-'+'x'*100)/'worker-runtime'/('attempt-'+'y'*60)
    output=tmp_path/'capture';alias=short_temp_alias(output,scratch/'tmp')
    code='''import json,os,tempfile
from multiprocessing import reduction,resource_sharer
with tempfile.TemporaryFile() as original:
    transferred=reduction.DupFd(original.fileno()).detach()
    os.close(transferred)
print(json.dumps({'address':resource_sharer._resource_sharer._address}))
'''
    try:
        result=subprocess.run([sys.executable,'-c',code],env={**os.environ,'TMPDIR':str(alias)},
                              capture_output=True,text=True,timeout=10,check=True)
        address=json.loads(result.stdout)['address']
        assert address.startswith(str(alias)+'/')
        assert len(os.fsencode(address))<108
    finally:cleanup_temp_alias({'output':str(output),'temp_alias':str(alias)},scratch)
