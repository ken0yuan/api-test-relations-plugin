"""Explicit opt-in; never runs a user's previously completed PyTorch tests."""
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import pytest
from api_test_relations.cli import PACKAGE


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get('RUN_API_RELATIONS_CUDA_IPC_TESTS')!='1',reason='explicit CUDA IPC integration opt-in')
def test_real_cuda_parent_child_and_sibling_capture(tmp_path):
    repo=tmp_path/'repo';(repo/'test').mkdir(parents=True)
    shutil.copyfile(Path(__file__).parent/'fixtures/cuda_ipc_transport_case.py',repo/'test/test_transport.py')
    out=tmp_path/'capture'
    env={k:v for k,v in os.environ.items() if not k.startswith('API_RELATION_')}
    env.update(PYTHONPATH=str(PACKAGE.parent),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with tempfile.TemporaryDirectory(prefix='atr-ipc-') as short:
        env['TMPDIR']=short
        result=subprocess.run([sys.executable,'-m','pytest',str(repo/'test'),'-q','--tb=short',
            '-p','api_test_relations.pytest_plugin','--api-relations-out',str(out),
            '--api-relations-repo',str(repo),'--api-relations-cuda-ipc-vmm'],
            cwd=repo,env=env,capture_output=True,text=True,timeout=70)
    assert result.returncode==0,result.stdout+result.stderr
    with sqlite3.connect(out/'relations.sqlite') as db:
        assert not db.execute('SELECT * FROM gaps').fetchall()
        assert len(db.execute("SELECT DISTINCT nodeid FROM phases WHERE phase='call' AND state='closed'").fetchall())==3
        assert db.execute('SELECT count(*) FROM relations').fetchone()[0]>0
        assert all(row[0] in ('acknowledged_complete','helper_service_observed') for row in db.execute('SELECT state FROM processes'))
    policies=[json.loads(p.read_text()) for p in (out/'shards/cuda-ipc-compat').glob('*.json')]
    assert len(policies)>=4
    assert all(p['authorized_parent_pid']>1 and not p['global_kernel_policy_changed'] for p in policies)
