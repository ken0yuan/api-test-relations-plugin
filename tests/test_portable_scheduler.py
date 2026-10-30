import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import pytest
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from scheduler_backends import DeviceBackend, DeviceLeases, parse_npu
from scheduler_state import Ledger, create_database, verdict
from scheduler_policy import ExecutionPolicy
from single_scheduler_helpers import launch_spec, cleanup_temp_alias


def test_npu_mapping_uses_logical_devices_and_all_chips():
    mapping = '0 0 3 Ascend\n0 1 - Mcu\n2 0 7 Ascend\n2 1 9 Ascend\n'
    usage = '''| 0 910B | OK | 99 | 40 |
| 0 | 0000:01:00.0 | 0 | 0 / 0 | 4 / 100 |
| 2 910B | OK | 99 | 40 |
| 0 | 0000:02:00.0 | 0 | 0 / 0 | 5 / 200 |
| 1 | 0000:02:00.1 | 0 | 0 / 0 | 6 / 200 |'''
    devices = parse_npu(mapping,usage)
    assert [(d['id'],d['free_mib']) for d in devices]==[('3',96),('7',195),('9',194)]
    with pytest.raises(ValueError):parse_npu(mapping,'unknown memory format')


@pytest.mark.parametrize('backend,mask',[('cuda','CUDA_VISIBLE_DEVICES'),('npu','ASCEND_RT_VISIBLE_DEVICES')])
def test_inventory_respects_visibility_and_no_stale_admission(monkeypatch,backend,mask):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.delenv('ASCEND_RT_VISIBLE_DEVICES',raising=False)
    driver = DeviceBackend({'backend':backend,'device_probe_command':['configured-probe']})
    monkeypatch.setattr(driver,'command',lambda cmd:json.dumps({'devices':[{'id':str(i),'free_mib':8000} for i in range(8)]}))
    assert len(driver.sample()['devices'])==8
    monkeypatch.setattr(driver,'command',lambda cmd:'invalid')
    assert driver.sample()['devices']==[] and driver.sample()['gpu_free_mib'] is None
    other = DeviceBackend({'backend':backend,'device_probe_command':['configured-probe'],'device_ids':['2']})
    monkeypatch.setenv(mask,'3,5')
    monkeypatch.setattr(other,'command',lambda cmd:json.dumps({'devices':[{'id':str(i),'free_mib':8000} for i in range(8)]}))
    assert other.sample()['devices']==[]


def test_eight_single_card_leases_and_exclusive_barrier(tmp_path):
    devices=[str(i) for i in range(8)]
    pool=DeviceLeases(devices,'npu',tmp_path/'locks')
    assert {pool.acquire(str(i),False,devices)['devices'][0] for i in range(8)}==set(devices)
    assert pool.acquire('dist',True,devices) is None
    other=DeviceLeases(devices,'npu',tmp_path/'locks')
    assert other.acquire('other',False,devices) is None
    for i in range(8):pool.release(str(i))
    assert pool.acquire('dist',True,devices)['devices']==devices
    assert other.acquire('normal',False,devices) is None
    pool.release('dist')
    lease=other.acquire('next',False,devices);assert lease is not None
    other.release('next')


def test_inherited_lease_is_kept_until_worker_exit(tmp_path):
    pool=DeviceLeases(['5'],'cuda',tmp_path/'locks')
    lease=pool.acquire('parent',False,['5'])
    process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.4)'],pass_fds=tuple(lease['fds']))
    try:
        pool.release('parent')
        other=DeviceLeases(['5'],'cuda',tmp_path/'locks')
        assert other.acquire('overlap',False,['5']) is None
        process.wait(timeout=5)
        assert other.acquire('after',False,['5']) is not None
        other.release('after')
    finally:process.wait(timeout=5)


def test_phase_selection_fills_all_regular_cards_before_exclusive():
    policy=ExecutionPolicy.__new__(ExecutionPolicy)
    policy.phase_order=['regular','distributed']
    policy.rows={'a':{'resource':1},'dist':{'resource':'all'}}
    shards={'a':{'module':'a'},'dist':{'module':'dist'}}
    states={'a':'pending','dist':'pending'}
    assert policy.choose(states,shards,{})=='a'
    states['a']='collecting'
    assert policy.choose(states,shards,{'a':{'execution_kind':'regular'}}) is None
    states['a']='attempted'
    assert policy.choose(states,shards,{})=='dist'


def ledger_fixture(tmp_path):
    nodes=['pytorch/test/a.py::test_'+name for name in ['pass','crash','after']]
    path=tmp_path/'database'
    create_database(path,{'apis':[]},[{'module':'a','file':'pytorch/test/a.py','nodeids':nodes}])
    worker={'module':'a','attempt':'one','allowed_nodeids':nodes,'devices':['5'],'started_at':1,'events':str(tmp_path/'events.jsonl')}
    return path,nodes,worker


def append_event(path,kind,**values):
    with Path(path).open('a',encoding='utf-8') as stream:stream.write(json.dumps({'kind':kind,'at':1,**values})+'\n')


def test_native_crash_records_current_and_leaves_unstarted_for_continuation(tmp_path):
    path,nodes,worker=ledger_fixture(tmp_path)
    with Ledger(path) as ledger:
        ledger.register(worker)
        append_event(worker['events'],'test_start',nodeid=nodes[0])
        for phase in ['setup','call','teardown']:
            append_event(worker['events'],'report',nodeid=nodes[0],phase=phase,outcome='passed',duration=.1)
        append_event(worker['events'],'test_start',nodeid=nodes[1])
        remaining=ledger.finish(worker,-6)
        assert remaining==[nodes[2]]
        assert ledger.db.execute('SELECT nodeid,status,test_started FROM execution_results ORDER BY nodeid').fetchall()==sorted([(nodes[0],'passed',1),(nodes[1],'crashed',1)])
        assert ledger.summary()['pending']==1
        with pytest.raises(ValueError):ledger.record({**worker,'attempt':'replacement'},nodes[0],'passed','')


def test_torn_event_is_not_a_completed_test_and_collection_errors_are_explicit(tmp_path):
    path,nodes,worker=ledger_fixture(tmp_path)
    append_event(worker['events'],'collection_error',detail='Missing verified dependency')
    with Path(worker['events']).open('ab') as stream:stream.write(b'{"kind":"report","detail":"\xe4')
    with Ledger(path) as ledger:
        ledger.register(worker);assert ledger.finish(worker,2)==[]
        assert ledger.summary()['outcomes']=={'collection_error':3}
        assert ledger.summary()['tests_started']==0


def test_exit_between_cases_leaves_the_unstarted_tail_pending(tmp_path):
    path,nodes,worker=ledger_fixture(tmp_path)
    append_event(worker['events'],'test_start',nodeid=nodes[0])
    for phase in ['setup','call','teardown']:
        append_event(worker['events'],'report',nodeid=nodes[0],phase=phase,outcome='passed',duration=.1)
    with Ledger(path) as ledger:
        ledger.register(worker)
        assert ledger.finish(worker,7)==nodes[1:]
        assert ledger.summary()['outcomes']=={'passed':1}


@pytest.mark.parametrize('backend,filter,mask,devices',[
    ('cuda','cuda','CUDA_VISIBLE_DEVICES',['2','5']),
    ('npu','privateuse1,cpu','ASCEND_RT_VISIBLE_DEVICES',['3','7'])])
def test_launch_binds_the_reserved_cards_and_keeps_backend_features(tmp_path,backend,filter,mask,devices):
    repo=tmp_path/'repo';source=repo/'test/a.py';source.parent.mkdir(parents=True);source.touch()
    manifest={'repo':str(repo),'python':sys.executable,'plugin_root':str(tmp_path/'plugin'),'backend':backend,
              'device_filter':filter,'catalog':str(tmp_path/'catalog'),'execution_results':True,'case_timeout_seconds':19}
    cmd,env,output=launch_spec(manifest,tmp_path/'run',{'module':'a','source':str(source),'runner':'pytest'},'attempt',devices)
    try:
        assert env[mask]==','.join(devices)
        assert '--timeout=19' in cmd and 'scheduler_results' in cmd
        assert ('torch_npu' in cmd)==(backend=='npu')
        assert env['PYTORCH_TESTING_DEVICE_ONLY_FOR']==filter
    finally:cleanup_temp_alias({'output':str(output),'temp_alias':env['TMPDIR']},tmp_path/'run/worker-runtime/attempt')
