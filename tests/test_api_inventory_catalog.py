import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from api_inventory_catalog import inventory_catalog
from test_relation_collector import collecting


def test_catalog_preserves_all_authoritative_labels_and_separates_binding_paths(tmp_path):
    names=['Tensor.add','Function.backward','view','.normalization.rms_norm','torch.compile','torch_npu.npu_rms_norm']
    file=tmp_path/'apis.jsonl';file.write_text(''.join(json.dumps({'api':n})+'\n' for n in names),encoding='utf-8')
    cat=inventory_catalog(file)
    assert [r['qualified_name'] for r in cat['apis']]==names
    assert [r['api_id'] for r in cat['apis']]==['python:'+n for n in names]
    assert cat['apis'][0]['lookup_name']=='torch.Tensor.add'
    assert cat['apis'][2]['lookup_name']=='view'
    assert 'torch_npu' in cat['namespace_roots']


def test_new_namespace_loaded_inside_test_is_observed_without_importing_it(monkeypatch):
    def operation():return 3
    monkeypatch.delitem(sys.modules,'inventory_extra',raising=False)
    cat={'namespace_roots':['inventory_extra'],'apis':[{'api_id':'python:inventory_extra.op','qualified_name':'inventory_extra.op','kind':'function','language':'python'}]}
    with collecting(SimpleNamespace(),cat) as (collector,hits,gaps):
        assert collector.registry.status['python:inventory_extra.op']=='namespace_not_loaded'
        token=collector.enter(('test','call'))
        module=ModuleType('inventory_extra');module.op=operation
        monkeypatch.setitem(sys.modules,'inventory_extra',module)
        module.op()
        collector.leave(token)
    assert hits==[(('test','call'),('python:inventory_extra.op',))]
    assert gaps==[]


def test_callable_packets_with_shared_protocol_keep_exact_receiver_identity():
    class Packet:
        def __call__(self):return 3
    first=Packet();alias=first;second=Packet();foreign=Packet()
    root=SimpleNamespace(first=first,alias=alias,second=second,Packet=Packet)
    names=['first','alias','second','Packet.__call__']
    cat={'apis':[{'api_id':'python:torch.'+n,'qualified_name':'torch.'+n,'language':'python','kind':'function'} for n in names]}
    with collecting(root,cat) as (collector,hits,gaps):
        token=collector.enter(('test','call'))
        first();second();foreign()
        collector.leave(token)
    objects=[ids for owner,ids in hits if ids!=('python:torch.Packet.__call__',)]
    assert objects==[('python:torch.alias','python:torch.first'),('python:torch.second',)]
    assert sum(ids==('python:torch.Packet.__call__',) for owner,ids in hits)==3
    assert gaps==[]
