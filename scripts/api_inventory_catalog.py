"""Keep authoritative API labels; binding paths are separate, auditable metadata."""
import json
from pathlib import Path

PYTORCH_BINDING_PREFIXES = {
    'Tensor': 'torch.Tensor', 'Function': 'torch.autograd.Function',
    'FunctionCtx': 'torch.autograd.function.FunctionCtx',
    'Node': 'torch.autograd.graph.Node', 'Optimizer': 'torch.optim.Optimizer',
}


def inventory_catalog(path):
    rows=[];seen=set();roots={'torch'}
    with Path(path).open(encoding='utf-8-sig') as handle:
        for line in handle:
            name=json.loads(line)['api']
            if not isinstance(name,str) or not name or name in seen:
                raise ValueError('Empty/duplicate API inventory label')
            seen.add(name)
            prefix,sep,tail=name.partition('.')
            binding=PYTORCH_BINDING_PREFIXES[prefix]+sep+tail if sep and prefix in PYTORCH_BINDING_PREFIXES else name
            parts=binding.split('.')
            if len(parts)>1 and all(part.isidentifier() for part in parts):roots.add(parts[0])
            rows.append({'api_id':'python:'+name,'qualified_name':name,'lookup_name':binding,
                         'language':'python','kind':'function','inventory_name':name})
    return {'schema':'api_test_relations_inventory_catalog/1','namespace_roots':sorted(roots),'apis':rows}
