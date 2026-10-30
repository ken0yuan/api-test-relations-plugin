"""Prepare a portable CUDA/NPU run from exact JSONL inventories; never execute tests."""
import argparse
from collections import defaultdict
import fnmatch
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from instance_sharding import partition_nodes, validate_plan
from scheduler_backends import backend_kind
from scheduler_state import create_database
from api_inventory_catalog import inventory_catalog
from single_scheduler_helpers import read, save, sha, backend_exclusion_reason


def prepare(out, config, inventory):
    config_path = Path(config).resolve()
    meta = read(config_path); base = config_path.parent
    def resolved(value):
        path = Path(value).expanduser()
        return str((base/path).resolve() if not path.is_absolute() else path.resolve())
    for key in ['repo', 'catalog', 'api_inventory', 'plugin_root', 'package_site', 'device_lock_dir', 'publish_root', 'short_temp_root', 'worker_cwd']:
        if meta.get(key):
            meta[key] = resolved(meta[key])
    python = meta.get('python', sys.executable)
    meta['python'] = str(Path(shutil.which(python) or resolved(python)).absolute())
    if not meta.get('repo'):
        raise ValueError('Config requires repo')
    info = json.loads(subprocess.check_output([meta['python'], '-I', '-c',
        'import json,sys,sysconfig;from pathlib import Path;'
        'sys.path.insert(0,sys.argv[1]) if sys.argv[1] else None;import api_test_relations as p;'
        'print(json.dumps(dict(plugin_root=str(Path(p.__file__).parent.parent),'
        'catalog=str(Path(p.__file__).parent/"catalogs/pytorch-api-ledger.json"),'
        'package_site=sysconfig.get_path("purelib"),version=p.__version__,python=list(sys.version_info[:2]))))',meta.get('plugin_root','')],
        cwd=base, text=True, encoding='utf-8'))
    if info['python'] != [3, 11]:
        raise ValueError('Current collector requires CPython 3.11')
    for key in ['plugin_root', 'catalog', 'package_site']:
        meta.setdefault(key, info[key])
    meta.update(backend=backend_kind(meta), execution_results=True)
    meta.setdefault('device_filter', 'privateuse1,cpu' if meta['backend'] == 'npu' else 'cuda')
    meta.setdefault('case_timeout_seconds', 600)
    meta.setdefault('collection_timeout_seconds', 300)
    meta.setdefault('module_timeout_seconds', 21600)
    meta.setdefault('child_wait_seconds', 120)
    meta.setdefault('dispatch_interval_seconds', 0)
    meta.setdefault('phase_order', ['regular', 'distributed'])
    meta.setdefault('dependency_paths', [])
    meta['dependency_paths'] = [resolved(p) for p in meta['dependency_paths']]
    if meta.get('disk_roots'):
        meta['disk_roots'] = [resolved(p) for p in meta['disk_roots']]
    meta['collector_version'] = info['version']
    reserved = {'CUDA_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','PYTHONPATH','RELATION_CAPTURE_ATTEMPT',
                'RELATION_RESUME_SELECTION','RELATION_EXECUTION_EVENTS','PYTEST_PLUGINS','PYTEST_ADDOPTS'}
    if reserved & meta.get('environment', {}).keys():
        raise ValueError('Use dedicated device/path settings instead of reserved environment keys')
    for key in ['case_timeout_seconds','collection_timeout_seconds','module_timeout_seconds','cases_per_shard']:
        if key in meta and (not isinstance(meta[key], int) or meta[key] < 1):
            raise ValueError(key+' must be a positive integer')
    grouped = defaultdict(list); provenance = []; seen = set()
    paths = [Path(p).resolve() for p in inventory] if inventory else [Path(resolved(p)) for p in meta.get('inventories', [])]
    if not paths:
        raise ValueError('At least one exact JSONL inventory is required')
    for path in paths:
        count = 0
        with path.open(encoding='utf-8-sig') as handle:
            for count, line in enumerate(handle, 1):
                item = json.loads(line); file = item['file'].replace('\\', '/').removeprefix('pytorch/')
                relative = PurePosixPath(file)
                if relative.is_absolute() or '..' in relative.parts or not file.startswith('test/'):
                    raise ValueError('Invalid test source path')
                name = item.get('name')
                if name is None:
                    node = item['nodeid'].removeprefix('pytorch/')
                    prefix = file+'::'
                    if not node.startswith(prefix):
                        raise ValueError('Inventory nodeid/file mismatch')
                    name = node[len(prefix):]
                if not isinstance(name, str) or not name:
                    raise ValueError('Empty test name')
                node = 'pytorch/'+file+'::'+name
                if node in seen:
                    raise ValueError('Duplicate inventory identity: '+node)
                seen.add(node); grouped[file].append(node)
        provenance.append({'path': str(path), 'sha256': sha(path), 'records': count})
    root = Path(out).resolve()
    if root.exists():
        raise ValueError('Run directory already exists; use --resume on a prepared run')
    rows = []; shards = []; remaining = {}; dependencies = {}
    for file, nodes in sorted(grouped.items()):
        source = Path(meta['repo'])/file
        if not source.is_file():
            raise ValueError('Test source is missing: '+str(source))
        row = {'module': file, 'file': 'pytorch/'+file, 'source': str(source), 'sha256': sha(source),
               'runner': 'pytest', 'nodeids': nodes,
               'resource': 'all' if any(fnmatch.fnmatchcase(file,p) for p in meta.get('distributed_patterns',['test/distributed/**'])) else 1,
               'module_exclusive': any(fnmatch.fnmatchcase(file,p) for p in meta.get('module_exclusive_patterns', []))}
        rows.append(row); remaining[file] = nodes
        number = math.ceil(len(nodes)/meta.get('cases_per_shard', 512))
        shards.extend(partition_nodes(file, nodes, number))
        dependencies[file] = [resolved(p) for pattern, paths in meta.get('module_dependency_paths', {}).items()
                              if fnmatch.fnmatchcase(file, pattern) for p in paths]
    meta['module_dependency_paths'] = dependencies
    plan = {'schema': 'api_test_relations_instance_plan/1', 'remaining': remaining, 'shards': shards}
    validate_plan(plan, {r['module']:r for r in rows}, lambda row:backend_exclusion_reason(meta,row))
    root.mkdir(parents=True)
    for name in ['logs','attempts','selections','receipts','preserved','launches','execution-events','execution-xml']:
        (root/name).mkdir()
    if meta.get('api_inventory'):
        catalog=inventory_catalog(meta['api_inventory'])
        meta['catalog']=str(root/'api-inventory-catalog.json')
        save(meta['catalog'],catalog)
        meta['api_inventory_provenance']={'path':meta['api_inventory'],'sha256':sha(meta['api_inventory']),'records':len(catalog['apis'])}
    meta.setdefault('no_default_excludes',True)
    meta.update(tests=[{k:v for k,v in row.items() if k != 'nodeids'} for row in rows],
                database=str(root/'relations.sqlite'), publish_root=meta.get('publish_root', str(root)),
                inventories=provenance, inventory_count=len(seen))
    meta['commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=meta['repo'],text=True).strip() if (Path(meta['repo'])/'.git').exists() else None
    scripts = Path(__file__).resolve().parent
    frozen = list(scripts.glob('*.py')) + list((Path(meta['plugin_root'])/'api_test_relations').rglob('*.py'))
    frozen += [Path(meta['catalog'])]
    frozen += list((Path(meta['plugin_root'])/'api_test_relations/runtime').glob('*.c'))
    frozen += list((Path(meta['plugin_root'])/'api_test_relations/runtime').glob('*.h'))
    meta['frozen_files'] = {str(path):sha(path) for path in frozen}
    create_database(meta['database'], read(Path(meta['catalog'])), rows)
    save(root/'instance-plan.json', plan)
    save(root/'queue.json', {row['module']:'pending' for row in rows})
    save(root/'manifest.json', meta)
    return {'out': str(root), 'backend': meta['backend'], 'cases': len(seen), 'modules': len(rows),
            'shards': len(shards), 'distributed_cases': sum(len(r['nodeids']) for r in rows if r['resource']=='all')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--config',required=True,type=Path,help='Deployment paths and backend policy JSON')
    parser.add_argument('--inventory',action='append',default=[],help='Repeat to include the historical skipped inventory too')
    args = parser.parse_args()
    print(json.dumps(prepare(args.out,args.config,args.inventory),ensure_ascii=False))


if __name__ == '__main__':
    main()
