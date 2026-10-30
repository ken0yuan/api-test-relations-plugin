"""Opt-in pytest adapter; CUDA IPC execution compatibility requires a separate flag."""
import importlib
import json
import os
from pathlib import Path
import sys
import pytest
from .cli import DEFAULT_CATALOG, digest, finalize, initialize, save


def pytest_addoption(parser):
    group = parser.getgroup('api-test-relations')
    group.addoption('--api-relations-out', help='New/empty evidence directory; enables capture')
    group.addoption('--api-relations-repo', help='Repository root used for pytorch/ source paths')
    group.addoption('--api-relations-catalog', default=str(DEFAULT_CATALOG))
    group.addoption('--api-relations-import', action='append', default=[], help='Explicit extension import before resolution, e.g. torch_npu')
    group.addoption('--api-relations-exclude', action='append', default=[])
    group.addoption('--api-relations-no-default-excludes', action='store_true',
                    help='Use the complete supplied catalog, including torch.compile; explicit exclusions still apply')
    group.addoption('--api-relations-label', default='unspecified', help='Metadata only, e.g. ascend-npu; does not set a device')
    group.addoption('--api-relations-child-wait', type=int, default=120)
    group.addoption('--api-relations-cuda-ipc-vmm', action='store_true',
                    help='Explicit Linux CUDA IPC compatibility: VMM allocator and per-process parent-family descriptor permission')


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    output = config.getoption('--api-relations-out')
    if not output: return
    if hasattr(config, 'workerinput') or getattr(config.option, 'numprocesses', 0):
        raise pytest.UsageError('v0.1 does not support pytest-xdist; use independent pytest processes with separate output directories')
    repo = config.getoption('--api-relations-repo')
    if not repo: raise pytest.UsageError('--api-relations-repo is required')
    if config.getoption('--api-relations-child-wait') < 0: raise pytest.UsageError('child wait must be nonnegative')
    if os.environ.get('API_RELATION_SHARDS') or os.environ.get('API_RELATION_PARENT'):
        raise pytest.UsageError('Nested/duplicate collectors are not supported')
    root = Path(output).resolve()
    imports = config.getoption('--api-relations-import')
    exclusions=list(config.getoption('--api-relations-exclude'))
    if not config.getoption('--api-relations-no-default-excludes'):
        exclusions.append('torch.compile')
    manifest = initialize(root, Path(repo), Path(config.getoption('--api-relations-catalog')),
                          exclusions, config.getoption('--api-relations-label'), imports,
                          runtime=os.environ.get('TRACE_API_CHAIN_OBSERVER_RUNTIME'))
    runtime = Path(manifest['runtime_directory'])
    sys.path.insert(0, str(runtime))
    updates = {'API_RELATION_REPO': manifest['repo'], 'API_RELATION_SHARDS': str(root / 'shards'),
               'API_RELATION_CATALOG': manifest['catalog'], 'API_RELATION_EXCLUDE': json.dumps(manifest['excluded_apis'])}
    previous = {key: os.environ.get(key) for key in updates}
    if config.getoption('--api-relations-cuda-ipc-vmm'):
        previous.update({key: os.environ.get(key) for key in ('API_RELATION_CUDA_IPC_VMM', 'PYTORCH_ALLOC_CONF')})
        updates['API_RELATION_CUDA_IPC_VMM'] = '1'
    os.environ.update(updates)
    config._relation_state = {'root': root, 'previous': previous, 'runtime': str(runtime), 'observer': None}
    try:
        compatibility = None
        if config.getoption('--api-relations-cuda-ipc-vmm'):
            from relation_cuda_ipc import configure
            compatibility = configure()
        for name in imports: importlib.import_module(name)
        from relation_plugin import RelationPlugin
        observer = RelationPlugin()
        config._relation_state['observer'] = observer
        config.pluginmanager.register(observer, 'api-test-relations-observer')
        versions = {name: {'version': str(getattr(sys.modules.get(name), '__version__', 'unknown')),
                           'file': str(getattr(sys.modules.get(name), '__file__', 'unknown'))}
                    for name in ['torch', *imports]}
        environment = {'modules': versions, 'backend_label_is_metadata_only': True}
        if compatibility is not None:
            environment['runtime_compatibility'] = compatibility
        save(root / 'environment.json', environment)
    except BaseException as exc:
        save(root / 'capture-error.json', {'stage': 'configure', 'reason': repr(exc)})
        raise


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session):
    state = getattr(session.config, '_relation_state', None)
    if not state: return
    repo = Path(os.environ['API_RELATION_REPO'])
    sources = {}
    for item in session.items:
        path = Path(item.path).resolve()
        key = 'pytorch/' + path.relative_to(repo).as_posix()
        if key not in sources:
            sources[key] = {'absolute_path': str(path), 'sha256': digest(path), 'tests': []}
        _, separator, suffix = item.nodeid.partition('::')
        sources[key]['tests'].append(key + separator + suffix)
    save(state['root'] / 'test-sources.json', sources)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    state = getattr(session.config, '_relation_state', None)
    if not state: return
    try: finalize(state['root'], session.config.getoption('--api-relations-child-wait'))
    except BaseException as exc:
        save(state['root'] / 'capture-error.json', {'stage': 'merge', 'reason': repr(exc)})
        raise


def pytest_unconfigure(config):
    state = getattr(config, '_relation_state', None)
    if not state: return
    observer = state['observer']
    if observer is not None:
        observer.stop_checkpoints()
        observer.bridge.close()
        if observer.collector.enabled: observer.collector.close()
        observer.db.close()
    for key, value in state['previous'].items():
        if value is None: os.environ.pop(key, None)
        else: os.environ[key] = value
    if state['runtime'] in sys.path: sys.path.remove(state['runtime'])
