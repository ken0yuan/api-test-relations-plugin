"""Four independent packaged-plugin captures; one aggregate writer, explicit lineage."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)); tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def proc(pid):
    try:
        root = Path('/proc', str(pid)); stat = (root/'stat').read_text().rpartition(')')[2].split()
        return {'pid': int(pid), 'start_ticks': stat[19], 'state': stat[0],
                'command': (root/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')}
    except (OSError, IndexError):
        return None


def live(worker):
    actual = proc(worker['pid'])
    return bool(actual and actual['state'] != 'Z' and actual['start_ticks'] == worker['start_ticks'])


def gpu_memory_sample():
    """Keep failures observable; an unavailable sample never authorizes dispatch.

    Preserve the existing command and five-second limit. No stale-value reuse,
    guessed free memory, device reset, or retry that extends the resource probe.
    """
    command = ['nvidia-smi', '--id=0', '--query-gpu=memory.free',
               '--format=csv,noheader,nounits']
    started = time.monotonic()
    detail = {'command':command, 'timeout_seconds':5, 'sampled_at':time.time()}
    def snippet(value):
        if isinstance(value, bytes): value = value.decode(errors='replace')
        return str(value or '')[:512]
    try:
        output = subprocess.check_output(command, text=True, timeout=5,
                                         stderr=subprocess.PIPE)
        value = output.strip()
        if not value.isascii() or not value.isdecimal():
            raise ValueError('expected one nonnegative integer in MiB')
        gpu = int(value)
        detail.update(state='ok', free_mib=gpu)
    except subprocess.TimeoutExpired as exc:
        gpu = None
        detail.update(state='timeout', exception=type(exc).__name__,
                      stdout=snippet(exc.output), stderr=snippet(exc.stderr))
    except subprocess.CalledProcessError as exc:
        gpu = None
        detail.update(state='command_failed', exception=type(exc).__name__,
                      returncode=exc.returncode, stdout=snippet(exc.output),
                      stderr=snippet(exc.stderr))
    except OSError as exc:
        gpu = None
        detail.update(state='os_error', exception=type(exc).__name__,
                      errno=exc.errno, error=snippet(exc))
    except ValueError as exc:
        gpu = None
        detail.update(state='invalid_output', exception=type(exc).__name__,
                      stdout=snippet(output), error=snippet(exc))
    except subprocess.SubprocessError as exc:
        gpu = None
        detail.update(state='subprocess_error', exception=type(exc).__name__,
                      error=snippet(exc))
    detail['elapsed_seconds'] = round(time.monotonic()-started, 3)
    return gpu, detail


def resources():
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    gpu, detail = gpu_memory_sample()
    return {'memory_available_mib': int(mem['MemAvailable'].split()[0])//1024,
            'gpu_free_mib': gpu, 'gpu_query':detail}


def can_launch(resource):
    return resource['memory_available_mib'] >= 3072 and resource['gpu_free_mib'] is not None and resource['gpu_free_mib'] >= 3072


def backend_exclusion_reason(manifest, row):
    """A CUDA campaign does not even collect XPU-only source files."""
    if manifest.get('device_filter') != 'cuda': return None
    for value in (row.get('module'), row.get('file')):
        if not value: continue
        parts = str(value).replace('\\', '/').lower().split('/')
        for part in parts:
            stem = part.removesuffix('.py')
            if stem == 'xpu' or stem == 'test_xpu' or stem.startswith('test_xpu_'):
                return 'excluded_backend_xpu'
    return None


def short_temp_alias(output, target):
    """Short AF_UNIX address spelling, with data still in attempt-private scratch."""
    base = Path('/tmp') / ('atr-' + str(os.getuid()))
    if base.is_symlink(): raise ValueError('Unsafe short-temp parent')
    base.mkdir(mode=0o700, exist_ok=True)
    info = base.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('Short-temp parent must be private and owned by this user')
    alias = base / hashlib.sha256(str(Path(output).absolute()).encode()).hexdigest()[:20]
    target = Path(target).absolute(); target.mkdir(parents=True, exist_ok=True)
    if alias.is_symlink():
        if os.readlink(alias) != str(target): raise ValueError('Short-temp alias target changed')
    elif alias.exists():
        raise ValueError('Refusing to replace short-temp content')
    else:
        alias.symlink_to(target, target_is_directory=True)
    return alias


def cleanup_temp_alias(worker, scratch):
    """Remove only this attempt's verified alias, never recursively delete /tmp."""
    value = worker.get('temp_alias')
    if not value: return  # Older adopted workers used the long scratch spelling.
    alias = Path(value)
    expected = Path('/tmp') / ('atr-' + str(os.getuid())) / hashlib.sha256(
        str(Path(worker['output']).absolute()).encode()).hexdigest()[:20]
    target = (Path(scratch) / 'tmp').absolute()
    if alias != expected: raise ValueError('Invalid short-temp alias ownership')
    if alias.is_symlink():
        if os.readlink(alias) != str(target): raise ValueError('Short-temp alias target changed')
        alias.unlink()
    elif alias.exists():
        raise ValueError('Short-temp alias was replaced by content')


def claim(db, rows):
    for row in rows:
        attempt = hashlib.sha256(row['module'].encode()).hexdigest()[:16]+'-'+str(time.time_ns())
        db.execute('BEGIN IMMEDIATE')
        changed = db.execute("UPDATE jobs SET state='collecting',attempt=? WHERE module=? AND state='pending'",
                             (attempt, row['module'])).rowcount
        db.commit()
        if changed:
            return row, attempt


def launch_spec(manifest, root, row, attempt):
    excluded = backend_exclusion_reason(manifest, row)
    if excluded: raise ValueError(excluded + ': ' + row['module'])
    repo = Path(manifest['repo']); scratch = root/'worker-runtime'/attempt
    output = root/'attempts'/attempt
    env = {k: v for k,v in os.environ.items() if not k.startswith('API_RELATION_')}
    env.pop('PYTEST_ADDOPTS', None); env.pop('PYTEST_PLUGINS', None)
    # PyTorch test files often import sibling helpers by bare module name
    # (for example test/nn/attention/test_fa3.py -> _fa_test_common.py).
    # Each worker owns exactly one source file, so exposing its parent directory
    # is scoped to that collection and preserves ordinary pytest import semantics.
    source_dir = str(Path(row['source']).resolve().parent) if row['runner'] != 'doctests' else ''
    pythonpath = [manifest['plugin_root'], str(Path(__file__).parent),
                  *manifest.get('module_dependency_paths', {}).get(row['module'], []),
                  *manifest.get('dependency_paths', []), source_dir,
                  str(repo/'test'), str(repo)]
    env.update(PYTHONPATH=':'.join(p for p in pythonpath if p),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTORCH_TESTING_DEVICE_ONLY_FOR=manifest['device_filter'],
        CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MAX_JOBS='2',
        TORCHINDUCTOR_COMPILE_THREADS='2', PYTORCH_NUM_PYTEST_RERUNS='0',
        RELATION_CAPTURE_ATTEMPT=str(output), RELATION_RESUME_SELECTION=str(root/'selections'/(attempt+'.json')))
    for name, part in [('TMPDIR','tmp'), ('TORCHINDUCTOR_CACHE_DIR','inductor'),
                       ('TRITON_CACHE_DIR','triton'), ('TORCH_EXTENSIONS_DIR','extensions'), ('XDG_CACHE_HOME','cache')]:
        path = scratch/part; path.mkdir(parents=True, exist_ok=True); env[name] = str(path)
    # multiprocessing.resource_sharer adds pymp-*/listener-* below TMPDIR.
    # Long run/attempt names exceed sockaddr_un even with collection disabled.
    alias = short_temp_alias(output, scratch/'tmp')
    env['TMPDIR'] = str(alias); env['RELATION_TEMP_ALIAS'] = str(alias)
    prefix = Path(manifest['python']).parent.parent
    env['PATH'] = str(prefix/'bin')+':'+env.get('PATH','')
    if (prefix/'bin/nvcc').exists():
        env.update(CUDA_HOME=str(prefix), CUDACXX=str(prefix/'bin/nvcc'))
        env['TORCH_CUDA_ARCH_LIST'] = manifest['cuda_capability']
        headers = list(prefix.glob('targets/*/include/cuda_runtime.h'))
        if headers: env['CPATH'] = str(headers[0].parent)+':'+env.get('CPATH','')
    cmd = [manifest['python'], '-m', 'pytest', row['source'], '-q', '--tb=no', '--no-header',
           '--continue-on-collection-errors', '-p', 'api_test_relations.pytest_plugin', '-p', 'retained_selection',
           '--api-relations-out', str(output), '--api-relations-repo', str(repo),
           '--api-relations-catalog', manifest['catalog'], '--api-relations-label', 'accelerator/cuda',
           '--api-relations-child-wait', '120', '--basetemp', str(scratch/'pytest')]
    if row['runner'] == 'doctests':
        cmd[3] = str(repo/'torch')
        cmd += ['-p', 'relation_doctest_plugin', '-o', 'python_files=__relation_no_ordinary_tests__',
                '--ignore-glob', '*distributed*', '--ignore', str(repo/'torch/_vendor')]
        env['TORCH_DOCTEST_CUDA'] = '1'; env.pop('TORCH_DOCTEST_DISTRIBUTED', None)
        if manifest.get('doctest_onnx'): env['TORCH_DOCTEST_ONNX'] = '1'
        else:
            cmd += ['--ignore', str(repo/'torch/onnx')]; env.pop('TORCH_DOCTEST_ONNX', None)
    if row['module'] in manifest.get('cuda_ipc_vmm_modules', []):
        cmd += ['--api-relations-cuda-ipc-vmm']
    if row['module'] in manifest.get('aot_const_format_modules', []):
        cmd += ['-p', 'aot_const_format_compat']
        env['RELATION_AOT_CONST_FORMAT'] = '1'
    return cmd, env, output


def attempt_processes(worker):
    expected = ('RELATION_CAPTURE_ATTEMPT='+worker['output']).encode()
    found = {}
    if live(worker): found[worker['pid']] = worker
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        try:
            if expected in (path/'environ').read_bytes().split(b'\0'):
                item = proc(int(path.name))
                if item and item['state'] != 'Z': found[item['pid']] = item
        except OSError: pass
    return found


def quiesce_attempt(worker, timeout=10):
    """Ask live instrumented processes to commit and block new record callbacks."""
    folder=Path(worker['shards']); token=str(time.time_ns())
    save(folder/'checkpoint-request.json',{'token':token,'requested_at':time.time()})
    deadline=time.monotonic()+timeout; stable_since=None; previous=None
    acknowledged=[];missing=[]
    while True:
        states={}
        for path in (folder/'checkpoints').glob('*.json'):
            try:
                state=read(path); actual=proc(state['pid'])
                if actual and actual['state']!='Z':states[state['pid']]=state
            except (ValueError,OSError,KeyError):continue
        missing=[pid for pid,state in states.items() if state.get('state')!='closed' and (state.get('request')!=token or state.get('state')!='quiesced')]
        if live(worker) and worker['pid'] not in states:missing.append(worker['pid'])
        acknowledged=[pid for pid,state in states.items() if state.get('state')=='closed' or (state.get('request')==token and state.get('state')=='quiesced')]
        census=tuple(sorted(states))
        if not missing and census==previous:
            stable_since=stable_since or time.monotonic()
            if time.monotonic()-stable_since>=1:break
        else:stable_since=None
        previous=census
        if time.monotonic()>=deadline:break
        time.sleep(.2)
    report={'token':token,'acknowledged':acknowledged,'missing':missing,'at':time.time(),
            'complete_for_registered_live_writers':not missing}
    save(folder/'checkpoint-barrier.json',report)
    return report


def terminate_attempt(worker):
    worker['checkpoint_barrier'] = quiesce_attempt(worker)
    found = attempt_processes(worker)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for item in found.values():
            if live(item):
                try: os.kill(item['pid'], sig)
                except ProcessLookupError: pass
        if sig == signal.SIGTERM and found: time.sleep(1)
    return sorted(found)


