"""Device discovery and leases; no test or callable-observation policy."""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def backend_kind(manifest):
    value = manifest.get('backend')
    if value is None:
        value = 'npu' if any(x in manifest.get('device_filter', '').split(',') for x in ('npu', 'privateuse1')) else 'cuda'
    if value not in ('cuda', 'npu'):
        raise ValueError('backend must be cuda or npu')
    return value


def parse_npu(mapping, usage):
    chips = []
    for line in mapping.splitlines():
        match = re.match(r'^\s*(\d+)\s+(\d+)\s+(\d+)\s+', line)
        if match:
            card, chip, logical = match.groups()
            chips.append((logical, card, chip))
    memory = {}; card = None
    for line in usage.splitlines():
        cells = [part.strip() for part in line.strip().strip('|').split('|')]
        if len(cells) < 3:
            continue
        header = re.match(r'^(\d+)\s+\S+', cells[0])
        if header and cells[1] in ('OK', 'Warning', 'Alarm'):
            card = header.group(1)
        elif card is not None and cells[0].isdigit() and ':' in cells[1]:
            pairs = [(int(a), int(b)) for a, b in re.findall(r'(\d+)\s*/\s*(\d+)', line)]
            nonempty = [(a, b) for a, b in pairs if b > 0]
            if nonempty:
                used, total = nonempty[-1]
                if used > total:
                    raise ValueError('NPU used memory exceeds capacity')
                memory[card, cells[0]] = total-used
    if not chips or any((card, chip) not in memory for _, card, chip in chips):
        raise ValueError('Unrecognized or incomplete npu-smi device/memory output')
    return [{'id': logical, 'free_mib': memory[card, chip], 'card': card, 'chip': chip}
            for logical, card, chip in chips]


class DeviceBackend:
    def __init__(self, manifest):
        self.manifest = manifest
        self.kind = backend_kind(manifest)
        self.ids = None

    def command(self, args):
        result = subprocess.run(args, capture_output=True, text=True, encoding='utf-8',
                                errors='replace', timeout=self.manifest.get('probe_timeout_seconds', 5),
                                env={**os.environ, **self.manifest.get('environment', {})})
        if result.returncode:
            raise RuntimeError('Device probe failed: '+result.stderr[-1000:])
        return result.stdout

    def sample(self):
        try:
            custom = self.manifest.get('device_probe_command')
            if custom:
                devices = json.loads(self.command(custom))['devices']
            elif self.kind == 'cuda':
                output = self.command(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                       '--format=csv,noheader,nounits'])
                devices = [{'id': row[0].strip(), 'uuid': row[1].strip(),
                            'free_mib': int(row[2].strip())} for row in csv.reader(output.splitlines()) if row]
            else:
                devices = parse_npu(self.command(['npu-smi', 'info', '-m']),
                                    self.command(['npu-smi', 'info']))
            by_id = {}
            for device in devices:
                device = dict(device, id=str(device['id']))
                if int(device['free_mib']) < 0 or device['id'] in by_id:
                    raise ValueError('Invalid/duplicate device inventory')
                device['free_mib'] = int(device['free_mib'])
                by_id[device['id']] = device
            mask_name = 'CUDA_VISIBLE_DEVICES' if self.kind == 'cuda' else 'ASCEND_RT_VISIBLE_DEVICES'
            inherited = os.environ.get(mask_name)
            aliases = {d.get('uuid', key): key for key, d in by_id.items()}
            def resolve(values):
                result = [aliases.get(str(value), str(value)) for value in values]
                if not result or len(set(result)) != len(result) or not set(result) <= by_id.keys():
                    raise ValueError('Requested visible devices are absent or duplicated')
                return result
            visible = resolve(inherited.split(',')) if inherited is not None else list(by_id)
            selected = resolve(self.manifest['device_ids']) if self.manifest.get('device_ids') is not None else visible
            if not set(selected) <= set(visible):
                raise ValueError('Configured device is outside the inherited visibility boundary')
            if not selected:
                raise ValueError('No accelerator devices available')
            if self.ids is not None and selected != self.ids:
                raise ValueError('Device inventory changed during the run')
            self.ids = selected
            return {'devices': [by_id[key] for key in selected], 'gpu_free_mib': max(by_id[k]['free_mib'] for k in selected),
                    'device_query': {'state': 'ok', 'backend': self.kind}}
        except (OSError, ValueError, KeyError, IndexError, RuntimeError, subprocess.SubprocessError) as exc:
            return {'devices': [], 'gpu_free_mib': None,
                    'device_query': {'state': 'error', 'backend': self.kind, 'reason': repr(exc)}}


def host_resources(manifest, root, database):
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines())
    roots = manifest.get('disk_roots') or [str(root), str(Path(database).parent)]
    available = [shutil.disk_usage(path).free / 2**30 for path in roots]
    return {'memory_available_mib': int(memory['MemAvailable'].split()[0])//1024,
            'host_c_free_gib': min(available), 'database_volume_free_gib': min(available),
            'disk_roots': roots}


class DeviceLeases:
    """Locks survive supervisor death when passed to its worker subprocess."""
    def __init__(self, devices, backend, directory):
        self.devices = list(devices); self.backend = backend; self.owned = {}
        self.directory = Path(directory)
        if self.directory.is_symlink():
            raise ValueError('Device lock directory must not be a symlink')
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        stat = self.directory.stat()
        if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
            raise ValueError('Device lock directory must be private and owned by this user')

    def acquire(self, attempt, exclusive, eligible):
        busy = {d for row in self.owned.values() for d in row['devices']}
        choices = [d for d in self.devices if d in eligible and d not in busy]
        if exclusive and (busy or choices != self.devices):
            return None
        groups = [choices] if exclusive else [[d] for d in choices]
        for group in groups:
            handles = []
            try:
                for device in group:
                    name = hashlib.sha256((self.backend+':'+device).encode()).hexdigest()+'.lock'
                    fd = os.open(self.directory/name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                    handles.append(fd)
                    if os.fstat(fd).st_uid != os.getuid():
                        raise ValueError('Foreign device lock owner')
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                for fd in handles: os.close(fd)
                continue
            except BaseException:
                for fd in handles: os.close(fd)
                raise
            lease = {'devices': group, 'fds': handles}
            self.owned[attempt] = lease
            return lease
        return None

    def adopt(self, attempt, devices):
        busy = {d for row in self.owned.values() for d in row['devices']}
        if set(devices) & busy or not set(devices) <= set(self.devices):
            raise ValueError('Overlapping adopted device assignment')
        self.owned[attempt] = {'devices': list(devices), 'fds': []}

    def release(self, attempt):
        for fd in self.owned.pop(attempt)['fds']:
            os.close(fd)
