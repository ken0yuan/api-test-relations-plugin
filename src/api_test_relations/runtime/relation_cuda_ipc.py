"""Explicit Linux CUDA VMM compatibility; no effect without opt-in.

pidfd_getfd is checked by Yama. Each producer permits only its actual parent
process and that parent's descendants, covering siblings in one test family.
This never sets PR_SET_PTRACER_ANY or changes a global kernel policy.
"""
import ctypes
import os
import sys

PR_SET_PTRACER = 0x59616D61


def allocator_config(environment):
    current = environment.get('PYTORCH_ALLOC_CONF')
    legacy = environment.get('PYTORCH_CUDA_ALLOC_CONF')
    if current and legacy and current != legacy:
        raise RuntimeError('Conflicting CUDA allocator environment settings')
    value = current or legacy or ''
    parts = [part.strip() for part in value.split(',') if part.strip()]
    for part in parts:
        if part.startswith('expandable_segments:') and part != 'expandable_segments:True':
            raise RuntimeError('CUDA IPC VMM conflicts with expandable_segments setting')
        if part.startswith('backend:') and part != 'backend:native':
            raise RuntimeError('CUDA IPC VMM requires the native allocator backend')
    if 'expandable_segments:True' not in parts:
        parts.append('expandable_segments:True')
    return ','.join(parts)


def configure():
    if os.environ.get('API_RELATION_CUDA_IPC_VMM') != '1':
        return None
    if sys.platform != 'linux':
        raise RuntimeError('CUDA IPC VMM compatibility requires Linux')
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        raise RuntimeError('CUDA IPC VMM must be configured before CUDA initialization')
    value = allocator_config(os.environ)
    parent = os.getppid()
    if parent <= 1:
        raise RuntimeError('CUDA IPC VMM requires a live supervising parent')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PTRACER, parent, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, 'Unable to authorize parent-family CUDA IPC descriptor import')
    os.environ['PYTORCH_ALLOC_CONF'] = value
    return {'mode': 'cuda_vmm_parent_family', 'allocator': value, 'pid': os.getpid(),
            'authorized_parent_pid': parent, 'permission_lifetime': 'this_process',
            'global_kernel_policy_changed': False, 'legacy_cuda_ipc_repaired': False}
