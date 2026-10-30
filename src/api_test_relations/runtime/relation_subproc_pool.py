"""Carry exact test ownership across the supported Inductor custom IPC pool.

The upstream pickled job/result bytes and job identifiers are preserved.  Only
the job transport gets a versioned envelope, removed before upstream decoding.
"""
from contextlib import contextmanager
import functools
import hashlib
import json
from pathlib import Path
import struct
import sys
import threading
import weakref

from relation_context import OWNER,TEST_FUNCTION

MODULE = 'torch._inductor.compile_worker.subproc_pool'
SOURCE_SHA256 = '5c95bd31af9266d201ebed9cc275a51948b682f6692af4513bbce99e59b2bea5'
MAGIC = b'\x00api-relation-subproc-owner\x00v1\x00'
MAX_CONTEXT = 1024 * 1024


def pack(context, payload):
    header = json.dumps(context, separators=(',', ':')).encode()
    if len(header) > MAX_CONTEXT:
        raise ValueError('Subproc owner context too large')
    return MAGIC + struct.pack('!I', len(header)) + header + payload


def unpack(data):
    if not data.startswith(MAGIC):
        return None, data
    offset = len(MAGIC)
    if len(data) < offset + 4:
        raise ValueError('Truncated Subproc owner header')
    size = struct.unpack('!I', data[offset:offset + 4])[0]
    if size > MAX_CONTEXT or len(data) < offset + 4 + size:
        raise ValueError('Invalid Subproc owner length')
    context = json.loads(data[offset + 4:offset + 4 + size])
    if (not isinstance(context, dict) or
            not all(isinstance(context.get(k), str) for k in ('token', 'nodeid', 'phase')) or
            context.get('transport') != 'subproc_pool_task' or
            not isinstance(context.get('settings'), dict)):
        raise ValueError('Invalid Subproc owner context')
    return context, data[offset + 4 + size:]


@contextmanager
def owned(context):
    if context is None:
        yield
        return
    token = OWNER.set((context['nodeid'], context['phase']))
    test_token = TEST_FUNCTION.set(context.get('test_function'))
    try:
        yield
    finally:
        TEST_FUNCTION.reset(test_token)
        OWNER.reset(token)


class SubprocPoolBridge:
    def __init__(self, bridge):
        self.bridge = bridge
        self.local = threading.local()
        self.lock = threading.RLock()
        self.pending = weakref.WeakKeyDictionary()
        self.installed = False
        self.rejected = False
        self.patches = []

    def patch(self, obj, name, replacement):
        original = vars(obj)[name]
        self.patches.append((obj, name, original, replacement))
        setattr(obj, name, replacement)

    def maybe_install(self):
        if self.installed or self.rejected or getattr(self.local, 'installing', False):
            return
        module = sys.modules.get(MODULE)
        if module is None or getattr(getattr(module, '__spec__', None), '_initializing', False):
            return
        pool = getattr(module, 'SubprocPool', None)
        main = getattr(module, 'SubprocMain', None)
        if pool is None or main is None:
            return
        self.local.installing = True
        try:
            file = Path(module.__file__)
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            methods = [(pool, 'submit', 300), (pool, '_send', 332),
                       (pool, '_handle_job_result', 371), (pool, 'shutdown', 594),
                       (main, '_submit_inner', 804), (main, 'do_job', 921)]
            valid = digest == SOURCE_SHA256
            for cls, name, line in methods:
                value = vars(cls).get(name)
                fn = value.__func__ if isinstance(value, staticmethod) else value
                code = getattr(fn, '__code__', None)
                valid = valid and code is not None and code.co_filename == str(file) and code.co_firstlineno == line
            if not valid:
                self.rejected = True
                owner = OWNER.get() or ('', 'session')
                self.bridge.plugin.gap(owner, 'subproc_pool_adapter_unsupported:' + digest)
                return
            self.install(module)
            from relation_process import document
            document('adapters', 'subproc-pool-' + str(self.bridge.pid),
                     {'pid': self.bridge.pid, 'source': str(file), 'source_sha256': digest,
                      'adapter': 'subproc-owner-envelope/1', 'profile_trace_preserved': True})
        finally:
            self.local.installing = False

    def install(self, module):
        from relation_process import context_for, document, run_owned
        pool, main = module.SubprocPool, module.SubprocMain
        original_submit, original_send = pool.submit, pool._send
        original_result, original_shutdown = pool._handle_job_result, pool.shutdown
        original_inner, original_job = main._submit_inner, main.do_job
        adapter = self

        @functools.wraps(original_submit)
        def submit(instance, job_fn, *args, **kwargs):
            owner = OWNER.get()
            context = context_for(owner, 'subproc_pool_task') if owner is not None else None
            stack = getattr(adapter.local, 'submissions', ())
            adapter.local.submissions = (*stack, (instance, context))
            try:
                return original_submit(instance, job_fn, *args, **kwargs)
            finally:
                adapter.local.submissions = stack

        @functools.wraps(original_send)
        def send(instance, msg_header, job_id=-1, data=b''):
            stack = getattr(adapter.local, 'submissions', ())
            context = stack[-1][1] if stack and stack[-1][0] is instance else None
            if context is not None and msg_header == module.MsgHeader.JOB:
                # Publish identity before the sidecar can return a result.
                document('tasks', context['token'], context)
                with adapter.lock:
                    adapter.pending.setdefault(instance, {})[job_id] = context
                pending = instance.pending_jobs[job_id]
                original_callbacks = pending.future._invoke_callbacks

                @functools.wraps(original_callbacks)
                def callbacks():
                    with owned(context):
                        return original_callbacks()
                pending.future._invoke_callbacks = callbacks
                data = pack(context, data)
            return original_send(instance, msg_header, job_id, data)

        @functools.wraps(original_inner)
        def inner(instance, job_id, data):
            context, _ = unpack(data)
            with owned(context):
                return original_inner(instance, job_id, data)

        @functools.wraps(original_job)
        def job(pickler, job_id, data):
            context, payload = unpack(data)
            if context is None:
                return original_job(pickler, job_id, payload)
            # Includes upstream unpickling, callable execution and result pickling.
            return run_owned(context, original_job, (pickler, job_id, payload), {})

        @functools.wraps(original_result)
        def result(instance, job_id, data, *, failed=False):
            with adapter.lock:
                context = adapter.pending.get(instance, {}).get(job_id)
            try:
                with owned(context):
                    return original_result(instance, job_id, data, failed=failed)
            finally:
                with adapter.lock:
                    adapter.pending.get(instance, {}).pop(job_id, None)

        @functools.wraps(original_shutdown)
        def shutdown(instance):
            try:
                return original_shutdown(instance)
            finally:
                with adapter.lock:
                    adapter.pending.pop(instance, None)

        for cls, name, fn in [(pool, 'submit', submit), (pool, '_send', send),
                              (main, '_submit_inner', inner), (main, 'do_job', staticmethod(job)),
                              (pool, '_handle_job_result', result), (pool, 'shutdown', shutdown)]:
            self.patch(cls, name, fn)
        self.installed = True

    def close(self):
        for obj, name, original, replacement in reversed(self.patches):
            if vars(obj).get(name) is replacement:
                setattr(obj, name, original)
        self.patches.clear()
        self.pending.clear()
        self.installed = False
