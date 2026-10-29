# Synthetic traceback line caches on CPython 3.11

PyTorch's `torch.utils._traceback.report_compile_source_on_error` creates a short
`eval` code object and copies an original, longer `co_linetable` onto it. This
preserves traceback position metadata. Under CPython 3.11 tracing,
`_PyCode_CreateLineArray` allocates for the short bytecode but fills entries for
the entire copied table. In a real minifier capture, an 11-unit code object had
a table covering 300 units. The resulting heap overwrite surfaced later in GC,
list iteration, or memory allocation.

The observer protects the exact native `CodeType.replace` return boundary using
the existing boundary adapter. While capture is active, a returned code object
with an overlong table receives a cache with exactly one entry per actual code
unit. Each entry comes from `PyCode_Addr2Line` for that valid offset. Other code
objects use CPython's ordinary cache creation. The interpreter owns and frees
the cache through its existing allocator and lifecycle.

This happens before the new code object reaches `eval`: initializing the cache
inside the first trace callback would be too late. The callable/descriptor
identity and calling convention are preserved. The code object, bytecode,
original line/column table, filename, and first line remain unchanged. Profiling
and opcode tracing stay enabled; API identity, alias resolution, callsite
selection, phase ownership, and storage verification retain their rules.

This adapter is specific to the supported CPython 3.11 layout. It covers code
replaced during active observation, including the minifier's synthetic traceback
path; it does not promise to validate arbitrary malformed code objects created
before capture or through every possible native constructor.

The regression runs under `PYTHONMALLOC=debug`, checks allocation guards through
process exit, verifies complete API/alias/site/test ownership records, preserves
saved native method identity and exceptions, and compares ordinary tracing
events. Production recovery must still pass the original raw mapping, prior
positive subset, completed-instance, phase, foreign-key, and integrity checks.
Failed capture evidence must be retained.

Assisted-by: OpenAI Codex
