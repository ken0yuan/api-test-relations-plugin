# Observed AOT constant formatting

Assisted-by: OpenAI Codex

CPython 3.11 profiling/tracing bypasses the specialized in-place Unicode
concatenation instruction. PyTorch commit `1ee9fbab` formats an 8 MiB tensor's
constants using a Python loop with `const_cpp += ...`; under observation this
repeatedly copies the growing string. Stack samples confirmed this exact loop
was still advancing, rather than waiting for CUDA, SQLite, or a compiler child.

The optional scheduler plugin `aot_const_format_compat` changes only the nested
`format_consts_to_cpp` code object. It starts a list, extends it in place with
one-element tuples, and joins once at return. Tuple extension avoids adding a
profiled `list.append` call for every constant byte. All original string
fragments, separators, alignment, symbols, and line breaks remain unchanged.
The compiler function object retains its identity. Ancestor code objects keep
their bytecode and metadata, replacing only the nested code constant; sibling
code objects are unchanged.
No profile/trace channel is disabled, and PyTorch files are not edited.

This is an explicit compatibility mode, not an API recognition rule or a new
upstream release. The scheduler enables it only for modules named in
`manifest.aot_const_format_modules`, using `-p aot_const_format_compat` and
`RELATION_AOT_CONST_FORMAT=1`. Installation occurs after ordinary collection
imports, before test bodies. An earlier installation changed import/alias
initialization order in a diagnostic comparison and was rejected. The final
implementation leaves that order unchanged.

Installation requires the exact codecache source SHA256
`c7873c46a80a2e72c3ab901aa1e18632f95e9dde7fa9baac98296a19b1a5be45`, exactly one
nested formatter, its expected signature, and the expected renderer structure.
Any mismatch aborts instead of silently changing an unknown implementation.
Receipts include original/replacement code hashes in
`aot-const-format-compat.json` and `environment.json`.

Validation compares byte and bytearray inputs, empty buffers, line boundaries,
every byte value, untouched code, callable identity, and rejection of changed
source/structure. Independent real observation comparisons check API identities,
complete aliases, call sites, phases, and test ownership. The final 8 MiB probe
completed in 4.75 seconds with full capture and exactly matched the unobserved
reference output hash. This timing is diagnostic evidence, not a regression
test assertion or a guarantee for other workloads.

Deployment pauses the previous supervisor through its original checkpoint
barrier and verified append path. Existing complete identities and their scoped
relations must remain unchanged before continuation. Only the interrupted AOT
module is requeued, with all completed full nodeids excluded. Existing evidence,
single-database schema, collection matching, XPU exclusions, and resource limits
are preserved. This does not repair older missing observations automatically.
