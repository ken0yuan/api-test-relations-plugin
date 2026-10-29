# Custom Inductor compile-pool ownership

The supported PyTorch `SubprocPool` sends jobs through a custom pipe to a sidecar,
which then uses `ProcessPoolExecutor`. Instrumenting the executor alone observes
the sidecar's pool-creation test instead of the original job submitter.

The adapter carries the submitting full nodeid and phase in a versioned envelope
around the original pickled job bytes. It restores ownership before sidecar
submission, worker unpickling/execution/result pickling, parent result decoding,
and Future callbacks. Job ids, original payload bytes, outputs and workload
exceptions remain under upstream control. Unowned legacy messages pass through.
The exact upstream source hash and method origins are checked before patching;
an unsupported implementation leaves a capture gap instead of a silent claim.
The adapter is installed after the module has finished importing and removed with
the process bridge. It does not edit the PyTorch source files or disable tracing.

Task submit/finish records retain causal ownership across the sidecar and worker
processes. Per-process bootstrap lifetime records are separate from those task
records. A child's still-entered bootstrap phase must not turn an already closed
parent test phase into an interrupted phase at a scheduler checkpoint.

An isolated test covers A creating the pool, A completing its parent phase before
its asynchronous job finishes, B submitting a job with different actual aliases,
payload/result decoding, completion callbacks, and a preserved workload error.
A separate real CUDA control checks complete API/alias/site/phase/owner records.
The production sample set and completed-nodeid exclusions are unchanged.

Historical receipts identify captures that used the affected channel and preserve
the original worker/sidecar evidence. Exposure alone does not establish which API
records are wrong. Missing original submission context cannot be reconstructed
by copying the bootstrap owner's relationships to other tests. Historical rows
and provenance are retained; source-level audit findings are recorded separately.

Assisted-by: OpenAI Codex
