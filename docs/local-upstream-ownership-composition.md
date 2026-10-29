# Upstream ownership fix in the local maintenance candidate

This candidate includes upstream b79891c51b0337a876fb9c43967eee870256c1bc
(release 0.1.11) and 7bfa203. The compile-pool task/result/Future ownership
bridge and exact binding-epoch optimization are applied without changing
their evidence or alias semantics.

The maintenance package retains its existing private 0.1.12 version number,
plus the previously validated lifecycle-audit change and the explicitly
authorized reexecution-union option. Exact deployed composition is identified
by the Git revision and source-file hashes, rather than the version alone.

The upstream instance-shard command is available but does not replace the
current task's eight-device, eight-preparer, one-writer scheduler.

Source-hash guards on the SubprocPool adapter remain strict. Unsupported
upstream implementations produce an explicit gap; they are never silently
claimed as correctly attributed or fully captured.

Assisted-by: OpenAI Codex
