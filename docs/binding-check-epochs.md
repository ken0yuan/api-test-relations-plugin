# Exact binding checks during compiler workloads

Assisted-by: OpenAI Codex

The CPython 3.11 observer used to traverse a mixed array of namespace objects on
every Python and native call. Inductor workloads with many small compilations
made this check a substantial CPU cost. This change keeps every observation and
the original binding, alias, source location, and test ownership rules.

The filter now stores dictionary-version addresses and structure addresses in
separate compact arrays. Every check still verifies class MRO and object type
identity. Instance dictionaries still use CPython's original dictionary accessor,
including whole-dictionary replacement and native/variable object layouts.

A private dictionary alternates between the two Boolean singletons. CPython
3.11 assigns a globally unique incrementing version to each dictionary state.
Consecutive private versions prove that no other dictionary changed between
checks. Only in that case can the unchanged dictionary-version scan be reused.
Any intervening dictionary state change causes the complete original dictionary
comparison. This is independent of elapsed time, event count, API name, or test
name. Structure checks are unconditional, because class and MRO changes need not
write the watched dictionary. Zero or wrapping epoch values take the full path.

The trace guard also records frame flags in its own freshly allocated private
dictionary. Each successful write to that dictionary is accounted by its exact
new version, so these bookkeeping writes do not invalidate a clean workload
snapshot. A gap in the version sequence latches a required full scan; later
private writes cannot hide an intervening workload or client callback mutation.
Client profiler/trace behavior and every-opcode rearming remain unchanged.

During a registry refresh, repeated catalog prefixes share a static lookup only
while all of that lookup's original namespace versions and structures still
match. Each reuse performs the exact native dependency check. A changed prefix
is read again, and pre-read versions prevent changes during lookup from being
published as an already-current snapshot. The per-refresh cache is discarded
after the pass; it does not change historical aliases or binding resolution.

A detected difference is latched until the registry publishes a new snapshot.
Repeated native-filter, Python-dispatch, and refresh checks cannot accidentally
clear it. Existing historical callable identities and saved aliases remain in
the registry. Profiling clients and the protected profile/trace channels are
unchanged. The implementation remains specifically guarded for CPython 3.11.

Internal scan counters support regression assertions about work avoided without
wall-clock timing thresholds. Regressions cover mutation, unrelated dictionary
writes, repeated dirty checks, MRO changes, instance class changes, replacement
instance dictionaries, aliases, and ownership. Storage, retention, recovery,
strict append verification, and production sample selection are unchanged.
