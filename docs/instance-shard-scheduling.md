# Instance shard scheduling

`scripts/run_instance_shards.py --out /path/to/prepared-run --max-workers N`
requires a positive concurrency ceiling supplied by the caller. There is no
fixed four-worker maximum. Resource checks and a 30-second dispatch interval
may keep the live count below that ceiling. The deployment used for this CUDA
campaign passes `--max-workers 4`.

An explicitly prepared `instance-plan.json` partitions registered unfinished
`(module, full nodeid)` identities. Its shards must be disjoint and exactly
cover the declared remaining identities. Module IDs in the formal database are
unchanged: shard IDs identify scheduling assignments, not new test modules.

Each launch refreshes completed identities from the formal database and selects
only its own still-unfinished full nodeids. The selection adapter checks the
exact set before test execution. Other shards are recorded separately from
completed deselections. Tests, process receipts, tasks and raw relation owners
must all stay inside the launch's assignment before merge is allowed.

Workers have private ext4 caches, output and logs. The scheduler serializes the
unchanged strict `append_capture`, checks every original mapping/alias/site and
prior positive subset, and retains a SHA-verified private raw archive before
successful originals are disposed by the original append flow. Logs and private
scratch remain. No collector, identity, alias or relationship rule changes.

Resource admission keeps the existing 3072 MiB RAM/GPU and 10 GiB disk thresholds,
and reserves another 2048 MiB after the largest observed worker PSS estimate.
RAM below 2048 MiB checkpoint-stops the newest worker and lowers the effective
parallel capacity. Other healthy workers keep their original time limits.
The 21600-second attempt limit and bounded child wait remain. No-progress
continuations require diagnosis rather than silently rerunning indefinitely.

An unexpected owner, collection mismatch, missing checkpoint acknowledgement or
verification failure stops dispatch and preserves raw evidence. The ordinary
append guard against completed-instance replacement is never bypassed. XPU
source exclusions remain in the launch path.
