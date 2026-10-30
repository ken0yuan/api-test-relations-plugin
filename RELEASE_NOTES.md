# 0.1.13 — Direct and indirect test-body API calls

Classify each observed API before deduplication using the current pytest body's
function identity. Dynamic callable parameters and input construction in that body
are direct; helper and fused-API internals remain indirect. Unknown identities and
legacy evidence are not guessed. Keep both kinds and their own first sites when
the same API is called in both ways, independent of execution order.

Carry the identity through existing child and pool ownership contexts. Preserve
classified provenance through raw merging, verified compaction, single-store
append and `query --call-kind direct|indirect|unknown|all`. Original relation tables,
alias groups, completed identities and 0.1.12 zombie handling remain intact.
Checkpoint failure handling also retains both pending batches and records errors
without assuming the historical pending-key length.

See [0.1.13 release details](docs/release-0.1.13.md).

# 0.1.12 — Zombie-aware post-exit evidence handling

An exited, unreaped Linux producer still has a `/proc/PID` directory. Settle and
hot-journal recovery now recognize its `Z` state instead of refusing the operation
indefinitely. Settle, recovery, compact and child draining share one conservative
check: live or unreadable/unknown processes still block post-exit operations, and
recorded child start times continue to protect against PID reuse.

Already committed relations, aliases, sites and shard provenance are merged through
the existing validation path. Raw gaps and missing completion receipts remain gaps;
this does not recover uncommitted observations or prove complete capture. Settle
preserves its source and records the actual finalizer version.

Validation includes real unreaped children, committed relationship retention,
hot-journal rollback with incident copies, compact verification, live-writer refusal
and PID reuse. See [release notes](docs/release-0.1.12.md) for the verified scope.

# 0.1.7 — File-scoped collection gaps and sibling test imports

File-level pytest collection gaps are now retained as module-owned records only when their source path exactly matches that module. Unknown collection paths and test-phase gaps without a test owner still block verification. Each pytest worker also receives its own test file's parent directory on `PYTHONPATH`, fixing sibling imports such as `test/nn/attention/test_fa3.py` importing `_fa_test_common.py`.

The single-database schema, append verification, and completed full-nodeid exclusion remain unchanged. This fix does not rerun completed test instances.

# 0.1.6 — Incremental single-database runner

Add transactional append to the normalized single_database/1 baseline. Completed identities, including zero-observation tests and tests with existing gaps, are excluded before execution and rejected again at merge if a new phase/relationship names them. New rows are independently checked against raw shards and the destination. Replacing an interrupted instance requires the new observations to contain every prior positive API/phase/alias tuple; otherwise rollback and preserve raw evidence. Original completed provenance remains unchanged.

The cooperating CUDA runner in scripts/run_single_database.py uses one writer and at most four distinct module workers. It preserves per-attempt selection and process/task boundary evidence inside the database. Verified raw output and that attempt's private compiler/temp caches are removed after exit; unresolved merges stop dispatch. No full baseline snapshots are copied. Database writes use a separate zero-byte lock file to avoid DrvFS/Windows conflicts with SQLite locks. Interrupted phase ownership is recorded before termination so finally blocks cannot promote it to completed.

Limits remain explicit: the runner uses a prepared manifest and CUDA resource policy; it is not an automatic backend adapter. Existing positive evidence cannot prove zero omissions. The append command targets single_database/1, while compact's dictionary-view format remains a separate per-run export format. Neither command silently converts between formats.

# 0.1.5 — Script-verified single-database retention

New post-exit `compact --out RUN --database FILE` reconstructs all shards and child receipts without copying the raw directory. Independent raw-row checks reject conflicting ownership, paths, aliases, phases or first sites. Dictionary encoding removes repeated relation strings while a `relations` view preserves the query interface. Every merged table is compared in both directions against the compact database; foreign keys and integrity are checked before commit and integrity is checked again after reopening the committed file.

Only after exclusive publication, fsync, unchanged-source checks and dead-producer checks does the script delete the disposable run directory. Verification failure preserves the originals. Cleanup errors are reported as incomplete, with the verified database retained. Manifests, source metadata, hashes and verification receipt live inside that database. Existing destination files, nested destinations, symlinks, repository roots and unknown top-level files are refused. No AI decision is involved and no tests are rerun. This command handles one completed or interrupted run; cross-run append and scheduler integration remain separate work. Capture gaps are retained, not interpreted as zero omissions.

# 0.1.4 — Periodic durable relationship checkpoints

Each instrumented process commits buffered relations every second, including while a test phase is still running. The internal writer has no test owner or profiling guard; record owners, aliases and first call sites are captured by the original callback. Database access and buffer changes share a process-local lock. WAL transactions use synchronous=FULL; buffers clear only after a successful commit and survive rollback. Unchanged registry snapshots are not rewritten every interval.

Checkpoint status is available in shards/checkpoints/<pid>.json. A cooperating runner may atomically write shards/checkpoint-request.json with a unique token; each writer commits pending observations and blocks further record callbacks before acknowledging quiesced with that token. Wait for acknowledgements before termination. Missing acknowledgements are incomplete capture, never proof of zero calls. Normal phase/session end still flushes; interrupted phases never become completed just because their relations were saved.

These changes preserve committed observations after SIGKILL; unexpected power loss, SIGKILL before the next commit, a GIL held indefinitely by native code, or storage failure can still prevent the latest uncommitted batch being saved. Full zero-omission coverage is not claimed. Existing completed baseline observations are retained with their original version, not retroactively repaired.

# 0.1.3 — Exact-identity performance optimization

Native conservative rejection avoids Python callbacks for unrelated functions, but dirty bindings always resolve before rejection. Native version snapshots track dictionaries, object classes and MRO; unrelated namespace changes no longer rebuild the full ledger. Dependency-topology changes force rebuilding even when the current callable stays identical. Historical callable references and aliases remain observable.

Every-opcode profile rearming remains enabled. Only redundant per-opcode frame bookkeeping is skipped when there is no trace client. No sampling, API/test-specific exclusions, or reduction in child/thread attribution is introduced.

51 regression and real CUDA integration checks passed. Three completed synthetic real-PyTorch cases retain all 11 exact relation keys, aliases and sites; new child receipt is complete, no gaps. The fourth new compiler-heavy synthetic case finishes in 37.5 seconds for the whole optimized run; the old run timed out at 180 seconds. This is not a universal throughput guarantee. Existing child-exit/native/class-call coverage boundaries remain; ARM/NPU is not hardware-validated.

# 0.1.2 — 真实测试发现的收尾缺口

真实 PyTorch 对照发现 pytest 的 sessionfinish 汇总早于部分子进程退出：一个运行的原始分片最终有 381 条关系，当次汇总只有 370 条。新增 `settle --out ORIGINAL --destination NEW`，等待生产者退出后，从原始分片和最终回执重新生成独立汇总；原目录保留。修复不依赖具体 API、测试文件或编译器名称。

空数据库分片不再静默忽略，记录 `shard_without_metadata`。迟到的未完成回执也会补入，因此 gap 数可能增加；这表示更完整地报告边界，不能解释为已经补齐了子进程内部未落盘的 API。

39 项隔离回归通过，另 2 项真实 PyTorch/CUDA 集成检查通过。真实 CUDA 小样本确认 cProfile 期间及关闭后的 add/mul 调用恢复；两个上游 profiler 测试保留全部 193 条历史关系并新增 79 条。ARM/NPU 和全量吞吐仍未验证。旧版安装包、原始运行和历史基线保留。

# 0.1.1 — 三类历史缺陷修复

本次只修改独立插件，不修改被冻结的 v4r2 采集器，不恢复原测试，不修改或重采旧数据。0.1.0 安装包和源码包保留。

| 问题 | 修复 | 对应验证 |
|---|---|---|
| 多线程刷新注册表发生字典变化/半更新 | 锁内构建私有副本，发布完整映射；失败回退；遍历期间绑定变化保留脏标记 | 并发刷新、新命名空间增长、注入刷新异常、扫描时重新绑定 |
| cProfile / `_lsprof` 覆盖调用通知导致漏采 | 原生 opcode 守护在后续 Python 操作前恢复多路 profile；将事件转发给原生 profiler，保留客户端状态 | 提前创建、提前启用、同一行启停、上下文管理器、异常、线程、原生统计计数、既有 trace 客户端事件 |
| 子进程异常退出后热日志阻止只读合并 | 从调度器 v2 迁入受限恢复；关闭失败读取连接；保留原始字节、恢复副本验证、恢复原件核对、重试合并 | 已提交行保留、日志原件保留、活 PID 拒绝、无关损坏拒绝、真实关系分片恢复后进入汇总 |

36 项回归通过，2 项需要真实 PyTorch/CUDA 的验证跳过。使用隔离 CPython 3.11 / x86 Linux 环境及模拟 torch 命名空间，未进行真实 ARM/NPU 验收。

原生守护增加 opcode trace 开销；尚未做真实全量吞吐评估。任意代码同时破坏 trace 和 profile 仍属于显式覆盖边界。SQLite 恢复只能保留可恢复的已提交数据，不能恢复未提交事务中的观测。

服务器使用新包创建独立运行目录；不要在旧运行目录替换冻结 runtime。已有 ARM/NPU 环境与 CPython 3.11 的要求不变。
