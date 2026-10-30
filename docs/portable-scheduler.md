# CUDA / NPU 全量用例调度

这套入口复用 `run_instance_shards.py` 的分片、原始证据校验和单写入者关系汇总。后端由配置指定为 `cuda` 或 `npu`，原来的 CUDA 清单仍可以使用。调度器脚本从当前仓库运行，采集包使用配置所指的安装环境；本次改动没有修改已发布的 0.1.12 wheel。

## 环境与准备

使用 Linux、CPython 3.11 和已安装的 `api-test-relations`、`pytest-timeout`。CUDA 需要可用的 `nvidia-smi`；NPU 需要已加载 CANN 环境、可导入的 `torch_npu` 和可用的 `npu-smi`。先在目标环境运行依赖检查，再准备清单。

路径由配置给出。相对配置路径以配置文件所在目录为基准，命令行路径以调用目录为基准。不会绑定某个服务器地址、用户目录、源码目录或虚拟环境位置。

配置示例（按实际部署填写）：

```json
{
  "backend": "npu",
  "repo": "../pytorch",
  "python": "../venv/bin/python",
  "inventories": ["test_cases.jsonl", "skipped_test_cases.jsonl"],
  "api_inventory": "api_names_all_sheets.jsonl",
  "cases_per_shard": 512,
  "distributed_patterns": ["test/distributed/**"],
  "module_exclusive_patterns": ["test/test_cpp_extensions*.py"],
  "case_timeout_seconds": 600,
  "collection_timeout_seconds": 300,
  "module_timeout_seconds": 21600,
  "child_wait_seconds": 120
}
```

`plugin_root` 和 `package_site` 默认从所选解释器推导，也可以显式配置。指定 `api_inventory` 时，用其中所有原名生成本次台账；未指定时才使用 `catalog` 或随包台账。原 API 名称和 ID 不改写，`Tensor.*` 等明确的 PyTorch 类路径通过单独的 `lookup_name` 绑定。无法唯一定位、未加载模块、非可调用对象等状态明确保留。第三方命名空间只观察测试已经加载的模块，不自动安装或导入第三方包。完整列表不代表每个条目都已观测到调用。

新计划默认取消隐式 API 排除，包含 `torch.compile`；`exclude_apis` 可以显式排除。原 pytest 入口保留兼容默认，单独使用时加 `--api-relations-no-default-excludes`。`api_binding_states` 记录各 API 曾出现的绑定状态及来源派发，不能把未解析算成没有测试调用。

其他环境差异使用 `dependency_paths`、按文件匹配的 `module_dependency_paths`、`environment`、`environment_unset`、`worker_cwd`、`short_temp_root`、`disk_roots`。不要把具体测试的临时兼容处理写入调度器。需要多卡的文件必须加入 `distributed_patterns`；它表达资源需求，不直接照搬 nightly 的业务分类名称。

```bash
python scripts/prepare_instance_run.py --config "$CONFIG" --out "$RUN"
python scripts/run_instance_shards.py --out "$RUN"
```

`--max` 是 `--max-workers` 的完整别名。它只设置普通 worker 的并发上限；例如 8 卡主机使用 `--max 8`，分布式阶段仍只有一个执行 worker 租用全部 8 卡。

两份 JSONL 都会纳入，历史跳过清单不会被再次过滤。用例按完整的 `file + name` 匹配，名字中的换行和 `::` 都保留。准备阶段拒绝重复身份、缺失文件和覆盖既有运行目录，并记录清单、源码和采集代码的校验值。

## 卡的使用与互斥

默认发现当前可见的全部卡，不填写 `--max-workers` 就使用相同数量的普通 worker。8 卡主机就是 8 个普通 worker，每个 worker 固定租用不同的一张物理卡，进程内使用映射后的设备 0。队列尾部不足 8 份工作、内存/磁盘不足、卡被占用或文件需要串行时，并发会相应减少。

分布式阶段同时只运行一个执行 worker，持有全部可见卡；测试内部可以创建多个 rank。两个阶段完全互斥，默认先普通后分布式，也可以通过 `phase_order` 调整。卡租约使用进程持有的文件锁，传给 worker，在调度进程意外退出后仍由存活 worker 持有。需要协调多个运行目录时，必须使用同一个 `device_lock_dir`。该锁约束使用本调度器的任务，不能约束不遵守锁的外部任务。

`device_ids` 可以显式限定资源池，继承的 CUDA/NPU 可见范围同样会受到尊重。CUDA 使用 `CUDA_VISIBLE_DEVICES`，NPU 使用 `ASCEND_RT_VISIBLE_DEVICES`；NPU 分支不调用 CUDA 工具。资源探测失败会停止新派发并显示原因，不复用旧读数。

## 结果、异常与续派

`relations.sqlite` 同时保存关系和执行账本，结果表与关系采集是否完整分别统计：

- `execution_attempts`：每次派发、设备分配、退出码和采集汇总状态。
- `execution_events`：持久化的收集、开始执行和 setup/call/teardown 事件。
- `execution_results`：每条完整用例身份对应的状态、原因、耗时、是否开始执行和所属派发。

通过、断言失败、setup/teardown 错误、有原因的跳过、预期失败、崩溃和超时都明确记录。未生成的精确身份记为 `not_collected`，文件导入失败记为 `collection_error`，这两种状态的 `test_started` 为 0。它们不能被说成测试体已经执行过；同文件已收集的其他用例仍会尝试运行。

pytest 先按用例超时；信号无法中断时，调度器在宽限期后保存栈并终止该派发。崩溃或超时只终结正在执行的那条，尚未开始的队尾用例继续派发。资源暂停保留待续状态。采集证据完整性、归属或冻结源码校验失败时停止新派发并保留证据，避免把不可信关系汇总为完成。

```bash
python scripts/scheduler_status.py --out "$RUN" --results-jsonl "$RESULTS"
python scripts/run_instance_shards.py --out "$RUN" --resume
```

只有旧调度器停止后才能 `--resume`。它接管仍存活的原 worker，依据 PID 和启动时间核对身份，不重复派发它们。已完成的用例结果不会被覆盖。每次派发还保留原始事件、JUnit XML、日志、回执和关系证据归档。

`pending = 0` 表示清单中每条都有终态记录，仍需检查 `not_collected`、`collection_error` 等结果，修复原因后安排单独复测。`capture_complete` 只表示 API 关系采集的完整性，不能用来统计测试通过数。

完整 API 清单与 8 卡 NPU、8 rank HCCL、`torch.compile`、NPU 可调用算子对象的实测，以及异常续派和重启回归，见[验证收据](validation/portable-scheduler.json)。CUDA 保留逻辑回归和模拟调度验证，未在本轮使用 CUDA 实机。

## 定期汇报和清理

### 采集器修复后的续跑

调试器或 profiler 的 Python 回调抛出异常时，CPython 会清除该用户回调。原生多路器现在保留原异常和用户回调停用语义，并在返回解释器前恢复独立采集钩子。插件自己的观察和导入包装代码标记为 Dynamo 跳过编译，调用到的用户代码仍可编译，避免自动断点重新进入插件包装器造成递归。真实未受保护的钩子替换仍会阻断汇总。验证见[修复收据](validation/callback-exit-recovery.json)。

冻结运行不能原地替换采集器。`prepare_recovery_run.py --previous "$OLD_RUN" --out "$NEW_RUN" --config "$CONFIG" --validation "$VALIDATION" --retry-collection-errors` 为已停止任务创建新运行，核验清单、测试源码及已验证采集器的一致性后，通过 SQLite 备份保留旧库。它只重派经过精确身份复测、证明采集完整的受影响用例，以及显式要求重试的导入失败用例；已完成采集拒绝重置。

重派前的执行结果保存在 `execution_result_history`，不完整关系与缺口保存在 `recovery_quarantine_*`，复核依据保存在 `recovery_reviews`。原运行和原始阻断证据保留。新目录首次启动使用普通启动命令，后续恢复才使用 `--resume`；旧运行不得同时启动。

`progress_report.py --out "$RUN" --prune` 提供累计数及相对上轮增量，区分已派发、实际开始执行、已产生终态、未收集、收集失败、超时、崩溃和普通测试失败。关系总数使用 `DISTINCT(api_key,test_key)`；含阶段和别名的 `observations` 行数另列。当前不区分直接和间接调用。

清理由独立维护进程完成，不替换运行中的冻结代码。`prune_instance_evidence.py` 默认只预检，加 `--apply` 后才删除。它复用入库的文件清单契约，校验正式库、已提交回执、归档哈希和生产者退出情况后，删除对应的 `preserved` 归档、私有临时/编译缓存和经过归属核验的短路径别名。未入库、核验失败、原始清理未完成或仍有生产者的证据一律保留。清理回执保存在运行目录，执行结果、调用位置证据、采集缺口和入库核验信息均在正式 SQLite 内；小型运行日志和事件文件暂留用于排障，最终交付物为单个 SQLite 文件。
