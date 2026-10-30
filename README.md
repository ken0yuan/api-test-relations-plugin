# API ↔ Test Relations

当前版本 **0.1.13** 增加直接、间接调用分类，详见 [0.1.13 发布说明](docs/release-0.1.13.md)。测试函数正文实际发起的 API 调用计为 `direct`，包括动态参数与输入构造；辅助函数、融合 API 内部等调用保留为 `indirect`，身份不足为 `unknown`。旧关系表及历史数据不改写，分类证据单独保存。[0.1.12](docs/release-0.1.12.md) 的僵尸进程修复和此前的编译池归属、checkpoint、精确绑定保护继续保留，见 [数据契约](docs/data-contract.md)。

独立 pytest 插件：在已有测试执行期间记录 **台账 API ↔ 测试实例** 的关系。
目标部署环境是 **ARM64 Linux + 昇腾 NPU**；采集核心与设备无关，x86 Linux 也可以使用。
插件不设置 CUDA/NPU 设备、不启动后台调度、不安装或替换 PyTorch/torch_npu、不采集测试通过失败结果。
独立的 `--api-relations-cuda-ipc-vmm` 选项会改变 CUDA 分配器和该进程的描述符访问许可；仅在验证需要的测试进程上显式启用，不能用于验证默认分配器或传统 CUDA IPC 本身。
不需要 `-g1`、GDB、CUDA 工具链。它观察的是 Python/C callable 身份，不是底层 NPU 算子调用栈。

## 文件树

```text
api-test-relations-plugin/
├── pyproject.toml                   # 安装信息、pytest 插件、CLI 入口
├── README.md                        # 安装、接入、查询
├── ORIGINS.json                     # 从旧采集器提取的文件及原始校验值
├── VALIDATION.json                  # 本次实际验证范围，不代替 ARM/NPU 验收
├── docs/
│   ├── data-contract.md             # 输出规范和关联语义
│   └── limitations.md               # 能力边界、已知缺陷
├── src/api_test_relations/
│   ├── pytest_plugin.py             # 显式启用、测试源码清单、结束时合并
│   ├── cli.py                       # doctor / query / status / audit / compact / settle
│   ├── retention.py                 # 脚本逐条核验、单库压缩、成功后自动删除本次原件
│   ├── settle.py                    # 进程退出后，从原始证据生成新汇总
│   ├── storage.py                   # SQLite 分片合并、子进程回执、API 查询
│   ├── recovery.py                  # 已退出进程的热日志恢复与原证据保留
│   ├── audit.py                     # 只读结构检查
│   ├── catalogs/pytorch-api-ledger.json # 冻结的 10,362 条台账
│   └── runtime/
│       ├── relation_plugin.py       # pytest setup/call/teardown 归属
│       ├── relation_collector.py    # callable 身份、别名、线程上下文
│       ├── relation_identity.c      # CPython 身份适配器，在目标 CPU 上编译
│       ├── relation_guard.h         # 原生 profiler 转发及 opcode 守护
│       ├── relation_context.py      # 当前测试归属上下文
│       ├── relation_process.py      # 子进程、进程池任务、回执
│       ├── relation_launch.py       # Python 启动命令解析
│       ├── relation_child_entry.py  # 子进程采集入口
│       └── bootstrap/sitecustomize.py
├── examples/test_ascend_smoke.py     # 在真实 NPU 服务器上运行的小样本
├── scripts/check_smoke_relations.py  # 检查小样本的 API 归属
└── tests/                           # 独立回归，无需真实 NPU
```

## 环境要求

- Linux，ARM64 或 x86_64；**本版仅支持 CPython 3.11**。
- pytest 7–9；C 编译器（`cc` 或 `CC` 指定）及当前 Python 的开发头文件。
- 服务器已经能用兼容的 PyTorch、torch_npu、CANN 运行目标测试。
- 在服务器上原生构建小型 C 适配器；不携带本机的 x86 `.so`。适配器没有 CPU 汇编或固定源码偏移，但依赖 CPython 3.11 的内部结构。
- Python 3.10/3.12 等版本需要额外适配，当前会明确拒绝运行，不会退化成不可靠的名字匹配。

在服务器**原有测试环境**中安装，以下安装不会依赖或替换 torch：

```bash
  python -m pip install ./api_test_relations-0.1.13-py3-none-any.whl
api-test-relations doctor
```

也可以进入源码目录执行 `python -m pip install .`。`doctor` 只编译并加载身份适配器，不导入 PyTorch，不证明 NPU 测试已经验证。

## 接入现有测试命令

保留原来可以运行 NPU 测试的环境变量、参数和筛选条件，追加下面几个插件参数即可：

```bash
cd /path/to/pytorch
python -m pytest test/YOUR_EXISTING_NPU_TEST.py \
  --api-relations-out /data/api-relations/run-001 \
  --api-relations-repo "$PWD" \
  --api-relations-import torch_npu \
  --api-relations-label ascend-npu
```

`YOUR_EXISTING_NPU_TEST.py` 替换为服务器上已能运行的测试。`--api-relations-import torch_npu` 只是先导入扩展；`--api-relations-label` 只是元数据。**插件不会把原本的 CPU/CUDA 测试自动改成 NPU 测试**。分布式和不适用测试仍由原命令筛掉；插件不会自行增删测试。

安装后 pytest 自动发现插件，但只有提供 `--api-relations-out` 才启用采集。如果环境设置了 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，另加 `-p api_test_relations.pytest_plugin`。

每次 pytest 使用新的、空的输出目录。核心插件不自动重跑或调度；需要并行续跑时显式使用仓库的调度脚本。多个独立 pytest 进程分别使用不同输出目录，当前不支持 pytest-xdist。内部启动的 Python 子进程和进程池任务由采集桥接跟踪。

默认使用随包台账；可用 `--api-relations-catalog /path/to/catalog.json` 替换。默认排除 `torch.compile`。只解析台账中的 `torch.*` 条目；额外 `torch_npu.*` 台账需要新的命名空间适配，当前会记录 `outside_namespace`。

## 查询

pytest 的 `sessionfinish` 发生在解释器退出之前，长期存活的子进程可能随后才写入分片或回执。等待 pytest 和相关子进程退出后，执行下列命令。**核验由脚本执行，不需要 AI 审批；通过后自动删除 `run-001` 整个插件输出目录，只留数据库。**

```bash
api-test-relations compact --out /data/api-relations/run-001 \
  --database /data/api-relations/relations.sqlite
```

脚本从全部原始分片和子进程回执重新合并，逐条核对测试身份、源码路径、阶段、API、别名与首次调用点；冲突会阻止删除。再对合并库与压缩库所有数据表做双向差集比较，并检查 SQLite 完整性和外键，提交并重新打开验证后才清理。重复字符串字典化，关系内容没有抽样或删减；gap、阶段状态、源码元数据及校验记录保存在同一个库内。

生产者仍在运行、文件变化、数据库错误、未知顶层文件或软链接都会阻止清理。目标必须是不存在的新文件且位于运行目录外；本命令暂不负责多个运行的追加合并或调度续跑。失败时保留原件；若验证通过但删除发生文件系统错误，库仍有效，返回 `cleanup=incomplete`。机器在发布后突然断电可能留下已验证库和原件两份，应先检查库内 `retention_receipt`。不删除运行目录以外的编译缓存、游戏缓存或个人文件。

这证明合并与压缩保持已有记录，不证明采集阶段没有遗漏。需要长期保留原始证据时仍可显式使用旧 `settle --out RUN --destination NEW`；它复制证据，不节约空间。`audit` 仍面向清理前的运行目录。

```bash
api-test-relations status --out /data/api-relations/relations.sqlite
api-test-relations query --out /data/api-relations/relations.sqlite --api torch.Tensor.fill_ --json fill-relations.json
```

返回完整参数化 nodeid、`pytorch/` 开头的测试文件、阶段、API ID、同一对象的别名、首个调用点和采集缺口。未观测到关系不等于没有测试调用该 API；`capture_finished` 表示采集会话收尾，不是零遗漏声明。

结束时等待已记录子进程最多 120 秒，可用 `--api-relations-child-wait` 调整。超时不杀服务器进程，保留缺口和原始分片。异常中断保留已写分片；不要将它当成完整采集，或对原目录直接重新运行。

## 单库增量续跑

已有规范化 `api_test_relations_single_database/1` 基线时，可用 `api-test-relations append --out RUN --database BASE.sqlite --module MODULE` 追加核验后的未完成实例，并自动清理本次原始目录。完成实例禁止覆盖；中断实例若重采没有包含旧的全部正向关系，事务回滚、保留原件。该格式与上面 `compact` 的单次运行导出格式不同，不会自动互转。

仓库 `scripts/run_instance_shards.py --out PREPARED_RUN --max-workers N` 提供单写入者、按完整实例分片的 CUDA 调度接入。并发上限由调用参数指定，没有固定四 worker 上限；资源门槛和 30 秒错峰策略可以使实际并发更低。它读取显式准备的 `manifest.json`、`queue.json` 和 `instance-plan.json`，校验分片互不重叠且覆盖声明的剩余实例，每次启动再排除正式库中已完成的 nodeid。合并仍使用原来的严格核验，合并前另存带 SHA 的分片归档，日志和私有缓存保留。详见[实例分片调度](docs/instance-shard-scheduling.md)。原模块级入口 `scripts/run_single_database.py` 继续保留，不能同时监督同一个正式库。

冻结清单包括插件及调度器源文件哈希、源码提交与测试文件哈希、解释器/台账/数据库路径、测试模块列表、资源与时间限制。脚本不提供任意目录自动恢复或后端自动迁移。NPU 接入继续使用现有 pytest 命令，不能直接套用此 CUDA 资源策略。调度脚本位于源码仓库，不会随 pytest 自动发现启动；核心插件仍可独立使用，当前不支持 pytest-xdist。

WSL 运行时，应将活动数据库、分片、临时/编译缓存、调度状态和日志全部放在 WSL 的 Linux 文件系统（例如 `/home/USER/api-relations/`），不要放到 `/mnt/c` 或 `/mnt/d`。按测试查询的大库应先创建 `CREATE INDEX IF NOT EXISTS observations_by_test ON observations(test_key)`；它只增加索引，不修改关系。Windows 端仅保留源码及静态入口。

调度器为每个 attempt 创建 `/tmp/atr-<uid>/<token>` 短路径别名，指向该 attempt 私有的 `worker-runtime/.../tmp`。这使 `multiprocessing.resource_sharer` 创建的 Unix socket 地址保持足够短，避免长运行目录造成 `AF_UNIX path too long`，随后 DataLoader 一直等待队列结果。进程退出且入库核验成功后，仅删除核对归属的别名和私有目录。特殊上游 runner 缺少直接 Python 源码时，记录未执行边界；监督器接管已有 worker 时保留原 PID、分片和启动时间。详见[调度可靠性与验证范围](docs/scheduler-reliability.md)。

## ARM + 昇腾实机验收

在插件源码目录、已配置好的 NPU 环境执行：

```bash
api-test-relations doctor
python -m pytest examples/test_ascend_smoke.py \
  --api-relations-repo "$PWD" \
  --api-relations-out /data/api-relations/npu-smoke-001 \
  --api-relations-import torch_npu \
  --api-relations-label ascend-npu
python scripts/check_smoke_relations.py --out /data/api-relations/npu-smoke-001
```

这里的根目录是插件源码目录，因此示例路径规范化为 `pytorch/examples/...`，真实测试接入时应指定真正的 PyTorch 根目录。

本次在隔离的 x86 Linux / CPython 3.11 环境验证采集逻辑，真实 ARM/NPU 验收尚未执行。NPU 专属扩展、替换后的绑定和目标测试的采集缺口，需要以上实机检查确认。

0.1.1 修复注册表并发刷新、cProfile 抢占、SQLite 热日志合并三类问题；0.1.2 增加进程退出后的证据重汇总。39 项隔离回归及另外 2 项真实 PyTorch/CUDA 集成检查通过；还进行了真实上游测试对照。修复范围与剩余边界见 `RELEASE_NOTES.md` 和 `docs/limitations.md`。已有基线不会自动恢复或重采。

普通回归无需安装 trace-api-chain。6 项 `paired_plugins` 联合回归按用户约定默认跳过，3 项真实 PyTorch/CUDA 集成测试仍为显式 opt-in。0.1.11 的独立快照为 **179 通过、9 跳过**，详见[发布验证收据](docs/validation/0.1.11.json)。跳过不计为通过，真实 CUDA 调度对照也不能替代 ARM/NPU 验收。
