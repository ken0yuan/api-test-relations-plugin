# 输出规范

0.1.13 在原关系之外增加调用分类证据。原始 `call_kinds` 的键为
`(nodeid, phase, api_id, aliases, kind)`，每种 kind 分别保留首次 site；汇总的
`call_kinds` 与 `call_kind_sources` 另外保留 job、所有分片首次位置及 shard。
kind 值域为 direct/indirect/unknown，含义见 [分类口径](release-0.1.13.md)。
旧分片缺少分类时不猜测回填；分类来源必须对应同一原始分片中的观测身份。
单库 `/2` 的可选附加表 `observation_call_kinds` 保存 API/test/phase/alias 的
外键、kind、site、source_capture、source_shard。分类不能覆盖旧正向关系；
同一逻辑关系允许直接和间接证据共存。查询 `--call-kind direct` 才是直接关系清单。

每次 pytest 独立建立一个目录：

```text
run-001/
├── manifest.json           # 仓库、平台、解释器、台账/采集器哈希、排除项
├── environment.json        # torch / 显式导入扩展的版本及加载路径
├── test-sources.json        # 测试源文件路径、SHA256、完整参数化实例名
├── api-catalog.json         # 此次使用的冻结台账
├── relations.sqlite        # 汇总关系，不保存测试断言/结果
├── progress.json           # 采集状态、计数
├── capture-error.json      # 仅发生采集初始化/合并异常时存在
├── runtime/                # 此次冻结代码和当前 CPU 编译的身份适配器
└── shards/
    ├── <pid>[-token].sqlite # 进程独立原始证据
    ├── launches/           # 启动命令和父测试归属
    ├── receipts/           # 子进程启动/结束回执
    ├── tasks/              # 进程池任务提交归属
    ├── task-receipts/      # worker 执行回执
    └── sqlite-recovery/   # 0.1.1：恢复前原件、副本与恢复核对记录
```

逻辑关系键为 `(job, nodeid, phase, api_id)`，此独立插件一次运行的 job 为 `pytest`。从 0.1.8 起，一条逻辑关系允许多条证据，汇总表的唯一键包含 `aliases, site`。`phase` 是 setup/call/teardown；分别保存，查询可判断关系发生在正文还是 fixture。
`aliases` 表示当时解析为同一 callable 的台账名称，并不证明用户采用了其中每一个拼写。原始分片每个进程、测试阶段、API 和别名组保留首个调用位置；汇总保留来自不同分片的所有这些位置，不是每次调用的完整轨迹。
`relation_sources` 保存每条证据对应的原始分片文件名（含 PID/子进程 token），与进程回执配合核对来源。完全相同的证据可复用一行，但不同来源仍分别保留。查询的 `source_shards` 返回这些文件名；旧库没有来源记录时返回空列表，不猜测来源。`relations` 行数从此是证据数；逻辑关系数应按上述四列去重计算。

父进程观察 `[manual_seed]`、子进程观察 `[manual_seed, random.manual_seed]` 时，两组原样保留，不相互覆盖，也不取并集制造新观测。普通汇总、压缩后的 `relation_keys` 与增量库使用一致的多证据语义。

增量写入会在原有写锁和事务内将 `api_test_relations_single_database/1` 升级为 `/2`：`observations` 唯一键增加 `alias_group_key`，并增加 `observation_evidence` 保存位置、采集目录和分片来源。旧观测和完成状态保留；升级或追加失败整体回滚。已丢失且原始分片不存在的旧证据不能恢复。新版本可从仍保留的 0.1.7 原始分片重新汇总；历史库升级本身不表示历史采集完整。

`tests` 保存完整 nodeid 和文件；`phases` 保存 entered/closed；`registry` 保存精确解析、未解析、排除、原生符号等状态；`gaps` 保存采集问题。
`processes`、`process_details`、`process_tasks` 保留因果归属。启动记录本身不代表采集成功。
`retained` 表沿用原存储格式但本版始终为空；本版不导入旧基线。

查询结果中的 `unobserved_means_no_test` 永远为 false。只读 audit 检查孤立关系、台账和路径，不证明全量零漏采，也不替代原始分片逐项核对。

代码、台账和源码位置均按实际运行生成/解析；没有按某个 API 的固定源文件行号建立关系。不同 PyTorch/torch_npu 版本会产生不同解析状态，必须各自保存运行证据，不混用结果。

0.1.2 的 `settle` 输出是独立派生目录，增加 `settlement.json`（原目录、原 manifest 校验值、原进度、复制文件校验值、汇总实现版本）。manifest 的 `package_version` 仍表示原采集器版本；`finalizer_version` 单独表示重汇总代码版本。原始分片不被修改，原摘要也保留。`capture_pid` 用于拒绝对仍在执行的采集进程收尾。
