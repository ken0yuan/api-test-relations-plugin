# 0.1.11：编译池归属与实例分片调度

发布日期：2026-10-30。

## 采集核心

修复受支持的 PyTorch Inductor `SubprocPool` 自定义 IPC 丢失提交测试上下文的问题。
原实现可能把 B 提交的任务记给创建编译池的 A；任务正常结束、甚至 gap 为零，也不能
证明端到端归属正确。现在通过版本化信封传递完整 nodeid／阶段，在 sidecar、工作进程、
载荷和结果反序列化及 Future 回调恢复提交者归属。原载荷、结果和异常语义保持，按精确
上游源码与方法来源做适配守卫；不支持的实现显式记录缺口。见[编译池归属](subproc-pool-ownership.md)。

原生 profiler、minifier 行缓存和绑定检查优化继续保留。profile／trace、完整台账、
测试参数和内部样本不减少；API callable 身份、别名、调用点规则及存储／append 契约
保持。历史关系和来源不会被自动重新分配或修复，仍限 CPython 3.11。

## 调度层

- 新增 `run_instance_shards.py --out PREPARED_RUN --max-workers N`；上限是必填正整数，
  不固定为四。每个独立 pytest 只执行自己分片中尚未完成的 full-nodeid。
- 分片必须互不重叠并恰好覆盖声明的剩余身份。保持正式模块身份不变，不创建虚拟模块
  混淆同一测试。所有原始 API、phase 和子进程／任务 owner 在合并前检查归属边界。
- 保留单写入者、原 append 六项核验、已完成实例保护、旧正向关系子集保护和完整性／外键
  检查。合并前归档该 attempt 的原始证据，原日志和私有缓存保留。
- 保持 30 秒派发间隔、RAM／GPU／磁盘门槛和模块／子进程时间界限，按观测 PSS 预留启动
  内存；低内存时 checkpoint 收束最新 worker、降低实际并行容量。
- CUDA 计划在收集前拒绝 XPU 文件；每模块依赖路径只作用于指定 worker，不自动安装依赖。
  GPU 查询保留原五秒限制，区分失败状态，失败时不使用旧空闲值派发。

详见[实例分片调度](instance-shard-scheduling.md)、[CUDA 范围](cuda-backend-scope.md)
与[GPU 查询诊断](gpu-query-diagnostics.md)。脚本从源码仓库执行，与可独立加载的 pytest
采集插件分开；本版仍明确不支持 pytest-xdist。

## 已验证范围与限制

真实 CUDA 隔离对照使用同一份采集实现与完整台账，串行和两个实际并行 worker 得到
完全相同的 API／别名／调用点／阶段／归属记录及输出，旧正向关系、已完成身份和 gap
保持。实际部署通过 checkpoint 收束旧 worker、原 append 核验及完成身份指纹比较后，
将剩余 888 个实例分为四个互斥分片；该次启动参数为 `--max-workers 4`。

这不是全量零遗漏声明。关系按 callable 和实际别名组去重，保存首次调用点；关系数量
不是调用次数，多个别名不表示分别执行了多个函数。历史缺证、错归暴露范围和补采证据
继续分别保留。新版本发布不会改写正在运行的冻结包或历史来源版本。

6 项双插件联合回放按用户约定默认跳过，3 项显式硬件集成默认跳过；不把跳过写成通过。
ARM／昇腾实机验收不由本次 CUDA 验证替代。

Assisted-by: OpenAI Codex
