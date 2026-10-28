# 0.1.10：独立观察通道、原生 profiler 切换与 fork 导入修复

基于远端 0.1.9 提交 `ab867db9de454a03de9d3b3951e6e35bb0a613e6`；完整保留它的显式 CUDA IPC VMM 兼容功能，NPU 验证默认不启用该选项。

## 修复

- 使用精确原生 callable 的边界保护，覆盖 cProfile enable/disable、已加载 Torch 的 profiler 控制函数和提前保存的 CPython setprofile/settrace。保持 callable 对象身份，转发原始参数、结果和异常；在真实 C 返回前恢复独立观察器。
- 为 trace-api-chain 0.3.2 提供独立 profile/trace 订阅。用户 profiler 的逻辑客户端、局部 trace 和 opcode 标志继续转发。未知原生入口绕过观察通道时仍暴露缺口，不补造事件。
- 通过线程局部 native 观察器静默区，避免观察器自身的 Torch 控制调用污染调用链窗口。该能力由已加载的 trace hook 提供，单独使用关系插件不要求加载 hook。
- fork 可以继承父进程尚未完成的 torch 导入。子进程使用已有绑定开始观察，不等待不会在该子进程继续的父导入；绑定版本检查继续工作。正常 spawn 仍等待正常导入完成。
- 双插件共同运行时，关系运行时源码和本机适配器可归档到调用链证据目录。关系中间分片仍按原核验流程删除；调用链依赖的源码、二进制和哈希另行保留。

审计入口先筛选影响采集有效性的生命周期事件；原先所有相关事件的校验均保留，普通属性读取不再重复查询进程号。

## 验证与边界

108 项关系插件回归通过、3 项显式集成跳过；包含导入中途 fork 的真实进程回归、原生 profiler、客户端切换、拒绝 setter 时的异常语义，以及独立观察通道的调用返回平衡。另由主任务在 ARM64 NPU 上验证原始异常用例，最终结果以该任务的完整证据为准。

仍限 CPython 3.11。独立观察通道不会让 Python 关系采集变成 C++ 调用栈采集，也不扩大 trace-api-chain 原本声明的线程/进程范围。

Codex watermark: Assisted by OpenAI Codex.

## 独立回归入口

普通回归不要求安装 trace-api-chain。6 项需要两插件的测试标记为 `paired_plugins`，默认跳过；用户明确要求本轮不运行它们。真实 PyTorch/CUDA 的 3 项集成测试仍按已有设置默认跳过。独立回归预期为 102 通过、9 跳过。历史 108 通过含这 6 项联合验证，不作为普通独立回归的数量。

test-api 的安装依赖仍只有 pytest，没有给它增加 trace-api-chain 依赖。服务器正式双插件采集任务的加载配置由任务本身决定，不受测试标记影响。
