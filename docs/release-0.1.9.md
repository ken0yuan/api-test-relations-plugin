# 0.1.9：显式 CUDA IPC VMM 兼容

## 原因和适用范围

2026-10-28 的 WSL 实测中，CPU Tensor 跨进程正常；关闭采集器后 CUDA Tensor 仍在 `cudaIpcOpenMemHandle` 报 `invalid resource handle`。独立 CUDA Runtime 实验确认内存句柄导出成功、导入返回 400，事件句柄导入及等待正常。原测试的子进程在 `queue.get()` 反序列化时退出，父进程在无超时的 `ready.wait()` 等待，这不是正常的测试计算耗时。

这些证据定位到该环境的传统 CUDA IPC 内存导入路径，不证明所有 WSL 都不支持 IPC。NVIDIA 当前文档声明 R510 起支持传统 IPC，参见 [CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/)。插件不会安装 Linux 显卡驱动、替换源码 PyTorch 或修改系统内核权限策略。

## 显式使用

```bash
python -m pytest test/test_multiprocessing.py \
  -p api_test_relations.pytest_plugin \
  --api-relations-out /home/USER/capture-new \
  --api-relations-repo /home/USER/pytorch \
  --api-relations-cuda-ipc-vmm
```

需要当前源码 PyTorch 支持 expandable-segment CUDA IPC、Linux `pidfd_getfd` 和对应 CUDA VMM。先以小样本验证，不能从 GPU 可用推断 IPC 可用。可选单库调度器清单通过 `cuda_ipc_vmm_modules` 精确指定模块，其他 worker 保持原配置。

启用后：

- 在 CUDA 初始化前设置 `PYTORCH_ALLOC_CONF=expandable_segments:True`；已有冲突配置会报错，不能静默覆盖。测试本身仍可以显式修改 allocator，因此此选项不能保证所有 IPC 用例可用。
- 每个参与进程通过 `PR_SET_PTRACER` 允许其真实父进程及该父进程的后代读取自身描述符，供 CUDA VMM 的 `pidfd_getfd` 导入使用。主 worker 的范围是监督进程的子树，子进程的范围是其测试进程家族。许可持续到该进程结束；不使用 `PR_SET_PTRACER_ANY`，不修改全局 Yama/内核配置。
- 主进程配置保存在 `environment.json.runtime_compatibility`；子进程策略保存在 `shards/cuda-ipc-compat/`，传递的显式配置也进入 launches/receipts。正常单库入库继续保留配置、来源哈希和进程证据。
- API 身份、别名、位置、阶段和归属匹配逻辑不变；实际 Tensor 仍在 GPU 上共享，保留梯度、视图偏移和共享修改，不以 CPU 拷贝冒充 CUDA 共享。

默认关闭，ARM/NPU 等环境无需启用。该选项不代表传统 CUDA IPC 被修复，也不适合比较默认 allocator 的行为或性能。

## 验证与恢复约束

合成回归覆盖默认不改变配置、只授权真实父进程、权限失败、已初始化 CUDA、冲突分配器；真实合成检查覆盖父子双向、兄弟进程、梯度、视图、共享修改。旧版参考采集与新模式在同一个共享样例的 15 条 API/别名/调用位置/所有者证据完全一致。该范围不证明全量零错漏。

恢复前先请求提交屏障，再结束已确认卡死的 attempt，按原始分片精确核验合并；不能把卡住的 entered 阶段登记为完成。健康 worker 可按 PID/start_ticks 和原始路径接管。修复后成功采集的未完成实例直接核验入库，后续按完整 nodeid 排除；已经完成的实例（含零观测、有 gap）不能重跑。

Codex watermark: API-test-relations-plugin 0.1.9 explicit CUDA IPC compatibility.
