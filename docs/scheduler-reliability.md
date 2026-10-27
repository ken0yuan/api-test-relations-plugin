# 0.1.8 调度可靠性补丁

此补丁修改仓库的可选单库调度脚本。采集核心仍为 0.1.8，API 身份匹配、台账、阶段和线程/子进程归属规则保持原有实现。

## 短临时路径解决 DataLoader 队列等待

长运行名称与 attempt ID 拼接出的 `TMPDIR` 可能超过 Unix socket 地址限制。Python 的 `multiprocessing.resource_sharer` 在其下创建 `pymp-*/listener-*`；Tensor 传输的队列发送线程会抛出 `OSError: AF_UNIX path too long`。父进程仍等待返回的数据，子进程仍可存活。pytest 捕获输出时，错误可能直到测试结束才呈现，因此存活进程和持续提交的采集 checkpoint 无法证明测试还在推进。

`single_scheduler_helpers.launch_spec` 为每个输出目录计算独立 token，在当前用户私有的 `/tmp/atr-<uid>/` 中创建短别名，指向原 attempt 的私有临时目录，并将别名作为 `TMPDIR`。所有实际运行数据继续保存在 Linux 文件系统。

别名父目录必须属于当前用户且权限为 `0700`。已有文件、错误归属或指向其他位置的别名会阻止使用。worker 元数据记录 `temp_alias`。入库核验成功且相关进程已退出后，清理核对输出目录 token 和目标路径，只取消该别名，再删除私有 scratch；不递归清理 `/tmp`。

修复不修改测试正文、worker 数量、数据量、Tensor 共享策略或 API 采集过滤规则。冻结运行必须使用新的调度快照，不能热改运行中的脚本。

## 监督器恢复与特殊 runner

- 准备清单可提供 `adopt_workers`。替代监督器使用原 PID/start_ticks、原启动时间和原始输出目录观察已启动的 worker，避免重新启动其测试；旧目录的私有数据在验证后由正确的 `runtime_root` 清理。
- 进程是否结束与采集是否完整分别由存活检查、阶段和会话证据判断。接管句柄不推断 pytest 断言结果。
- `upstream_special_runner` 等缺少直接 Python 源码的条目记录为 `runner_unavailable`，保存模块归属的 collection gap 和 `executed=false` 收据。未知或变化的普通 Python 源码也保留明确边界。它们不会因 `Path(None)` 异常导致整个调度器退出，也不会被登记为已完成。
- 所有恢复继续在 collection 排除完成的完整 nodeid，并在入库时拒绝重复执行已完成实例。失败保留原始证据。

## 2026-10-28 验证

Linux / CPython 3.11，保持原有源码 PyTorch 环境：

- 7 项调度回归通过：来源检查、特殊 runner 边界、无重启接管、短 Unix listener、真实 FD 共享往返，以及错误别名清理拒绝。
- 两个先前未完成的原始实例做有界控制变量对照。长 `TMPDIR` 下关闭采集仍等待到 65 秒上限；仅缩短路径后，无插件分别约 6.7 秒、2.8 秒结束；启用原 0.1.8 插件分别约 25.1 秒、7.8 秒完成采集。
  - `test/test_dataloader.py::TestDataLoader::test_batch_sampler`
  - `test/test_datapipe.py::TestSharding::test_legacy_custom_sharding_with_old_dataloader`
- 两个成功采集分别保留 74 / 54 条完整关系证据、270 / 88 条分片来源。脚本核对原始 API/别名/位置/归属/来源，确认旧正向关系为新记录子集，8 / 2 个子进程完成回执及管理服务边界正确；本次两个精确实例未报告采集 gap。
- 成功结果直接追加到单库，后续续跑排除这两个已修复的实例。原有 102,317 个完成实例身份及来源，以及相关模块原已完成实例的关系指纹保持一致。

这些验证确认当前停滞的路径配置原因和所列修复范围，不证明所有 DataLoader 场景或全量 API 关系零遗漏。原始记录已经缺失的历史证据不会因这个补丁自动恢复。

Codex watermark: API-test-relations-plugin 0.1.8 scheduler reliability patch.
