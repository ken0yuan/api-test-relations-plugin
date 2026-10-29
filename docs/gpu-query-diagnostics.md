# CUDA 调度显存查询诊断

资源发布增加 `gpu_query`：包含查询命令、采样时间、耗时、原五秒时限，以及
`ok`、`timeout`、`command_failed`、`invalid_output`、`os_error` 或
`subprocess_error` 状态。命令失败保留退出码和有长度上限的原始输出。

`gpu_free_mib=0` 是有效的零空闲值；查询失败仍为 `null`，禁止新派发。
不使用过期读数、不推测空闲值，不重置 GPU 或 WSL，也不延长查询时限。
已有 worker 与 API 身份、别名、调用位置、归属、落盘和数据库契约均不改变。

旧版本曾把多个失败原因统一写成 `null`，无法事后仅凭该值确认是超时。
2026-10-28 的现机查询已恢复，连续十次 WSL 查询成功，耗时约 0.04–2.72 秒；
当时显存空闲仅 69–793 MiB。Windows 的一条 GPU 错误事件以及 WSL GPU
等待中断日志不足以单独确证驱动重置或上次查询失败的唯一原因。

运行冻结目录保持不变。部署此调度修复应使用新的冻结目录；只有监督器
不在生产合并中且所有原 worker 身份、原始目录与实际进程相符时，才可
交接监督权。原 worker 不停止、不重启，保留原 PID、开始时间和排除清单。

Codex watermark: Assisted by OpenAI Codex.
