# CUDA 采集范围

当 `device_filter=cuda` 时，调度器排除路径中包含 `xpu/`，或文件名为
`test_xpu.py`、`test_xpu_*.py` 的模块。排除发生在 pytest 收集之前，
`launch_spec` 也拒绝为这些文件构造启动命令。

`backend-exclusions.json` 保留原队列状态并记录 `excluded_backend_xpu`。
既往关系、测试身份、来源和采集边界不删除，也不把已采历史改写成未采集。
接管清单若仍含活动 XPU worker，监督器会拒绝接管；应先按既有屏障流程
保存、终止并核验合并对应 attempt。其他后端和 CUDA 文件不受影响。

Codex watermark: Assisted by OpenAI Codex.
