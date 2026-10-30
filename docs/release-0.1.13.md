# 0.1.13：区分测试函数的直接与间接 API 调用

本版基于 0.1.12，保留僵尸进程处理、精确 callable 识别、子进程与编译池归属、checkpoint、别名和证据来源规则。
安装自检显式加载本次编译的临时身份适配器，兼容 `PYTHONSAFEPATH=1`，不改变宿主安全路径设置。

## 分类口径

- `direct`：实际发起 API 调用的函数是当前 pytest 用例的测试函数正文。`torch.add(x,y)`、运行时参数 `op(x,y)` 和正文内用于输入构造的 API 都计入。
- `indirect`：当前测试拥有该调用，但调用语句位于辅助函数、融合 API 内部、fixture、线程或任务执行辅助代码等位置。不是按 API 名或测试意图猜测。
- `unknown`：没有足够测试函数身份，或跨进程不能确认同源函数。保留观测，不猜为直接或间接。

测试调用 `helper(torch.add,x,y)` 时，add 在 helper 中执行，属于间接调用。测试调用融合 API 时，融合 API 入口为直接，其内部 add 为间接。同一用例的同一个 API 可以同时具有 direct 和 indirect。

pytest 装饰器通过 `__wrapped__` 识别实际测试正文，包装函数自身的调用为间接。父进程以实际函数对象核对；子进程以源文件身份、定义位置和限定名核对本地函数对象，不比较跨进程指针，也不要求 pytest 改写前后的字节码相等。身份通过已有线程、subprocess、multiprocessing、标准池和受支持的 Inductor 编译池上下文传递，按任务恢复并在结束时还原。

## 存储与查询

分类在去重之前记录。原始分片新增 `call_kinds`，按测试、阶段、API、当时别名组和 kind 分别保留首次位置。旧 `relations` 保持原语义，不能据旧的单个首次位置推断完整分类。

汇总新增 `call_kinds` 和 `call_kind_sources`。严格核验分类对应的原始观测、kind、位置和分片来源；声明分类但缺少记录时拒绝清理原始证据。compact 保留新表并双向核验；单库 `/2` 增加 `observation_call_kinds`，旧观察和完成身份不变，中断重试不能丢掉已有正向分类。

新建直接关系清单应使用：

```bash
python -m api_test_relations query --out RUN_OR_DATABASE --api torch.add --call-kind direct
```

`--call-kind indirect` 查询间接关系，`unknown` 查询未分类记录。默认 `all` 保留原查询的 `tests` 行为，并添加 `classified_tests`；指定分类时 `tests` 返回该类证据。普通运行目录、compact 数据库和单库数据库都支持分类查询。进度中 `call_kind_relations` 分别统计每类的不同测试/API 对，三类不是互斥分组，不应相加当作总关系数。

旧数据、旧分片和旧冻结 runtime 不自动补写分类；缺少分类证据时显示 unknown。分类是调用位置关系，不是断言意图、调用次数、算子精度或底层内核覆盖判定。

## 验证

发布附件与 `docs/validation/0.1.13.json` 保存实际回归、NPU 小样本、包字节一致性及校验结果。原先 9 项需单独开启的联合/CUDA 测试仍按既有规则跳过，不计为通过。未启动全量上游用例重跑。

Assisted-by: OpenAI Codex
