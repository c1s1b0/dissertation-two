# 组件候选保留实验：代码、参数与归档记录

本仓库支持历史模型输出的离线重分析，不包含完整在线大模型服务、多代理调度或自然语言澄清系统。无需 API 密钥，不运行新模型。

## 环境和运行

Python 3.12.7、NumPy 1.26.4、Pydantic 2.12.4；依赖见 `requirements-offline.txt`。

```powershell
python -m pip install -r requirements-offline.txt
python scripts/verify_cmc_bundle_20261014.py --full
```

必须在干净解包副本执行。入口先核对 SHA-256，计算期间禁用 Python 网络连接，然后运行 26 项发布测试、400 个冻结任务与三重复、Support、480 个结构实例/4320 次求解、精确解集召回、十随机种子敏感性以及软件契约审计。所有已发布分析结果逐文件比较，排除本机计时，不排除历史延迟。输出会被重算，再次验证应重新解包。

快速模式不重算主分析、结构与随机敏感性，但仍执行测试、Support 和完整软件契约审计。独立执行入口如下：

```powershell
python tests/test_reconstruction_20261013.py
python tests/test_cmc_offline_20261014.py
python tests/test_software_contracts_20261016.py
python tests/test_artifact_paths_20261016.py
python scripts/reconstruct_paper2_20261013.py
python scripts/structural_stress_20261013.py
python scripts/reconstruction_diagnostics_20261013.py
python scripts/cmc_offline_20261014.py
python scripts/audit_software_contracts_20261016.py
```

## 返修实验与离线重放

| 实验 | 复现说明 |
| --- | --- |
| 历史配额、独立固定配额和异质接口 | [离线补充实验](research_artifacts/cmc-revision-20261106/返修实验说明.md) |
| 24 任务本地 Qwen3-8B 前瞻实验与语法约束诊断 | [前瞻实验复现](research_artifacts/cmc-prospective-20261106/复现说明.md) |
| 24 任务固定首提案及混合接口单位案例 | [固定首提案复现](research_artifacts/cmc-fixed-first-20261109/复现说明.md) |
| 400 历史任务 BM25/E5 对照、48 任务扩展和 KTR 目录案例 | [检索与扩展实验复现](research_artifacts/cmc-retrieval-extension-20261109/复现说明.md) |

基础核验入口检查全包哈希并重算基础实验。返修实验请按各自的说明重放归档响应与检索分数。归档重放无需模型权重或 API 密钥。

## 数据与布局

`src/` 保存离线生成器、求解器和候选池分析；`scripts/` 保存统计、审计和复现入口；`tests/` 保存发布测试。`experiment_parameters.json` 集中记录参数、种子和历史服务路由。

标准数据目录为 `research_artifacts/`，包含 `benchmarks`、`formal_runs_v2`、`formal_llm_runs`、`reconstruction-20261013`、`cmc-offline-20261014`、`cmc-audit-input-20261014`、`reviewer-supplement-20260820`、`software-correction-20261016`。从早期扁平版更新时，请移除仓库根目录的旧数据目录，然后解包本版。根目录与 `research_artifacts/` 下各有一套 `benchmarks` 时，入口会拒绝运行。`文件校验清单.json` 记录发布字节，不对重算后的计时字节作一致性承诺。

## 修正与历史数据

本版独立修复全目录重放的类型开关、一次性历史迭代器、真值枚举的全局无解/端口/场景/角色检查及目录定位。独立向量解集枚举也显式检查端口。回归包含不兼容的类型、单位、方向和缺失端口，作用仅为实现正确性验证，不是新模型或工业效果实验。

冻结任务和历史模型输出不修改。修正标签另存于 `software-correction-20261016/corrected_truth_labels.json`；12 个冻结负控制原有正标签被标明为历史错误。生成器现在输出修正标签，因而不应期待它与旧冻结任务在 `expected_feasible` 字段上逐字相同；其余字段应相同。审计逐任务比较独立真值、求解器与完整解集，并重放全部 6000 条历史方法记录。最终数字以 `software_verification.json` 和运行后生成的 `独立复现核验.json` 为准。

## 统计与结论边界

每任务三次重复采用二元多数表决，配对 bootstrap 为 10000 次，种子 20261013。历史比较为独立 16 项 Holm 家族；含 Support 的当前匹配比较为另一 16 项家族。旧 `matched_statistics.json` 的 12 项匹配统计为归档版本。

P 四组成功率为 1.00/0.90/0.92/0.90；Rule 为 1.00/0.90/0.91/0.85；Support 为 1.00/0.90/0.94/0.90。未证明 P 显著优于 Rule 或 Support，也未证明等效。只匹配逐角色候选数，不匹配时间、token 或费用。结构实验不调用模型。20 个全局无解标记是软件负控制，不是物理不可装配证明；原目录端口同质，不能据此推断类型检查的实际收益。

历史路由 `gpt-5.6-terra` 的底层身份未独立核实。审计含 1840 条请求、39 次失败；30/1200 个任务重复出现重试；条件验证接受数 1801/1809。token 仅 8 条正数、1801 条零值、31 条缺失；170229 是部分和，不是总量或费用。私密端点、密钥和原始响应不发布。

主求解器返回有限组合中的首个可行见证，不保证成本最优、最小改动或真实物理装配。结构求解器单独具有搜索预算和 unknown 状态。来源及许可见 `THIRD_PARTY_NOTICES.md`；未新增统一开源许可证。
