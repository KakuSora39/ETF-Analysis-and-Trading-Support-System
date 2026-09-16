# ETF 分析与交易辅助系统

> 本项目基于 [zhuleimed/etf-daily-sync-and-backtest](https://github.com/zhuleimed/etf-daily-sync-and-backtest) 二次开发。原项目采用 MIT License；本仓库的 [LICENSE](LICENSE) 保留 `Copyright (c) 2026 zhuleimed and contributors`。本项目保留行情同步、策略回测和原模拟交易框架，并扩展全市场 ETF 筛选、多策略共识、市场状态、趋势/反弹/防守机会、风险检查、真实持仓辅助、综合模拟盘和 A/B 前向验证。

这是一个本地运行的 A 股场内 ETF 分析工具。它依据当日已知行情给出候选、风险和行动建议。**模拟盘使用假资金，不是实盘交易；项目目前不连接券商自动下单。**

## 三种用途

| 用途 | 数据与交易 | 状态来源 |
| --- | --- | --- |
| 回测 | 用过去的历史数据检验策略 | 原 `strategies/` 回测框架 |
| 综合模拟盘 | 从首次启动日开始，逐日用新行情和假资金前向交易 | `simulation/output/state_etf_universe_*.json` |
| 真实持仓 | 用户手动买卖，系统只分析、提醒并给出退出建议 | `data/etf_holdings.json` 等人工配置 |

市场分析、真实持仓和模拟账户各自独立。模拟盘不会把人工持仓文件当作自己的资产；旧的独立策略模拟盘与回测框架仍可使用。

## 系统流程与项目结构

```text
ETF / 指数日线同步 → 数据质量与价格断层检查 → 全市场基础筛选
→ 正式候选池 / 新 ETF 观察池 → 同指数与相似 ETF 去重
→ 多策略信号与历史验证 → 市场环境 → 趋势 / 反弹 / 防守机会
→ ETF 结构风险与当前交易风险 → 最终行动
   ├─ 分析报告和 Excel
   ├─ 人工真实持仓分析
   └─ Conservative / Progressive 综合模拟盘
```

| 目录 | 主要用途 |
| --- | --- |
| `etf_sync/` | ETF、指数日线同步 |
| `etf_universe/` | 全市场筛选、共识、验证、三赛道、风险与报告 |
| `strategies/` | 原独立策略及回测 |
| `simulation/framework/` | 原模拟盘状态、经纪报价、T+1 和风控基础设施 |
| `simulation/etf_universe/` | 双账户前向实验；复用原框架的原子状态写入、经纪报价及交易限制检查 |
| `tests/` | 回归与模拟盘测试 |
| `data/` | 本地 SQLite、配置、人工持仓及分析输出 |

## 安装与快速开始

在仓库根目录执行：

```powershell
python -m pip install -r etf_universe/requirements.txt
python main.py --sync-only
python -m etf_universe --config data/action-test-config.json --output data/universe
python -m simulation.etf_universe.daily --analysis-root data/universe
```

初次使用若没有 `data/action-test-config.json`，可直接运行 `python -m etf_universe` 使用代码默认筛选参数。分析命令生成带时间戳的目录；模拟盘读取其中当日机器可读行动快照，并向原 Excel 增加模拟盘 Sheet。只有完整分析才会产生可供模拟盘使用的快照。`--offline` 可只用本地资料缓存；历史资料快照缺失时不会伪装成已知数据。

每日完整流程可运行 `python pipeline.py`：先同步、运行 ETF Universe、执行上一交易日的模拟订单并生成今日订单，然后继续原有独立策略模拟与日志/推送流程。Pipeline 的原通知配置可能向已配置的接收方推送消息，请按部署环境检查。

### 推荐命令

| 目的 | 命令 |
| --- | --- |
| 仅同步 | `python main.py --sync-only` |
| 仅全市场分析 | `python -m etf_universe --config data/action-test-config.json --output data/universe` |
| 仅模拟盘 | `python -m simulation.etf_universe.daily --analysis-root data/universe` |
| 完整每日流程 | `python pipeline.py` |
| 离线分析 | `python -m etf_universe --offline --config data/action-test-config.json --output data/universe` |
| 只检查基础筛选 | `python -m etf_universe --screen-only --offline` |
| 生成真实持仓模板 | `python -m etf_universe --write-holdings-template data/etf_holdings.json` |
| 检查价格修正表 | `python -m etf_universe.data_quality --db data/etf_daily.db --force` |
| 测试 | `python -m unittest discover -s tests -q` |

`--include-simulation-holdings` 仅供原人工持仓分析模块附加读取原独立策略模拟状态；综合模拟盘自身不使用此开关，也不读取人工持仓。`python -m strategies.momentum_vol_filter.run` 一类命令只运行对应的原有策略。

## 数据同步、筛选和质量

日线位于 `data/etf_daily.db`。同步以 Tencent 为主源，单只失败时尝试 Sina；备用源缺乏稳定复权标记，因此不会覆盖已有重叠历史。普通同步补历史缺口，并只在允许取得完整当日日线后纳入今天的数据。

分析前检查异常价格跳变。可解释的拆分/除权只修正分析口径，无法解释的极端点隔离，原始日线不覆盖。Excel 的“数据异常检查”列出修正、可信度和人工核验状态。

正式池默认要求规模不少于 1 亿元、近 20 日成交额中位数不少于 1000 万元、上市满 180 天。上市 60–180 天进入新 ETF 观察池，更短历史仅标记数据不足。相同跟踪指数保留更合适的代表；相似主题且高度相关的产品再去重。筛选参数见 [示例配置](etf_universe/config.example.json)。

## 策略验证、市场环境与三类机会

全市场候选分别汇总策略数与独立家族数。历史验证检查风险收益、回撤、样本和成本；反弹额外看信号后 5/10 日修复表现。样本短时资格可能在强、合格、弱优势和未通过之间切换。模拟盘配置保留 `strategy_validation_refresh`（`daily`/`weekly`，默认 `weekly`）及历史缓存接口；当前正式分析仍每天重算验证，尚未把 weekly 冻结用于正式买点，以保留现有分析输出。

市场状态分为大盘强弱与候选横截面结构两层。趋势机会看中期方向、有效策略支持与价格位置；反弹机会看真实超跌、止跌 0–5 分及持续性；防守机会看波动、回撤及防守模型支持，**不要求满足趋势突破条件**。报告把“值得关注”和“可执行买入信号”分开，允许当天完全没有买点。

ETF 结构风险包括流动性、价格异常、跨境/QDII、折溢价及特殊机制；当前交易风险包括过热、趋势转弱、反弹失败等。跨境/QDII、港股通等特殊 ETF 仅提示额外风险，模拟盘不推算实时 NAV。资料或折溢价不完整时行动等级受限。

## 综合模拟盘：两组前向实验

`conservative` 是当前正式规则的**稳定基准对照组**。只有行动层明确给出可执行买入信号，且风险检查通过时才创建订单；继续观察、反弹观察、等待回调、暂不追高及未确认的防守候选不会买入。

`progressive` 是分阶段实验组：允许有策略支持、短期改善且未过热的早期趋势试仓；反弹 3/5 或 4/5 时还需超跌模型支持、未创新低、MA5 改善及无高风险阻断，才可小仓试错；5/5 且持续性通过可确认加仓；低波动且有防守支持的候选可形成防守仓。过热只阻止新增仓位，不自动卖出。两组使用相同本金、候选、行情、手续费、滑点、整手与成交规则，差别只在行动策略。实验要检验保守规则的踏空与进取规则的错误试仓，而非预设哪一组更好。

动作包括 `no_position`、`watch`、`probe_entry`、`add_position`、`hold`、`reduce`、`exit`，对应“暂不参与、观察、小仓试错、确认加仓、继续持有、减仓、退出”。配置位于 [模拟盘示例配置](simulation/etf_universe/config.example.json)：初始资金 100000 元，单只 ETF 上限为账户资产 25%，最多 4 只；试仓为该标的计划仓位的 20%，确认为 60%，最大计划比例 100%，反弹时间止损 8 个交易日。20%/60% 指单只计划预算的比例，不是总资产比例。

首次启动时，两账户分别创建空持仓、全额现金、启动日与策略版本，**不回填历史虚拟交易**。T 日收盘分析产生唯一 ID 的 `pending_orders` 并原子保存；T+1 同步完成后，用真实 `open[T+1]`、手续费及滑点模拟执行，再以 T+1 收盘更新净值和新信号。买入按 100 份取整；涨停不能买、跌停不能卖，停牌不成交。错过首个后续交易日的订单作废，不补造历史成交。同一日重复运行不会重复买卖、扣费或增加净值行。状态损坏时停止并保留原文件。版本写入状态和交易日志。

账户、订单、交易和净值保存在 `simulation/output/state_etf_universe_{conservative,progressive}.json`；同目录导出 `trades_etf_universe_*.csv`、`nav_etf_universe_*.csv` 与 `paper_summary.json`。JSON 是权威状态，CSV 可重建。真实持仓始终由用户维护。

### 按买入理由退出

买入时记录 `entry_type`、理由、日期、实际开盘成交价、入场指标、策略支持及市场状态。持仓评估复用现有分赛道持仓逻辑：趋势重点看连续跌破 MA20、策略支持恶化、趋势结构及峰值回撤，单日失守不立即卖；反弹检查前低、止跌分、MA5、修复目标和可配置的时间止损，失败更快退出；防守检查波动恶化与支持消失，不因涨得慢而卖。退出也先生成 T+1 待执行订单。

### A/B、踏空与错误试仓

每日记录初始本金、现金、市值、总资产、日/累计收益、峰值、当前/最大回撤、交易次数及持仓数。A/B 报告比较收益、回撤、交易次数、现金占比、持仓数、已实现和未实现收益。未来可扩展年化收益、Sharpe、Sortino、胜率、盈亏比、持有期、换手与交易成本。

`simulation/output/signal_history.db` 的 `signal_feature_history` 保存当日已知候选特征。`signal_outcomes` 记录保守与进取行动，并在**之后真实产生至少 10 个交易日行情时**回填 5/10 日收益、最大涨幅和最大回撤；达到上涨阈值标记 `missed_opportunity`，试仓后明显下跌标记 `false_probe`。未来字段位于独立表，决策代码只读取当日行动快照，避免未来数据泄露。阈值与这些标签是实验统计，不是自动交易模型。

## Excel 与输出

原工作簿保留“今日总览、趋势机会、反弹机会、防守机会、新ETF观察、持仓跟踪、卖出信号、策略验证、ETF筛选过程、风险检查、数据异常检查、完整候选池”。运行模拟盘后追加“模拟盘总览、模拟盘持仓、模拟交易记录、A_B模拟对比”。当天无成交时“模拟交易记录”明确显示“今日无模拟交易”。重复运行替换这四个 Sheet。

## 测试、限制与后续方向

运行 `python -m unittest discover -s tests -q`。测试覆盖筛选、行动、风险、报告、真实持仓与模拟盘状态、T+1、幂等及 A/B。`python -m simulation.etf_universe.daily --help` 查看模拟盘参数。

本地前向交易依赖收盘后完整、可靠的日线和资料缓存；缺失下一交易日开盘行情的订单不会假设成交。交易所复杂撮合、实时折溢价及真实券商接口尚未模拟。策略资格的 weekly 冻结仍需足够长的前向数据验证后再接入正式行动。机器学习交易模型尚未实现；未来可在积累足够长的特征与前向结果后，评估 Logistic Regression、LightGBM 或 XGBoost 对候选未来 5/10 日风险收益的二次评分，不直接替代规则引擎。

历史扩展说明存于 [文档归档](docs/history/etf_universe_legacy.md)。项目按 [MIT License](LICENSE) 分发，原作者版权声明保留在许可证正文中。
