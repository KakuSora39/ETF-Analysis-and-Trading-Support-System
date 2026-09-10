"""Chinese, plain-language presentation helpers; calculations stay unchanged."""

from __future__ import annotations

from datetime import date
import pandas as pd


STRATEGY_NAMES = {
    "momentum_rotation": "趋势动量轮动", "dual_momentum": "双重趋势动量", "momentum_dual": "双重趋势动量",
    "median_momentum": "中位趋势动量", "momentum_ma_etf": "均线过滤趋势动量", "momentum_ma_filter": "大盘均线过滤趋势动量",
    "momentum_vol_filter": "波动率过滤趋势动量", "multi_period_momentum": "多周期趋势动量", "composite_momentum": "多因子趋势动量",
    "sharpe_ranking": "风险收益排序", "sortino_ranking": "下行风险收益排序", "low_vol_rotation": "低波动轮动",
    "dual_ma_crossover": "双均线趋势", "macd_trend_rotation": "MACD 趋势", "rsi_trend_rotation": "RSI 趋势确认",
    "adx_trend_rotation": "趋势强度", "bollinger_reversion": "布林超跌回归", "bollinger_rotation": "布林强度",
    "contrarian_reversion": "逆向超跌回归", "mean_reversion": "超跌回归", "mean_reversion_rotation": "多指标超跌回归",
    "adaptive_rotation": "市场状态自适应", "donchian_breakout": "突破策略", "relative_strength": "相对强弱",
    "volume_price": "量价配合", "vol_price_momentum": "量价趋势动量", "tail_risk": "尾部风险切换",
    "alpha_etf_rotation": "模型选ETF轮动", "asset_allocation": "资产配置", "combined": "组合策略",
    "cross_border": "跨境ETF策略", "dca_timing": "定投择时", "gold_safe_haven": "黄金避险切换",
    "hs300_ma_timing": "沪深300均线择时", "industry_momentum": "行业动量",
    "lstm_etf_rotation": "LSTM模型轮动", "market_breadth": "市场宽度择时",
    "market_regime_rotation": "市场状态轮动", "neural_momentum": "神经网络动量",
    "pair_trading": "配对交易", "risk_switch_momentum": "风险切换动量", "risk_timing": "风险择时",
    "sector_rotation": "板块轮动", "spread_reversion": "价差回归",
}

COLUMNS = {
    "symbol": "ETF代码", "name": "ETF名称", "analysis_rank": "策略分析排名", "action_rank": "行动排序",
    "family_votes": "支持策略家族数", "strategy_votes": "支持策略数", "family_agreement": "策略家族支持率",
    "consensus_rank": "共识排名", "validated_family_votes": "有效策略家族数", "validated_strategy_votes": "有效策略数",
    "validated_strategies": "通过验证且当前支持的策略", "mean_backtest_sharpe": "平均风险收益表现",
    "backtest_weighted_score": "历史验证加权分", "backtest_eligible": "历史数据足够",
    "current_price": "当前价格", "return_5d": "近5日涨跌幅", "return_20d": "近20日涨跌幅",
    "ma20": "20日均价（MA20）", "ma60": "60日均价（MA60）", "distance_ma20": "距离20日均价",
    "distance_ma60": "距离60日均价", "distance_20d_high": "距离近20日最高价", "volatility_20d": "20日年化波动率",
    "max_drawdown_60d": "近60日最大回撤", "activity_change_5d": "近期成交活跃度变化", "activity_basis": "成交活跃度依据",
    "current_state": "当前状态", "action": "行动建议", "action_reason": "通俗解释", "strategy": "策略名称",
    "family": "策略家族", "status": "运行状态", "passed": "是否通过历史验证", "sharpe": "风险收益表现",
    "max_drawdown": "最大回撤", "volatility": "波动率", "total_return": "累计收益率", "annualized_return": "年化收益率",
    "benchmark_return": "基准收益率", "excess_return": "超额收益率", "reason": "说明", "rank": "策略内排名",
    "score": "策略原始分", "points": "名次积分", "weighted_points": "验证加权积分", "backtest_sharpe": "策略风险收益表现",
    "quality_weight": "历史质量权重", "decision": "策略验证结论", "decision_reason": "策略验证说明",
    "eligible": "是否进入正式候选池", "reasons": "未通过原因", "avg_amount": "20日平均成交额",
    "median_amount": "20日成交额中位数", "qualification": "上市资格", "observation_eligible": "是否进入新ETF观察池",
    "observation_reasons": "观察池未通过原因", "history_days": "有效行情天数", "data_completeness": "数据完整率", "aum": "基金规模",
    "listing_days": "上市天数", "fund_type": "基金类型", "amount_estimated": "成交额是否估算",
    "index_id": "跟踪指数标识", "aum_yuan": "基金规模（元）", "momentum": "基础动量",
    "duplicate_of": "同类保留的ETF", "listed_date": "上市日期", "known_on": "资料可知日期", "aum_date": "规模数据日期",
    "ranking_exclusion": "未进入排名原因", "family_points": "策略家族积分", "strategy_points": "策略积分",
    "avg_rank": "平均策略内排名", "strategy_agreement": "策略支持率", "position_data_sufficient": "位置判断数据足够",
    "tested_symbols": "回测使用的ETF", "days": "回测交易日数", "calmar": "收益回撤表现", "trades": "交易次数",
    "exposure": "持仓时间占比", "selected_count": "本次选中数量", "eligible_count": "可参与比较数量", "detail": "状态说明",
    "return_1d": "近1日涨跌幅", "recent_avg_amount": "最近5日平均成交额", "consecutive_up_days": "连续上涨天数",
    "etf_type": "ETF类型", "special_risk_note": "特殊品种风险提示", "liquidity_status": "流动性检查",
    "liquidity_reason": "流动性说明", "price_risk_status": "价格异常检查", "price_risk_reason": "价格异常说明",
    "premium_rate": "折溢价率", "premium_status": "溢价检查", "premium_reason": "溢价说明",
    "premium_confirmation_needed": "买入前需要确认溢价", "etf_risk_level": "ETF特有风险级别",
    "structural_risk_level": "ETF结构风险", "structural_risk_reason": "结构风险说明",
    "trading_risk_level": "当前交易风险", "trading_risk_reason": "交易风险说明",
    "effective_consensus": "有效策略共识", "total_family_coverage": "总策略家族覆盖", "consensus_label": "共识标签",
    "risk_blocking": "是否阻止买入", "risk_block_reason": "阻止买入原因", "risk_force_observe": "是否要求继续观察",
    "risk_level": "结构化风险等级", "risk_tags": "风险标签", "warnings": "风险提示",
    "announcement_status": "公告检查", "announcement_warnings": "公告风险提示",
    "opportunity_lane": "机会赛道", "opportunity_type": "机会类型",
    "trend_support": "趋势原始支持组数", "rebound_support": "反弹原始支持组数", "defense_support": "防守原始支持组数",
    "validated_trend_support": "趋势有效支持组数", "validated_rebound_support": "反弹有效支持组数",
    "validated_defense_support": "防守有效支持组数", "stopping_score": "止跌确认分", "stopping_signals": "止跌信号",
    "return_3d": "近3日涨跌幅", "ma5": "5日均价（MA5）", "ma5_up": "MA5是否向上", "above_ma5": "是否站上MA5",
    "recent_3d_no_new_low": "最近3日是否未创新低", "rsi14": "RSI指标", "rsi_recovering": "RSI是否从低位回升",
    "close_above_previous": "收盘价是否高于前一日",
    "recent_3d_non_new_low_days": "最近3日守住前期低点天数", "recent_2d_sharp_decline": "最近2日是否连续明显下跌",
    "latest_broke_recent_low": "最新价是否跌破近期低点", "rebound_persistence_pass": "反弹持续性是否通过",
    "rebound_persistence_reason": "反弹持续性说明",
    "rebound_from_20d_low": "距20日低点反弹幅度", "forward_5d_mean": "信号后5日平均收益",
    "forward_10d_mean": "信号后10日平均收益", "forward_10d_win_rate": "信号后10日上涨概率",
    "average_adverse_excursion": "平均最大不利波动", "payoff_ratio": "盈亏比", "signal_count": "信号数量",
}

STATUS_NAMES = {"voted": "已投票", "abstained": "当前无信号", "not_applicable": "不适用于全市场横向比较", "controller": "市场环境控制器", "error": "运行失败", "completed": "回测完成"}


def strategy_list(value) -> str:
    return "、".join(STRATEGY_NAMES.get(item, item) for item in str(value or "").split(";") if item)


def chinese_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    # The old decision only measures strategy validation.  Once the action layer
    # exists, showing it beside the final action creates two apparent buy calls.
    if "action" in result:
        result = result.drop(columns=["decision", "decision_reason"], errors="ignore")
    result = result.drop(columns=["risk_result", "risk_level", "details"], errors="ignore")
    if "strategy" in result:
        result["strategy"] = result["strategy"].map(lambda x: STRATEGY_NAMES.get(str(x), str(x)))
    if "fund_type" in result:
        result["fund_type"] = result["fund_type"].map({
            "equity": "普通股票ETF", "qdii_equity": "跨境ETF / QDII", "commodity": "商品ETF",
            "qdii_commodity": "跨境商品ETF / QDII", "bond": "债券ETF", "money": "货币ETF", "unknown": "其他特殊类型",
        }).fillna("其他特殊类型")
    if "family" in result:
        result["family"] = result["family"].map({
            "not_comparable": "不参与横向比较", "market_controller": "市场环境控制", "regime": "市场状态",
            "unknown": "未分类", "unclassified": "未分类",
        }).fillna(result["family"])
    if "validated_strategies" in result:
        result["validated_strategies"] = result["validated_strategies"].map(strategy_list)
    if "status" in result:
        result["status"] = result["status"].map(lambda x: STATUS_NAMES.get(str(x), str(x)))
    if "opportunity_lane" in result:
        result["opportunity_lane"] = result["opportunity_lane"].map({"trend": "趋势机会", "rebound": "超跌反弹", "defense": "防守机会"}).fillna("")
    if "passed" in result:
        result["passed"] = result["passed"].map({True: "是", False: "否"}).fillna("")
    if "backtest_eligible" in result:
        result["backtest_eligible"] = result["backtest_eligible"].map({True: "是", False: "否"}).fillna("")
    if "position_data_sufficient" in result:
        result["position_data_sufficient"] = result["position_data_sufficient"].map({True: "是", False: "否"}).fillna("")
    for column in ("premium_confirmation_needed", "risk_blocking", "risk_force_observe", "ma5_up", "above_ma5", "recent_3d_no_new_low", "rsi_recovering", "close_above_previous", "recent_2d_sharp_decline", "latest_broke_recent_low", "rebound_persistence_pass"):
        if column in result:
            result[column] = result[column].map({True: "是", False: "否"}).fillna("")
    for column in ("risk_tags", "warnings"):
        if column in result:
            result[column] = result[column].map(lambda values: "；".join(values) if isinstance(values, list) else str(values))
    if "sharpe" in result:
        result["risk_return_explanation"] = result["sharpe"].map(_sharpe_explanation)
    if "max_drawdown" in result:
        result["drawdown_explanation"] = result["max_drawdown"].map(
            lambda x: "数据不足" if pd.isna(x) else f"历史最差阶段曾从高点回落约 {abs(x):.1%}。"
        )
    if "volatility" in result:
        result["volatility_explanation"] = "数值越高，历史价格起伏越大；需要承受的短期波动也越明显。"
    result = result.rename(columns={
        "risk_return_explanation": "风险收益表现说明", "drawdown_explanation": "最大回撤说明",
        "volatility_explanation": "波动率说明",
    })
    return result.rename(columns={key: value for key, value in COLUMNS.items() if key in result.columns})


def _p(value) -> str:
    return "数据不足" if pd.isna(value) else f"{value:+.2%}"


def _sharpe_explanation(value) -> str:
    if pd.isna(value): return "缺少足够的历史验证结果。"
    if value >= 1: return "过去回测中，收益相对于承担的波动表现较好。"
    if value >= 0.5: return "过去回测中的风险收益表现尚可，但仍需结合回撤判断。"
    if value > 0: return "过去回测中的风险收益表现偏弱。"
    return "过去回测没有体现出有效的风险收益优势。"


def build_markdown_report(actions: pd.DataFrame, as_of: str) -> str:
    lines = [f"# ETF 分析报告（{as_of}）", "", "本报告先看策略是否经过历史验证，再检查 ETF 自身的流动性、价格异常、折溢价和特殊结构风险，最后判断当前价格位置。策略排名靠前不代表现在就适合买入。", ""]
    buys = actions[actions.action == "可以关注买入"] if not actions.empty else actions
    if buys.empty:
        lines += ["> 今日暂无合适买入候选。可以继续跟踪等待，但不需要为了交易而交易。", ""]
    for _, row in actions.iterrows():
        strategies = strategy_list(row.get("validated_strategies", "")) or "暂无通过历史验证且当前支持的策略"
        lines += [
            f"## {row['symbol']} {row.get('name', '')}", "",
            "**为什么入选**", "",
            f"- 当前支持策略：{strategies}",
            f"- 共 {int(row.get('validated_family_votes', 0))} 类有效策略支持", "",
            "**历史策略表现**", "",
            f"- 平均风险收益表现：{row.get('mean_backtest_sharpe', float('nan')):.2f}" if pd.notna(row.get('mean_backtest_sharpe')) else "- 平均风险收益表现：数据不足",
            f"- 说明：{_sharpe_explanation(row.get('mean_backtest_sharpe'))}", "",
            "**ETF 特有风险检查**", "",
            f"- ETF 类型：{row.get('etf_type', '数据不可用')}",
            f"- 特殊品种提示：{row.get('special_risk_note', '数据不可用')}",
            f"- 流动性：【{row.get('liquidity_status', '数据不可用')}】；{row.get('liquidity_reason', '')}",
            f"- 价格异常：【{row.get('price_risk_status', '数据不可用')}】；{row.get('price_risk_reason', '')}",
            f"- 溢价检查：【{row.get('premium_status', '数据不可用')}】；{row.get('premium_reason', '')}",
            f"- 公告检查：【{row.get('announcement_status', '暂无可靠数据')}】" + (f"；{row.get('announcement_warnings')}" if row.get('announcement_warnings') else ""),
            f"- ETF 特有风险级别：{row.get('etf_risk_level', '数据不可用')}", "",
            "**当前状态**", "",
            f"- 近 5 日：{_p(row.get('return_5d'))}", f"- 近 20 日：{_p(row.get('return_20d'))}",
            f"- 距离 20 日均价：{_p(row.get('distance_ma20'))}", f"- 距离 60 日均价：{_p(row.get('distance_ma60'))}",
            f"- 距离近 20 日最高价：{_p(row.get('distance_20d_high'))}", f"- 20 日年化波动率：{_p(row.get('volatility_20d'))}",
            "- 波动率说明：数值越高，近期价格起伏越大。",
            f"- 近 60 日最大回撤：{_p(row.get('max_drawdown_60d'))}", "",
            f"- 最近{row.get('activity_basis', '成交')}活跃度变化：{_p(row.get('activity_change_5d'))}",
            "- 成交说明：最近 5 日平均活跃度相对之前 15 日的变化，正数表示交易更活跃。", "",
            f"**行动建议：【{row['action']}】**", "", f"通俗解释：{row['action_reason']}", "",
        ]
    lines += ["## 总表", "", "| ETF | 为什么入选 | ETF特有风险 | 当前状态 | 行动建议 | 通俗解释 |", "|---|---|---|---|---|---|"]
    for _, row in actions.iterrows():
        why = strategy_list(row.get("validated_strategies", "")) or "策略依据不足"
        explanation = str(row.get("action_reason", "")).replace("|", "｜")
        risk = f"{row.get('etf_type', '类型未知')}；{row.get('liquidity_status', '流动性未知')}；{row.get('premium_status', '溢价未知')}"
        lines.append(f"| {row['symbol']} {row.get('name', '')} | {why}；{int(row.get('validated_family_votes', 0))} 类有效支持 | {risk} | {row['current_state']} | {row['action']} | {explanation} |")
    return "\n".join(lines) + "\n"


def build_lane_markdown_report(
    lanes: dict[str, pd.DataFrame], regime, as_of: str, report_date: date,
    summary_lanes: dict[str, pd.DataFrame] | None = None,
    validated_denominators: dict[str, int] | None = None,
) -> str:
    """Keep the readable Markdown report, separated by opportunity type and correct dates."""
    lines = [
        f"# ETF分析与交易辅助报告（{report_date.isoformat()}）", "",
        f"- 报告生成日期：{report_date.isoformat()}", f"- 行情截止日期：{as_of}",
        f"- 大盘状态：{regime.broad_label}", f"- 市场结构：{regime.structure_label}",
        f"- 当前市场：{regime.label}", f"- 说明：{regime.explanation}", "",
    ]
    actionable = {
        "trend": {"可以关注买入", "等待回调", "继续观察", "暂不追高"},
        "rebound": {"小仓反弹试错", "较高优先级反弹候选", "高优先级反弹候选", "反弹观察", "继续观察"},
        "defense": {"防守候选", "稳健持有候选"},
    }
    if not any((~frame.action.eq("暂不参与")).any() for frame in lanes.values() if not frame.empty):
        lines += ["> 今日暂无合适机会。", ""]
    worth = any((~frame.action.eq("暂不参与")).any() for frame in lanes.values() if not frame.empty)
    executable_actions = {"可以关注买入", "小仓反弹试错", "较高优先级反弹候选", "高优先级反弹候选"}
    executable = any(frame.action.isin(executable_actions).any() for frame in lanes.values() if not frame.empty)
    lines += [f"- 今日存在值得关注的机会：{'是' if worth else '否'}", f"- 今日存在可执行买入信号：{'是' if executable else '否'}", ""]
    titles = {"trend": "趋势机会", "rebound": "反弹机会", "defense": "防守机会"}
    display_lanes = summary_lanes or lanes
    for lane, title in titles.items():
        frame = display_lanes[lane]
        shown = frame[frame.action.isin(actionable[lane])] if not frame.empty else frame
        lines += [f"## {title}", ""]
        if shown.empty:
            if lane == "defense" and int((validated_denominators or {}).get("defense", 0)) == 0:
                lines += ["当前暂无有效防守策略，系统暂时无法可靠评价防守机会。", ""]
            else:
                lines += ["今日暂无满足条件的该赛道机会。", ""]
            continue
        lines += ["| ETF | 有效策略支持 | 有效家族支持 | 全部家族覆盖 | 当前状态 | ETF结构风险 | 当前交易风险 | 行动建议 | 原因 |", "|---|---|---|---|---|---|---|---|---|"]
        strategies_col = f"validated_{lane}_strategies"
        for _, row in shown.iterrows():
            strategy_support = row.get("effective_strategy_consensus", "暂无")
            family_support = f"{row.get('effective_family_consensus', '暂无')}（{row.get('consensus_label', '')}）"
            reason = str(row.action_reason).replace("|", "｜")
            lines.append(f"| {row.symbol} {row.get('name', '')} | {strategy_support} | {family_support} | {row.get('total_family_coverage', '')} | {row.current_state} | {row.get('structural_risk_level', '')} | {row.get('trading_risk_level', '')} | {row.action} | {reason} |")
        lines.append("")
    from .theme import build_theme_summary
    themes = build_theme_summary(display_lanes)
    defense_themes = themes[themes.opportunity_lane == "defense"] if not themes.empty else themes
    lines += ["## 防守主题摘要", ""]
    if defense_themes.empty:
        lines += ["暂无防守主题。", ""]
    else:
        lines += ["| 防守主题 | 代表ETF | 备选ETF |", "|---|---|---|"]
        for _, row in defense_themes.iterrows():
            lines.append(f"| {row.theme} | {row.representative_symbol} {row.representative_name} | {row.alternative_etfs} |")
        lines.append("")
    excluded = sum(int(frame.action.eq("暂不参与").sum()) for frame in lanes.values() if not frame.empty)
    lines += ["## 暂不参与", "", f"三个赛道合计有 {excluded} 条候选记录暂不参与，详细原因请查看Excel对应工作表。", ""]
    return "\n".join(lines)
