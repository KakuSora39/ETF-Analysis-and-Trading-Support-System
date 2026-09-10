"""CLI: full-market cleanup, strategy voting, and consensus focus list."""

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import sys

import pandas as pd

from .action import attach_position_metrics
from .consensus import run_consensus
from .data import DEFAULT_DB, load_market, load_metadata, load_strategy_market, write_template
from .data_quality import prepare_adjusted_history
from .excel_report import generate_excel_report
from .holdings import analyze_holdings, load_holdings, write_holdings_template
from .lane_actions import build_lane_actions
from .market_regime import detect_market_regime
from .opportunities import STRATEGY_LANES, attach_lane_validation_metrics, attach_weak_rebound_support, build_lane_shortlist, validated_lane_support
from .presentation import build_lane_markdown_report, chinese_frame
from .risk import RiskConfig, assess_etf_risks
from .screen import FilterConfig, screen_universe
from .theme import deduplicate_summary_lanes
from .validation import backtest_strategies, validate_rebound_strategies


def _parser():
    parser = argparse.ArgumentParser(description="ETF全市场分析与交易辅助报告")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--as-of", help="信号日 YYYY-MM-DD；默认数据库最新交易日")
    parser.add_argument("--metadata", type=Path, help="基金资料 CSV / JSON；主要用于历史快照或离线测试")
    parser.add_argument("--cache", type=Path, help="自动资料与真实成交额缓存；默认与 --db 同目录的 etf_universe.db")
    parser.add_argument("--offline", action="store_true", help="只用本地缓存，不访问网络")
    parser.add_argument("--refresh", action="store_true", help="强制刷新资料与成交额")
    parser.add_argument("--workers", type=int, default=4, help="自动采集并发数，默认4")
    parser.add_argument("--config", type=Path, help="JSON 基础清洗配置，字段见 FilterConfig")
    parser.add_argument("--write-template", type=Path, help="导出全市场资料模板后退出")
    parser.add_argument("--holdings", type=Path, help="人工持仓JSON；默认 data/etf_holdings.json，不存在时按空持仓处理")
    parser.add_argument("--write-holdings-template", type=Path, help="写出人工持仓JSON模板后退出")
    parser.add_argument("--include-simulation-holdings", action="store_true", help="同时读取 simulation/output/state_*.json 模拟持仓")
    parser.add_argument("--simulation-state-dir", type=Path, default=Path("simulation/output"), help="模拟盘状态目录")
    parser.add_argument("--output", type=Path, default=DEFAULT_DB.parent / "universe")
    parser.add_argument("--screen-only", action="store_true", help="只做基础清洗，不运行策略共识")
    parser.add_argument("--strategy-top-n", type=int, default=3, help="每个策略投前N只，默认3")
    parser.add_argument("--focus-top-n", type=int, default=10, help="最终重点关注前N只，默认10")
    parser.add_argument("--strategy-history-days", type=int, default=756, help="策略信号与验证历史长度，默认756个交易日")
    parser.add_argument("--skip-backtest", action="store_true", help="跳过重点候选的统一验证回测")
    parser.add_argument("--backtest-days", type=int, default=504, help="统一验证回测天数，默认504个交易日")
    parser.add_argument("--rebalance-days", type=int, default=5, help="验证回测每隔多少交易日重新检查信号，默认5")
    parser.add_argument("--one-way-cost", type=float, default=0.0003, help="验证回测单边交易成本，默认0.03%%")
    parser.add_argument("--max-backtest-drawdown", type=float, default=0.35, help="策略通过验证允许的最大回撤，默认0.35")
    parser.add_argument("--min-backtest-sharpe", type=float, default=0.0, help="策略通过验证所需最低夏普，默认0")
    parser.add_argument("--min-buy-families", type=int, default=2, help="趋势机会判为可以关注买入所需最少有效支持组数，默认2")
    parser.add_argument("--min-buy-mean-sharpe", type=float, default=0.5, help="判为可考虑买入所需策略平均夏普，默认0.5")
    parser.add_argument("--liquidity-days", type=int)
    parser.add_argument("--min-avg-amount", type=float, help="最低日均成交额，元")
    parser.add_argument("--min-listing-days", type=int, help="最少上市自然日")
    parser.add_argument("--min-observation-days", type=int, help="进入新ETF观察池所需最少上市自然日，默认60；更短者仅标记数据不足")
    parser.add_argument("--min-aum", type=float, help="最低基金净资产规模，元")
    parser.add_argument("--max-metadata-age-days", type=int)
    parser.add_argument("--correlation-days", type=int)
    parser.add_argument("--max-correlation", type=float)
    parser.add_argument("--momentum-days", type=int)
    parser.add_argument("--top-n", type=int, help="基础清洗诊断动量榜数量；不参与策略共识")
    parser.add_argument("--volume-unit", choices=["shares", "lots"], help="人工资料模式下旧库成交量单位")
    return parser


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = _parser()
    args = parser.parse_args(argv)
    if args.offline and args.refresh:
        parser.error("--offline 与 --refresh 不能同时使用")
    try:
        values = json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else {}
        for key in asdict(FilterConfig()):
            value = getattr(args, key, None)
            if value is not None:
                values[key] = value
        config = FilterConfig(**values)
        if args.write_holdings_template:
            target = write_holdings_template(args.write_holdings_template)
            print(f"已写出人工持仓模板：{target}")
            return 0
        if args.metadata is None and config.volume_unit is not None:
            raise ValueError("自动模式使用真实成交额，无需 --volume-unit；该参数只用于显式 --metadata 模式")

        quality_summary, price_anomalies = prepare_adjusted_history(args.db)
        print(
            f"价格断层检查：异常 {quality_summary.anomaly_count} 条，"
            f"已修正 {quality_summary.repaired_count} 条，隔离 {quality_summary.isolated_count} 条"
        )
        daily, universe, as_of = load_market(args.db, config, args.as_of)
        if args.write_template:
            write_template(universe, args.write_template)
            print(f"已导出 {len(universe)} 只ETF的资料模板：{args.write_template}")
            return 0

        refresh_summary = None
        information_as_of = as_of
        cache_path = args.cache or args.db.parent / "etf_universe.db"
        if args.metadata is None:
            from .cache import UniverseCache
            from .provider import china_today
            from .refresh import refresh_cache

            cache = UniverseCache(cache_path)
            if args.as_of is None:
                information_as_of = china_today()
                if (pd.Timestamp(information_as_of) - pd.Timestamp(as_of)).days > 10:
                    raise ValueError(f"本地最新行情为 {as_of}，已超过10个自然日，请先同步行情；历史分析请显式指定 --as-of")
            dates = sorted(pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d").unique())
            historical = args.as_of is not None and as_of < china_today()
            if historical and cache.metadata(as_of).empty:
                raise ValueError(f"没有 {as_of} 当时可获知的历史资料；请提供 --metadata 历史快照，或不指定 --as-of 做当前分析")
            if not args.offline:
                refresh_summary = refresh_cache(
                    cache, universe["symbol"].tolist(), dates[-min(config.liquidity_days, len(dates))], as_of,
                    workers=args.workers, force=args.refresh, refresh_metadata=not historical,
                )
            daily = cache.overlay_amounts(daily)
            metadata = cache.metadata(information_as_of)
            if metadata.empty and as_of < china_today():
                raise ValueError(f"缓存中没有 {as_of} 当时可获知的资料")
        else:
            metadata = load_metadata(args.metadata)

        screened = screen_universe(daily, metadata, as_of, config, universe,
                                   information_as_of=information_as_of)
        holdings_path = args.holdings or args.db.parent / "etf_holdings.json"
        holdings = load_holdings(
            holdings_path,
            args.simulation_state_dir if args.include_simulation_holdings else None,
        )
        new_etfs = screened.new_etfs.copy() if screened.new_etfs is not None else pd.DataFrame()
        if not new_etfs.empty:
            recent_data = {
                symbol: group.sort_values("date").reset_index(drop=True)
                for symbol, group in daily[daily.symbol.isin(new_etfs.symbol)].groupby("symbol")
            }
            new_etfs = attach_position_metrics(new_etfs, recent_data, as_of)
            new_etfs["has_unresolved_data_anomaly"] = new_etfs.symbol.isin(
                price_anomalies.loc[price_anomalies.get("isolated", False).astype(bool), "symbol"].astype(str)
            ) if not price_anomalies.empty else False
            new_etfs = assess_etf_risks(new_etfs, RiskConfig())
            new_etfs["observation_status"] = new_etfs["qualification"].map({
                "数据不足": "数据不足", "新ETF观察": "新ETF观察",
            }).fillna("仅观察")
            new_etfs["observation_explanation"] = new_etfs.apply(
                lambda row: "上市不足60天，行情样本太短，不进入正式策略池，也不产生买入建议。"
                if row["qualification"] == "数据不足" else
                "上市60至180天，可观察当前行情和结构风险；不做长期历史验证，最高行动等级受限且本表不产生买入建议。",
                axis=1,
            )
        output = args.output / f"{as_of}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        output.mkdir(parents=True, exist_ok=False)
        chinese_frame(screened.audit).to_csv(output / "basic_filter_audit.csv", index=False, encoding="utf-8-sig")
        chinese_frame(screened.candidates).to_csv(output / "candidates.csv", index=False, encoding="utf-8-sig")
        chinese_frame(screened.ranking).to_csv(output / "screen_momentum_ranking.csv", index=False, encoding="utf-8-sig")
        chinese_frame(screened.selected).to_csv(output / "screen_momentum_top.csv", index=False, encoding="utf-8-sig")
        chinese_frame(new_etfs).to_csv(output / "new_etf_observation.csv", index=False, encoding="utf-8-sig")

        reasons = (screened.audit.loc[~screened.audit["eligible"], "reasons"]
                   .str.split(";").explode().value_counts().to_dict())
        summary = {
            "as_of": as_of,
            "information_as_of": screened.information_as_of,
            "config": asdict(config),
            "universe_count": len(screened.audit),
            "candidate_count": len(screened.candidates),
            "new_etf_count": len(new_etfs),
            "holding_count": len(holdings),
            "candidate_symbols": screened.symbols,
            "exclusion_counts": reasons,
            "amount_estimated_count": int(screened.audit["amount_estimated"].sum()),
            "data_refresh": refresh_summary,
            "signal_timing": "收盘后生成信号，最早在下一交易日执行",
            "price_quality": asdict(quality_summary),
        }

        focus = pd.DataFrame()
        holding_tracking, sell_signals = pd.DataFrame(), pd.DataFrame()
        if not args.screen_only and not screened.candidates.empty:
            strategy_data, benchmark, calendar = load_strategy_market(
                args.db, screened.symbols, as_of, args.strategy_history_days,
                cache_path=cache_path if args.metadata is None else None,
            )
            consensus = run_consensus(
                strategy_data, benchmark, screened.candidates,
                strategy_top_n=args.strategy_top_n, focus_top_n=args.focus_top_n,
                strategy_root=Path(__file__).resolve().parents[1] / "strategies",
            )
            chinese_frame(consensus.status).to_csv(output / "strategy_status.csv", index=False, encoding="utf-8-sig")
            chinese_frame(consensus.votes).to_csv(output / "strategy_votes.csv", index=False, encoding="utf-8-sig")
            chinese_frame(consensus.ranking).to_csv(output / "consensus.csv", index=False, encoding="utf-8-sig")
            regime = detect_market_regime(benchmark, strategy_data)
            lane_shortlist, lane_denominators = build_lane_shortlist(consensus, screened.candidates, args.focus_top_n)
            chinese_frame(lane_shortlist).to_csv(output / "focus.csv", index=False, encoding="utf-8-sig")
            focus = lane_shortlist
            counts = consensus.status["status"].value_counts().to_dict()
            summary.update({
                "strategy_calendar_start": calendar[0] if calendar else None,
                "strategy_calendar_days": len(calendar),
                "strategy_top_n": args.strategy_top_n,
                "focus_top_n": args.focus_top_n,
                "strategy_status_counts": counts,
                "voting_strategy_count": int((consensus.status.status == "voted").sum()),
                "voting_family_count": int(consensus.status.loc[consensus.status.status == "voted", "family"].nunique()),
                "focus_symbols": focus["symbol"].tolist(),
                "selected_symbols": focus["symbol"].tolist(),
                "market_regime": regime.label,
                "broad_market_state": regime.broad_label,
                "market_structure": regime.structure_label,
                "market_regime_explanation": regime.explanation,
                "lane_active_denominators": lane_denominators,
                "active_lane_family_denominators": lane_denominators,
                "active_lane_strategy_denominators": {
                    lane: int(lane_shortlist.get(f"{lane}_active_strategy_denominator", pd.Series([0])).iloc[0])
                    for lane in ("trend", "rebound", "defense")
                },
            })
            if not args.skip_backtest and not focus.empty and len(calendar) >= 311:
                backtests = backtest_strategies(
                    strategy_data, benchmark, focus["symbol"].tolist(),
                    backtest_days=args.backtest_days,
                    rebalance_days=args.rebalance_days,
                    one_way_cost=args.one_way_cost,
                    max_drawdown=args.max_backtest_drawdown,
                    min_sharpe=args.min_backtest_sharpe,
                )
                rebound_validation = validate_rebound_strategies(
                    strategy_data, focus["symbol"].tolist(),
                    backtest_days=args.backtest_days, signal_interval=args.rebalance_days,
                    top_n=args.strategy_top_n,
                )
                long_valid = set(backtests.loc[
                    backtests.get("passed", False).fillna(False) & ~backtests.strategy.map(STRATEGY_LANES).eq("rebound"), "strategy"
                ])
                rebound_valid = set(rebound_validation.loc[rebound_validation.get("passed", False).fillna(False), "strategy"])
                rebound_weak = set(rebound_validation.loc[rebound_validation.get("validation_level", "").eq("弱优势"), "strategy"])
                valid_strategy_names = long_valid | rebound_valid
                analysis, validated_denominators = validated_lane_support(
                    focus, consensus, valid_strategy_names,
                )
                analysis = attach_weak_rebound_support(analysis, consensus, rebound_weak)
                analysis = attach_lane_validation_metrics(analysis, consensus, backtests, rebound_validation)
                position_metrics = attach_position_metrics(analysis, strategy_data, as_of)
                relevant_anomalies = price_anomalies[
                    price_anomalies["date"].astype(str).le(as_of)
                    & price_anomalies["symbol"].astype(str).isin(position_metrics["symbol"].astype(str))
                ].copy()
                if not relevant_anomalies.empty:
                    anomaly_summary = relevant_anomalies.groupby("symbol").agg(
                        price_anomaly_count=("date", "size"),
                        unresolved_price_anomaly_count=("isolated", lambda values: int(values.astype(bool).sum())),
                    )
                    position_metrics = position_metrics.merge(anomaly_summary, left_on="symbol", right_index=True, how="left")
                for column in ("price_anomaly_count", "unresolved_price_anomaly_count"):
                    if column not in position_metrics:
                        position_metrics[column] = 0
                    else:
                        position_metrics[column] = position_metrics[column].fillna(0).astype(int)
                position_metrics["has_unresolved_data_anomaly"] = position_metrics["unresolved_price_anomaly_count"].gt(0)
                risks = assess_etf_risks(
                    position_metrics, RiskConfig(),
                )
                lanes = build_lane_actions(risks, regime, args.min_buy_families, args.min_buy_mean_sharpe)
                if not holdings.empty:
                    holding_data, _, _ = load_strategy_market(
                        args.db, holdings.symbol.drop_duplicates().tolist(), as_of,
                        args.strategy_history_days,
                        cache_path=cache_path if args.metadata is None else None,
                    )
                    holding_tracking, sell_signals = analyze_holdings(
                        holdings, holding_data, lanes, as_of,
                    )
                    chinese_frame(holding_tracking).to_csv(output / "holding_tracking.csv", index=False, encoding="utf-8-sig")
                    chinese_frame(sell_signals).to_csv(output / "sell_signals.csv", index=False, encoding="utf-8-sig")
                summary["sell_signal_count"] = len(sell_signals)
                summary_lanes = deduplicate_summary_lanes(lanes)
                actions = pd.concat([frame for frame in lanes.values() if not frame.empty], ignore_index=True, sort=False)
                selected_actions = {
                    "可以关注买入", "小仓反弹试错", "较高优先级反弹候选", "高优先级反弹候选",
                }
                selected = actions[actions.action.isin(selected_actions)].copy()
                validated_votes = consensus.votes[consensus.votes.strategy.isin(valid_strategy_names) & consensus.votes.symbol.isin(focus.symbol)].copy()
                validated_votes["opportunity_lane"] = validated_votes.strategy.map(STRATEGY_LANES)
                chinese_frame(backtests).to_csv(output / "strategy_backtests.csv", index=False, encoding="utf-8-sig")
                chinese_frame(rebound_validation).to_csv(output / "rebound_strategy_validation.csv", index=False, encoding="utf-8-sig")
                chinese_frame(validated_votes).to_csv(output / "validated_strategy_votes.csv", index=False, encoding="utf-8-sig")
                chinese_frame(risks).to_csv(output / "etf_risk_checks.csv", index=False, encoding="utf-8-sig")
                chinese_frame(actions).to_csv(output / "buy_analysis.csv", index=False, encoding="utf-8-sig")
                chinese_frame(selected).to_csv(output / "selected.csv", index=False, encoding="utf-8-sig")
                report_date = datetime.now(timezone(timedelta(hours=8))).date()
                (output / "analysis_report.md").write_text(
                    build_lane_markdown_report(lanes, regime, as_of, report_date, summary_lanes, validated_denominators), encoding="utf-8-sig"
                )
                summary.update({
                    "backtest_days_requested": args.backtest_days,
                    "rebalance_days": args.rebalance_days,
                    "one_way_cost": args.one_way_cost,
                    "max_backtest_drawdown": args.max_backtest_drawdown,
                    "min_backtest_sharpe": args.min_backtest_sharpe,
                    "min_buy_families": args.min_buy_families,
                    "min_buy_mean_sharpe": args.min_buy_mean_sharpe,
                    "backtested_strategy_count": len(backtests),
                    "validated_strategy_count": len(valid_strategy_names),
                    "validated_rebound_strategy_count": len(rebound_valid),
                    "validated_lane_denominators": validated_denominators,
                    "validated_lane_strategy_denominators": {
                        lane: int(analysis.get(f"{lane}_validated_strategy_denominator", pd.Series([0])).iloc[0])
                        for lane in ("trend", "rebound", "defense")
                    },
                    "active_lane_family_denominators": {
                        lane: int(analysis.get(f"{lane}_active_family_denominator", pd.Series([0])).iloc[0])
                        for lane in ("trend", "rebound", "defense")
                    },
                    "weak_rebound_strategy_count": len(rebound_weak),
                    "buy_symbols": lanes["trend"].loc[lanes["trend"].action == "可以关注买入", "symbol"].tolist(),
                    "rebound_trial_symbols": lanes["rebound"].loc[lanes["rebound"].action.isin([
                        "小仓反弹试错", "较高优先级反弹候选", "高优先级反弹候选",
                    ]), "symbol"].tolist(),
                    "defense_symbols": lanes["defense"].loc[lanes["defense"].action.isin(["防守候选", "稳健持有候选"]), "symbol"].tolist(),
                    "watch_symbols": actions.loc[actions.action.isin([
                        "等待回调", "继续观察", "暂不追高", "反弹观察", "防守候选", "稳健持有候选",
                    ]), "symbol"].drop_duplicates().tolist(),
                    "action_counts": actions["action"].value_counts().to_dict(),
                    "etf_risk_counts": risks["structural_risk_level"].value_counts().to_dict(),
                    "liquidity_risk_symbols": risks.loc[risks.liquidity_status == "流动性偏弱", "symbol"].tolist(),
                })
                summary["selected_symbols"] = selected["symbol"].drop_duplicates().tolist()
                workbook = generate_excel_report(
                    output, report_date, as_of, regime, summary, lanes, consensus.status,
                    backtests, rebound_validation, screened.audit, screened.candidates,
                    risks, focus, summary_lanes,
                    price_anomalies,
                    new_etfs,
                    holding_tracking,
                    sell_signals,
                )
                summary["excel_report"] = workbook.name
            else:
                chinese_frame(focus).to_csv(output / "selected.csv", index=False, encoding="utf-8-sig")
                summary["backtest_skipped"] = True
                if not args.skip_backtest and len(calendar) < 311:
                    summary["backtest_skip_reason"] = "共同历史不足311个交易日"
        else:
            chinese_frame(screened.ranking).to_csv(output / "consensus.csv", index=False, encoding="utf-8-sig")
            chinese_frame(screened.selected).to_csv(output / "selected.csv", index=False, encoding="utf-8-sig")
            summary["selected_symbols"] = screened.selected["symbol"].tolist()
            summary["screen_only"] = True

        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"行情日 {as_of}，资料截止 {information_as_of}：全市场 {len(screened.audit)} → 基础清洗候选 {len(screened.candidates)}")
        if not args.screen_only:
            print(f"投票策略 {summary.get('voting_strategy_count', 0)} 个，独立策略家族 {summary.get('voting_family_count', 0)} 个 → 三赛道合并候选 {len(focus)} 只")
            if not focus.empty:
                columns = [c for c in ("consensus_rank", "symbol", "name", "family_votes", "strategy_votes", "family_agreement") if c in focus]
                print(chinese_frame(focus[columns]).to_string(index=False))
            if "buy_symbols" in summary:
                print(f"分赛道历史验证通过 {summary['validated_strategy_count']} 个策略 → 趋势买入 {len(summary['buy_symbols'])} 只，反弹试错 {len(summary['rebound_trial_symbols'])} 只，防守候选 {len(summary['defense_symbols'])} 只")
                if not summary["buy_symbols"] and not summary["rebound_trial_symbols"] and not summary["defense_symbols"]:
                    print("今日暂无达到明确行动门槛的机会")
                print(f"ETF特有风险检查：{summary.get('etf_risk_counts', {})}")
                print("每只ETF的当前位置、行动建议和通俗原因见 analysis_report.md")
        for reason, count in reasons.items():
            print(f"  {reason}: {count}")
        print(f"输出目录：{output}")
        if refresh_summary and refresh_summary["failures"]:
            print(f"注意：有 {len(refresh_summary['failures'])} 项采集失败，详情见 summary.json")
        return 0 if (args.screen_only and not screened.selected.empty) or (not args.screen_only and not focus.empty) else 2
    except (ValueError, TypeError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f"筛选失败：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
