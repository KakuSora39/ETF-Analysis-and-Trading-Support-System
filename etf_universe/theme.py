"""Theme grouping used only to keep user-facing summaries representative."""

from __future__ import annotations

import re

import pandas as pd


THEME_KEYWORDS = (
    ("央企红利", ("港股通央企红利", "港股央企红利", "央企红利")),
    ("半导体芯片", ("半导体", "芯片", "集成电路")),
    ("航运港口", ("航运", "港口", "海运")),
    ("船舶", ("船舶",)),
    ("农业", ("农业", "粮食", "畜牧", "养殖", "种业")),
    ("证券", ("证券", "券商")),
    ("医药医疗", ("医药", "医疗", "创新药", "生物科技")),
    ("新能源", ("新能源", "光伏", "储能", "锂电", "电池")),
    ("人工智能", ("人工智能", "AI", "算力", "云计算")),
    ("机器人", ("机器人",)),
    ("军工", ("军工", "国防")),
    ("消费", ("消费", "食品饮料", "白酒")),
    ("黄金", ("黄金",)),
    ("有色金属", ("有色", "稀土", "铜", "铝")),
    ("能源化工", ("能源化工", "化工", "原油", "煤炭", "油气")),
)


def infer_theme(name: str, index_id: str = "") -> str:
    """Return a broad, explainable theme; exact index is the safe fallback."""
    text = f"{index_id or ''} {name or ''}"
    upper = text.upper()
    for theme, keywords in THEME_KEYWORDS:
        if any(keyword.upper() in upper for keyword in keywords):
            return theme
    if index_id:
        return re.sub(r"^(EM|CSI|SZ|SH):", "", str(index_id), flags=re.IGNORECASE).strip()
    normalized = re.sub(r"ETF|交易型开放式指数证券投资基金", "", str(name), flags=re.IGNORECASE)
    return normalized.strip() or "其他"


def themes_are_similar(
    left_name: str,
    left_index: str,
    right_name: str,
    right_index: str,
) -> bool:
    """Conservatively decide whether two highly correlated ETFs are substitutes.

    Exact tracking-index matches are always similar.  Otherwise both products
    must resolve to the same explainable broad theme.  Uncertain matches are
    deliberately retained instead of being removed by correlation alone.
    """
    left_index = str(left_index or "").strip().upper()
    right_index = str(right_index or "").strip().upper()
    if left_index and right_index and left_index == right_index:
        return True
    left = infer_theme(left_name, left_index)
    right = infer_theme(right_name, right_index)
    return bool(left and right and left != "其他" and left == right)


def deduplicate_summary_lanes(
    lanes: dict[str, pd.DataFrame],
    max_per_theme: int = 2,
) -> dict[str, pd.DataFrame]:
    """Limit repeated themes in summaries while leaving detailed lane sheets intact."""
    pieces = []
    for lane, frame in lanes.items():
        if frame.empty:
            continue
        part = frame[frame.action != "暂不参与"].copy()
        if part.empty:
            continue
        part["_summary_lane"] = lane
        support = f"validated_{lane}_support"
        part["_summary_support"] = pd.to_numeric(part.get(support, 0), errors="coerce").fillna(0)
        pieces.append(part)
    if not pieces:
        return {lane: frame.iloc[0:0].copy() for lane, frame in lanes.items()}

    combined = pd.concat(pieces, ignore_index=True, sort=False)
    combined["summary_theme"] = combined.apply(
        lambda row: infer_theme(row.get("name", ""), row.get("index_id", "")), axis=1,
    )
    action_priority = {
        "可以关注买入": 0, "高优先级反弹候选": 0, "稳健持有候选": 0,
        "较高优先级反弹候选": 1, "小仓反弹试错": 2, "防守候选": 2,
        "等待回调": 3, "反弹观察": 4, "继续观察": 4, "暂不追高": 5,
    }
    risk_priority = {"低": 0, "中": 1, "高": 2}
    combined["_summary_action"] = combined.action.map(action_priority).fillna(6)
    risk = combined["etf_risk_level"] if "etf_risk_level" in combined else pd.Series("", index=combined.index)
    amount = combined["avg_amount"] if "avg_amount" in combined else pd.Series(0, index=combined.index)
    aum = combined["aum_yuan"] if "aum_yuan" in combined else pd.Series(0, index=combined.index)
    combined["_summary_risk"] = risk.map(risk_priority).fillna(1)
    combined["_summary_amount"] = pd.to_numeric(amount, errors="coerce").fillna(0)
    combined["_summary_aum"] = pd.to_numeric(aum, errors="coerce").fillna(0)
    combined = combined.sort_values(
        ["_summary_action", "_summary_support", "_summary_risk", "_summary_amount", "_summary_aum", "symbol"],
        ascending=[True, False, True, False, False, True],
    )
    combined = combined.drop_duplicates("symbol", keep="first")
    combined = combined[combined.groupby("summary_theme").cumcount() < max_per_theme]
    helper = [column for column in combined if column.startswith("_summary_")]
    return {
        lane: combined[combined._summary_lane == lane].drop(columns=helper, errors="ignore").reset_index(drop=True)
        for lane in lanes
    }


def build_theme_summary(lanes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Describe each opportunity as one theme, one representative and alternatives."""
    rows = []
    lane_labels = {"trend": "趋势", "rebound": "反弹", "defense": "防守"}
    for lane, frame in lanes.items():
        if frame.empty:
            continue
        active = frame[frame.action != "暂不参与"].copy()
        if active.empty:
            continue
        if "summary_theme" not in active:
            active["summary_theme"] = active.apply(
                lambda row: infer_theme(row.get("name", ""), row.get("index_id", "")), axis=1,
            )
        for theme, group in active.groupby("summary_theme", sort=False):
            representative = group.iloc[0]
            alternatives = [
                f"{row.symbol} {row.get('name', '')}".strip()
                for _, row in group.iloc[1:].iterrows()
            ]
            rows.append({
                "opportunity_lane": lane, "opportunity_lane_label": lane_labels.get(lane, lane),
                "theme": theme, "representative_symbol": str(representative.symbol),
                "representative_name": representative.get("name", ""),
                "representative_action": representative.get("action", ""),
                "alternative_etfs": "；".join(alternatives) if alternatives else "无",
            })
    return pd.DataFrame(rows)
