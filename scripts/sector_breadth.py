"""用現有日線與 Pivot 計算盤後產業動能；可傳入官方產業或已核對題材對照。"""

from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import breakout
except ModuleNotFoundError:
    from scripts import breakout


MIN_SECTOR_STOCKS = 3
RS_DAYS = breakout.RS_DAYS
PIVOT_DAYS = breakout.CONFIG.pivot_days
MIN_BREAKOUT_VOLUME_RATIO = breakout.CONFIG.min_breakout_volume_ratio


def stock_at(group: pd.DataFrame, end: int, index_at) -> dict | None:
    """只讀 end 及之前的 K 棒；資料不足或無效時不納入分母。"""
    tail = group.iloc[max(0, end - PIVOT_DAYS):end + 1]
    if len(tail) < max(PIVOT_DAYS + 1, 51, RS_DAYS + 1):
        return None
    close = tail["close"].to_numpy(dtype=float)
    high = tail["high"].to_numpy(dtype=float)
    volume = tail["volume"].to_numpy(dtype=float)
    if (not np.isfinite(close).all() or not np.isfinite(high).all()
            or not np.isfinite(volume).all() or np.min(close) <= 0
            or np.min(high) <= 0 or np.min(volume) < 0):
        return None
    day = str(tail["date"].iloc[-1])
    base_day = str(tail["date"].iloc[-RS_DAYS - 1])
    ix_today, ix_base = index_at(day), index_at(base_day)
    if (ix_today is None or ix_base is None or not np.isfinite([ix_today, ix_base]).all()
            or ix_today <= 0 or ix_base <= 0):
        return None

    pivot = breakout.prior_pivot(high, PIVOT_DAYS)
    if pivot <= 0:
        return None
    price = close[-1]
    distance = (price / pivot - 1) * 100
    eps = 1e-9  # 101/100 在二進位浮點可得到 1.0000000000000009%
    ma20, ma50 = float(close[-20:].mean()), float(close[-50:].mean())
    rs20 = (price / close[-RS_DAYS - 1] - ix_today / ix_base) * 100
    prior_vol20 = breakout.previous_volume_average(volume)
    fresh = (close[-2] <= pivot < price <= pivot * 1.03
             and prior_vol20 > 0
             and volume[-1] / prior_vol20 >= MIN_BREAKOUT_VOLUME_RATIO)
    score = sum((price > ma20, price > ma50, ma20 > ma50, rs20 >= 5,
                 -5-eps <= distance <= 3+eps, fresh))
    if fresh:
        status = "FRESH_BREAKOUT"
    elif -1-eps <= distance <= 1+eps:
        status = "AT_PIVOT"
    elif -5-eps <= distance < -1-eps:
        status = "WATCH"
    elif distance > 3+eps:
        status = "EXTENDED"
    elif distance > 1:
        status = "ABOVE_PIVOT"
    else:
        status = "FAR_BELOW"
    return {
        "code": str(tail["code"].iloc[-1]),
        "name": str(tail["name"].iloc[-1]),
        "market": str(tail["market"].iloc[-1]),
        "close": round(float(price), 2),
        "strong_score": int(score),
        "strong": bool(score >= 4),
        "pivot_distance_pct": round(float(distance), 2),
        "pivot_status": status,
        "at_pivot_zone": bool(-5-eps <= distance <= 1+eps),
        "fresh_breakout": bool(fresh),
    }


def calculate(hist: pd.DataFrame, sectors: dict[str, str | None], index_at) -> dict:
    """今日與前一交易日各自計算 Strong%，相減得到百分點。"""
    if hist.empty:
        return {"date": None, "previous_date": None, "mapped_stocks": 0,
                "universe_stocks": 0, "valid_stocks": 0, "sectors": []}
    dates = sorted(hist["date"].unique())
    today, previous = dates[-1], dates[-2] if len(dates) >= 2 else None
    mapping = {str(k): v for k, v in sectors.items() if v}
    groups = defaultdict(lambda: {"today": [], "previous": []})
    for code, g in hist[hist["code"].isin(mapping)].sort_values("date").groupby("code", sort=False):
        g = g.reset_index(drop=True)
        for day, slot in ((today, "today"), (previous, "previous")):
            if day is None:
                continue
            loc = g.index[g["date"] == day]
            if len(loc):
                row = stock_at(g, int(loc[-1]), index_at)
                if row is not None:
                    groups[mapping[str(code)]][slot].append(row)

    result = []
    for sector, data in groups.items():
        current, prior = data["today"], data["previous"]
        if not current:
            continue
        total = len(current)
        strong_count = sum(r["strong"] for r in current)
        pivot_count = sum(r["at_pivot_zone"] for r in current)
        breakout_count = sum(r["fresh_breakout"] for r in current)
        strong_pct = strong_count / total * 100
        prev_pct = sum(r["strong"] for r in prior) / len(prior) * 100 if prior else None
        prev_by_code = {r["code"]: r for r in prior}
        members = []
        for row in current:
            yesterday = prev_by_code.get(row["code"])
            member = {k: row[k] for k in ("code", "name", "market", "close",
                                           "strong_score", "strong", "pivot_distance_pct",
                                           "pivot_status")}
            member["strong_change"] = (int(row["strong"]) - int(yesterday["strong"])
                                       if yesterday else None)
            members.append(member)
        rank = {"FRESH_BREAKOUT": 0, "AT_PIVOT": 1, "WATCH": 2,
                "EXTENDED": 3, "ABOVE_PIVOT": 4, "FAR_BELOW": 5}
        members.sort(key=lambda r: (rank[r["pivot_status"]], -r["strong_score"], r["code"]))
        result.append({
            "sector": sector, "total_stocks": total,
            "previous_total_stocks": len(prior),
            "small_sample": total < MIN_SECTOR_STOCKS,
            "strong_count": strong_count, "strong_pct": round(strong_pct, 2),
            "previous_strong_pct": round(prev_pct, 2) if prev_pct is not None else None,
            "delta_1d_pp": round(strong_pct - prev_pct, 2) if prev_pct is not None else None,
            "pivot_count": pivot_count, "pivot_pct": round(pivot_count / total * 100, 2),
            "breakout_count": breakout_count,
            "breakout_pct": round(breakout_count / total * 100, 2),
            "stocks": members,
        })
    result.sort(key=lambda r: (r["delta_1d_pp"] is None,
                               -(r["delta_1d_pp"] or 0), -r["strong_pct"], r["sector"]))
    return {"date": today, "previous_date": previous,
            "mapped_stocks": len(mapping),
            "universe_stocks": int((hist["date"] == today).sum()),
            "valid_stocks": sum(len(x["today"]) for x in groups.values()),
            "sectors": result}
