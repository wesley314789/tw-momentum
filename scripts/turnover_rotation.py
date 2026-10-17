"""Daily technology/traditional trading-activity comparison.

Trading value measures where turnover occurs; it is not net buying flow.
Taiwan values come from exchange daily totals. US values approximate daily
turnover with Yahoo close * volume for the Nasdaq $500M+ universe.
"""

import datetime as dt

import numpy as np
import pandas as pd


TW_TECH = {"24", "25", "26", "27", "28", "29", "30", "31", "34", "36"}
TW_EXCLUDED = {"17", "20", "22"}  # finance, miscellaneous, biotechnology
US_TECH = {"Technology", "Telecommunications"}
US_TRADITIONAL = {
    "Consumer Discretionary", "Consumer Staples", "Industrials",
    "Basic Materials", "Energy", "Utilities", "Real Estate",
}


def tw_group(industry_code):
    code = str(industry_code).zfill(2)
    if code in TW_TECH:
        return "tech"
    if code in TW_EXCLUDED or code not in _TW_KNOWN:
        return "other"
    return "traditional"


# Keep the known-code set explicit, so a newly introduced exchange category
# does not silently become a traditional industry.
_TW_KNOWN = {
    "01", "02", "03", "04", "05", "06", "08", "09", "10", "11",
    "12", "14", "15", "16", "17", "18", "20", "21", "22", "23",
    "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38",
}


def us_group(sector):
    if sector in US_TECH:
        return "tech"
    if sector in US_TRADITIONAL:
        return "traditional"
    return "other"


def _row(date, sums, counts, valid, universe, basis):
    tech = float(sums.get("tech", 0))
    trad = float(sums.get("traditional", 0))
    if tech + trad <= 0 or universe <= 0:
        return None
    return {
        "date": date,
        "tech_value": round(tech, 2),
        "traditional_value": round(trad, 2),
        "other_value": round(float(sums.get("other", 0)), 2),
        "tech_count": int(counts.get("tech", 0)),
        "traditional_count": int(counts.get("traditional", 0)),
        "valid_count": int(valid),
        "universe_count": int(universe),
        "coverage_pct": round(valid / universe * 100, 1),
        "tech_share_pct": round(tech / (tech + trad) * 100, 2),
        "universe_basis": basis,
    }


def taiwan_series(history: pd.DataFrame, mapping: pd.DataFrame,
                  limit: int = 120) -> list[dict]:
    """Use all ordinary shares in history, including stocks outside scanners."""
    if history.empty or mapping.empty:
        return []
    codes = mapping[["code", "market", "industry_code"]].copy()
    codes["code"] = codes["code"].astype(str)
    codes = codes.drop_duplicates(["code", "market"], keep="last")
    days = sorted(history["date"].dropna().astype(str).unique())[-limit:]
    h = history.loc[history["date"].astype(str).isin(days),
                    ["date", "code", "market", "value"]].copy()
    h["code"] = h["code"].astype(str)
    h = h.merge(codes, on=["code", "market"], how="left", validate="many_to_one")
    h["group"] = h["industry_code"].map(tw_group)
    h["value"] = pd.to_numeric(h["value"], errors="coerce")
    out = []
    for date, g in h.groupby("date", sort=True):
        good = g[np.isfinite(g["value"]) & (g["value"] > 0)]
        if len(good) < 0.75 * len(g):
            continue
        sums = good.groupby("group")["value"].sum().to_dict()
        counts = good.groupby("group").size().to_dict()
        row = _row(str(date), sums, counts, len(good), len(g), "current_sector_map")
        if row:
            out.append(row)
    return out


def us_series(arrays: dict, universe: pd.DataFrame, calendar: list[int],
              limit: int = 120) -> list[dict]:
    """Use current Nasdaq sectors; older days are reconstructed current-universe views."""
    if universe.empty or not calendar or "sector" not in universe:
        return []
    days = np.asarray(calendar[-limit:], dtype=np.int64)
    positions = {int(day): i for i, day in enumerate(days)}
    totals = {g: np.zeros(len(days)) for g in ("tech", "traditional", "other")}
    counts = {g: np.zeros(len(days), dtype=int) for g in totals}
    valid = np.zeros(len(days), dtype=int)
    symbols = universe.drop_duplicates("yahoo")
    for sym, sector in zip(symbols["yahoo"], symbols["sector"]):
        bars = arrays.get(sym)
        if bars is None:
            continue
        dates, closes, volumes = bars
        group = us_group(sector)
        for day, close, volume in zip(dates, closes, volumes):
            pos = positions.get(int(day))
            if pos is None or not (np.isfinite(close) and np.isfinite(volume)
                                   and close > 0 and volume > 0):
                continue
            totals[group][pos] += float(close * volume)
            counts[group][pos] += 1
            valid[pos] += 1
    out = []
    for i, day in enumerate(days):
        if valid[i] < max(500, 0.75 * len(symbols)):
            continue
        row = _row(dt.date.fromordinal(int(day)).isoformat(),
                   {g: v[i] for g, v in totals.items()},
                   {g: v[i] for g, v in counts.items()},
                   valid[i], len(symbols),
                   "current_observation" if i == len(days)-1 else "reconstructed_current")
        if row:
            out.append(row)
    return out
