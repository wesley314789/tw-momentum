#!/usr/bin/env python3
"""End-of-day study of listed/OTC stocks >20% below their own 52-week high.

This is an event study, not a trading strategy. All signals are known at day t's
close; index outcomes use sessions t+1 through t+20. Linked warning days are
collapsed to first activations with a 20-session cooldown.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import drawdown20_breadth
except ModuleNotFoundError:
    from scripts import drawdown20_breadth


ROOT = Path(__file__).resolve().parent.parent
HORIZON = 20
MIN_LISTED = 700
MIN_OTC = 500


def load_frame() -> pd.DataFrame:
    data = ROOT / "data"
    cols = ["date", "code", "market", "close"]
    history = pd.concat([
        pd.read_csv(data / "history_long.csv.gz", usecols=cols, dtype={"code": str}),
        pd.read_csv(data / "history.csv.gz", usecols=cols, dtype={"code": str}),
    ], ignore_index=True).drop_duplicates(["date", "code"], keep="last")
    actions = pd.read_csv(data / "corporate_actions.csv", dtype={"code": str})
    breadth = drawdown20_breadth.calculate(history, actions)
    listed = breadth[breadth.market.eq("上市")].drop(columns="market").rename(columns={
        "count": "listed_count", "eligible": "listed_eligible", "pct": "listed_pct"})
    otc = breadth[breadth.market.eq("上櫃")].drop(columns="market").rename(columns={
        "count": "otc_count", "eligible": "otc_eligible", "pct": "otc_pct"})
    index = pd.read_csv(data / "taiex_ohlc.csv", dtype={"date": str}).sort_values("date")
    index["index_dd252_pct"] = (
        index.close / index.close.rolling(252, min_periods=252).max() - 1) * 100
    missing = sorted(set(breadth.date) - set(index.date))
    if missing and (len(missing) != 1 or missing[0] != str(breadth.date.max())):
        raise ValueError(f"TAIEX OHLC is missing historical dates: {missing[:5]}")
    f = (index.merge(listed, on="date", validate="one_to_one")
         .merge(otc, on="date", validate="one_to_one")
         .sort_values("date").reset_index(drop=True))
    f["valid"] = f.listed_eligible.ge(MIN_LISTED) & f.otc_eligible.ge(MIN_OTC)
    f["near_high"] = f.index_dd252_pct.gt(-5)
    f["listed_rise20_pp"] = f.listed_pct - f.listed_pct.shift(20)
    f["otc_rise20_pp"] = f.otc_pct - f.otc_pct.shift(20)
    f["both_35"] = f.listed_pct.ge(35) & f.otc_pct.ge(35)
    f["both_50"] = f.listed_pct.ge(50) & f.otc_pct.ge(50)
    f["both_rise10"] = f.listed_rise20_pp.ge(10) & f.otc_rise20_pp.ge(10)
    f["breadth_mean"] = (f.listed_pct + f.otc_pct) / 2
    for horizon in (5, HORIZON):
        f[f"close_{horizon}d_pct"] = (f.close.shift(-horizon) / f.close - 1) * 100
        f[f"worst_{horizon}d_pct"] = [
            float(f.low.iloc[i+1:i+horizon+1].min() / f.close.iloc[i] - 1) * 100
            if i+horizon < len(f) else np.nan for i in range(len(f))]
    f["down5_next20"] = f.worst_20d_pct.le(-5)
    return f


def describe_days(f: pd.DataFrame, mask: pd.Series) -> dict:
    sub = f[mask & f.valid & f.close_20d_pct.notna()]
    if sub.empty:
        return {"days": 0}
    return {
        "days": len(sub),
        "down5_next20_pct": round(float(sub.down5_next20.mean() * 100), 1),
        "median_worst20_pct": round(float(sub.worst_20d_pct.median()), 2),
        "median_close5_pct": round(float(sub.close_5d_pct.median()), 2),
        "median_close20_pct": round(float(sub.close_20d_pct.median()), 2),
    }


def first_activations(state: pd.Series, cooldown: int = HORIZON) -> list[int]:
    """Use only the start of a run and keep outcome windows apart."""
    active = state.to_numpy(dtype=bool)
    rising = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    selected, last = [], -cooldown
    for i in rising:
        if i-last >= cooldown:
            selected.append(int(i))
            last = int(i)
    return selected


def describe_events(f: pd.DataFrame, mask: pd.Series) -> tuple[dict, pd.DataFrame]:
    activations = first_activations(mask & f.valid)
    observed = [i for i in activations if i+HORIZON < len(f)]
    sub = f.iloc[observed]
    if sub.empty:
        return {"events": 0, "censored": len(activations)}, pd.DataFrame()
    rows = sub[["date", "listed_pct", "otc_pct", "index_dd252_pct",
                "listed_rise20_pp", "otc_rise20_pp", "worst_20d_pct",
                "close_20d_pct", "down5_next20"]].copy()
    return {
        "events": len(sub), "censored": len(activations)-len(observed),
        "down5_events": int(sub.down5_next20.sum()),
        "down5_next20_pct": round(float(sub.down5_next20.mean() * 100), 1),
        "median_worst20_pct": round(float(sub.worst_20d_pct.median()), 2),
        "median_close20_pct": round(float(sub.close_20d_pct.median()), 2),
        "dates": [str(x) for x in sub.date],
    }, rows


def study(f: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    masks = {
        "listed_ge35": f.listed_pct.ge(35),
        "otc_ge35": f.otc_pct.ge(35),
        "both_ge35": f.both_35,
        "both_ge50": f.both_50,
        "near_high_listed_ge35": f.near_high & f.listed_pct.ge(35),
        "near_high_otc_ge35": f.near_high & f.otc_pct.ge(35),
        "near_high_both_ge35": f.near_high & f.both_35,
        "near_high_both_ge50": f.near_high & f.both_50,
        "near_high_both_rise10pp_20d": f.near_high & f.both_rise10,
    }
    baseline = {
        "all": describe_days(f, pd.Series(True, index=f.index)),
        "index_near_high": describe_days(f, f.near_high),
    }
    results, events = {}, []
    for name, mask in masks.items():
        event_summary, event_rows = describe_events(f, mask)
        results[name] = {"days": describe_days(f, mask), "events": event_summary}
        if not event_rows.empty:
            event_rows.insert(0, "rule", name)
            events.append(event_rows)
    quantiles = []
    days = f[f.valid & f.close_20d_pct.notna()].copy()
    days["breadth_quartile"] = pd.qcut(days.breadth_mean, 4, labels=False,
                                         duplicates="drop")
    for q, sub in days.groupby("breadth_quartile"):
        quantiles.append({"quartile": int(q)+1,
                          "mean_breadth_range": [round(float(sub.breadth_mean.min()), 1),
                                                 round(float(sub.breadth_mean.max()), 1)],
                          **describe_days(f, f.index.isin(sub.index))})
    return {
        "start": str(days.date.min()), "last_signal_date": str(days.date.max()),
        "index_history_end": str(f.date.max()),
        "method": "TAIEX within 5% of its trailing 252-session high; close at signal date vs next 5/20 closes and next 20 session lows; no execution or costs",
        "baseline": baseline, "quartiles": quantiles, "rules": results,
        "latest": {"date": str(f.date.iloc[-1]),
                   "listed_pct": float(f.listed_pct.iloc[-1]),
                   "otc_pct": float(f.otc_pct.iloc[-1]),
                   "index_dd252_pct": round(float(f.index_dd252_pct.iloc[-1]), 2)},
    }, pd.concat(events, ignore_index=True) if events else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Optional JSON summary path")
    args = parser.parse_args()
    f = load_frame()
    summary, events = study(f)
    if args.out:
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                       allow_nan=False), encoding="utf-8")
        events.to_csv(args.out.with_name(args.out.stem + "_events.csv"), index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
