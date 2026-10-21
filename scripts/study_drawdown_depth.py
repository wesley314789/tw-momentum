#!/usr/bin/env python3
"""Sweep stock drawdown depths for an *early* Taiwan market breadth warning.

The breadth trigger is fixed before comparing depths: at least 35% of both
listed and OTC eligible stocks are that far below their own 252-session high.
The market is still early only when TAIEX is within 3% of its own 252-session
high. Outcome: TAIEX low in the next 20 sessions <= 95% of signal-day close.
All signals are known at t close; no trading execution is assumed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import drawdown20_breadth as existing
    import market_risk_alert
except ModuleNotFoundError:
    from scripts import drawdown20_breadth as existing
    from scripts import market_risk_alert


ROOT = Path(__file__).resolve().parent.parent
DEPTHS = (5, 10, 15, 20, 25, 30)
BREADTH_TRIGGER = 35
HORIZON = 20
MIN_LISTED, MIN_OTC = 700, 500


def load_inputs():
    data = ROOT / "data"
    cols = ["date", "code", "market", "close"]
    history = pd.concat([
        pd.read_csv(data / "history_long.csv.gz", usecols=cols, dtype={"code": str}),
        pd.read_csv(data / "history.csv.gz", usecols=cols, dtype={"code": str}),
    ], ignore_index=True).drop_duplicates(["date", "code"], keep="last")
    actions = pd.read_csv(data / "corporate_actions.csv", dtype={"code": str})
    index = pd.read_csv(data / "taiex_ohlc.csv", dtype={"date": str}).sort_values("date")
    index["index_dd252_pct"] = (index.close / index.close.rolling(252).max() - 1) * 100
    return history, actions, index


def stock_drawdown(history, actions):
    """Same eligible universe as the published 20%-drawdown breadth."""
    h, unmatched = market_risk_alert.price_history(history, actions, True)
    h = h[h.market.isin(existing.MARKETS)].copy()
    unmatched = unmatched.loc[h.index]
    g = h.groupby("code", sort=False)
    high = g.price.transform(lambda s: s.rolling(existing.WINDOW,
                                                   min_periods=existing.WINDOW).max())
    jumps = g.price.pct_change(fill_method=None).abs().gt(.105).fillna(False) | unmatched
    clean = jumps.groupby(h.code, sort=False).transform(
        lambda s: s.rolling(existing.WINDOW, min_periods=existing.WINDOW).sum().eq(0))
    session = h.date.map({d: i for i, d in enumerate(sorted(h.date.unique()))})
    prior = session.groupby(h.code, sort=False).shift(existing.WINDOW)
    eligible = (high.notna() & h.price.gt(0) & clean & prior.notna() &
                (session-prior < 280))
    h = h.loc[eligible, ["date", "code", "market", "price"]].copy()
    h["high"] = high[eligible]
    return h


def breadth_for_depth(h, depth):
    b = h.assign(below=h.price.le(h.high * (1-depth/100))).groupby(["date", "market"]).agg(
        count=("below", "sum"), eligible=("below", "size")).reset_index()
    b["pct"] = b["count"] / b.eligible * 100
    listed = b[b.market.eq("上市")].drop(columns="market").rename(columns={
        "count": "listed_count", "eligible": "listed_eligible", "pct": "listed_pct"})
    otc = b[b.market.eq("上櫃")].drop(columns="market").rename(columns={
        "count": "otc_count", "eligible": "otc_eligible", "pct": "otc_pct"})
    return listed.merge(otc, on="date", validate="one_to_one")


def add_outcomes(index):
    f = index.copy().reset_index(drop=True)
    for horizon in (HORIZON, 40):
        f[f"worst_next{horizon}_pct"] = [
            100 * (f.low.iloc[i+1:i+horizon+1].min() / f.close.iloc[i] - 1)
            if i+horizon < len(f) else np.nan for i in range(len(f))]
    f["close_next20_pct"] = (f.close.shift(-HORIZON) / f.close - 1) * 100
    # Keep an exactly 5% drop inclusive despite binary float rounding.
    f["down5"] = f.worst_next20_pct.le(-5 + 1e-10)
    return f


def first_crossings(active):
    """Only breadth's first crossing, not a later re-entry into the price band."""
    state = active.to_numpy(dtype=bool)
    rising = np.flatnonzero(state & ~np.r_[False, state[:-1]])
    selected, last = [], -HORIZON
    for i in rising:
        if i == 0:  # a signal already on at the start has unknown onset
            continue
        if i-last >= HORIZON:
            selected.append(int(i))
            last = int(i)
    return selected


def metrics(f, mask):
    s = f.loc[mask & f.worst_next20_pct.notna()]
    if s.empty:
        return {"days": 0}
    s40 = f.loc[mask & f.worst_next40_pct.notna()]
    return {"days": len(s), "down5": int(s.down5.sum()),
            "risk_pct": round(float(s.down5.mean()*100), 1),
            "days40": len(s40),
            "risk40_pct": round(float(s40.worst_next40_pct.le(-5 + 1e-10).mean()*100), 1)
                          if len(s40) else None,
            "median_close20_pct": round(float(s.close_next20_pct.median()), 2)}


def study(h, index):
    index = add_outcomes(index)
    results, event_rows, daily_rows = [], [], []
    for depth in DEPTHS:
        f = index.merge(breadth_for_depth(h, depth), on="date", validate="one_to_one")
        f = f.sort_values("date").reset_index(drop=True)
        f["valid"] = f.listed_eligible.ge(MIN_LISTED) & f.otc_eligible.ge(MIN_OTC)
        f["near3"] = f.index_dd252_pct.gt(-3)
        f["near5"] = f.index_dd252_pct.gt(-5)
        f["joint35"] = f.listed_pct.ge(BREADTH_TRIGGER) & f.otc_pct.ge(BREADTH_TRIGGER)
        f["rise5_both"] = (f.listed_pct.sub(f.listed_pct.shift(20)).ge(5) &
                            f.otc_pct.sub(f.otc_pct.shift(20)).ge(5))
        start, end = str(f.loc[f.valid, "date"].min()), str(f.date.max())
        baseline3 = metrics(f, f.valid & f.near3)
        baseline5 = metrics(f, f.valid & f.near5)
        condition3 = metrics(f, f.valid & f.near3 & f.joint35)
        condition5 = metrics(f, f.valid & f.near5 & f.joint35)
        # Compare each signal day with days at a similar index drawdown from
        # its own annual high; broad market price weakness is a confounder.
        near = f[f.valid & f.near3 & f.worst_next20_pct.notna()].copy()
        near["price_band"] = pd.cut(near.index_dd252_pct, [-3, -2, -1, .01])
        band_risk = near.groupby("price_band", observed=True).down5.mean()
        signalled = near[near.joint35]
        matched = (float(signalled.price_band.map(band_risk).astype(float).mean()*100)
                   if len(signalled) else None)
        # First crossings are determined from breadth only. The 20-session
        # cooldown prevents repeated signals from the same deterioration.
        crossings = first_crossings(f.joint35 & f.valid)
        complete = [i for i in crossings if pd.notna(f.worst_next20_pct.iloc[i])]
        early3 = [i for i in complete if bool(f.near3.iloc[i])]
        early5 = [i for i in complete if bool(f.near5.iloc[i])]
        rising_early = [i for i in first_crossings(f.rise5_both & f.valid)
                        if bool(f.near3.iloc[i]) and pd.notna(f.worst_next20_pct.iloc[i])]
        for i in complete:
            row = f.iloc[i]
            event_rows.append({"depth": depth, "date": row.date,
                               "listed_pct": round(float(row.listed_pct), 2),
                               "otc_pct": round(float(row.otc_pct), 2),
                               "index_dd252_pct": round(float(row.index_dd252_pct), 2),
                               "near3": bool(row.near3), "near5": bool(row.near5),
                               "worst_next20_pct": round(float(row.worst_next20_pct), 2),
                               "down5": bool(row.down5)})
        def event_stats(indices):
            s = f.iloc[indices]
            return {"events": len(s), "hits": int(s.down5.sum()) if len(s) else 0,
                    "dates": [str(x) for x in s.date]}
        results.append({"stock_drawdown_threshold_pct": depth,
                        "start": start, "index_end": end,
                        "baseline_near3": baseline3,
                        "baseline_near5": baseline5,
                        "signal_near3": condition3,
                        "signal_near5": condition5,
                        "price_band_matched_risk_pct": round(matched, 1)
                        if matched is not None else None,
                        "first_crossings_near3": event_stats(early3),
                        "first_crossings_near5": event_stats(early5),
                        "first_crossings_all": event_stats(complete),
                        "rapid_rise_near3": {
                            "days": metrics(f, f.valid & f.near3 & f.rise5_both),
                            "first_crossings": event_stats(rising_early)}})
        daily_rows.append(f[["date", "listed_pct", "otc_pct", "listed_eligible",
                             "otc_eligible", "valid", "near3", "near5", "joint35",
                             "worst_next20_pct", "worst_next40_pct", "down5"]]
                          .assign(depth=depth))
    return results, pd.DataFrame(event_rows), pd.concat(daily_rows, ignore_index=True)


def verify_20pct(h, actions_history, actions):
    """Guard against a research formula drifting from the live website."""
    actual = breadth_for_depth(h, 20)
    official = existing.calculate(actions_history, actions)
    joined = actual.merge(official[official.market.eq("上市")][["date", "count", "eligible"]],
                          on="date", suffixes=("", "_published"), validate="one_to_one")
    if not (joined.listed_count.eq(joined["count"]) &
            joined.listed_eligible.eq(joined["eligible"])).all():
        raise AssertionError("20% study results differ from the published listed series")
    joined = actual.merge(official[official.market.eq("上櫃")][["date", "count", "eligible"]],
                          on="date", suffixes=("", "_published"), validate="one_to_one")
    if not (joined.otc_count.eq(joined["count"]) &
            joined.otc_eligible.eq(joined["eligible"])).all():
        raise AssertionError("20% study results differ from the published OTC series")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="JSON summary; related CSVs use same stem")
    args = parser.parse_args()
    history, actions, index = load_inputs()
    h = stock_drawdown(history, actions)
    verify_20pct(h, history, actions)
    results, events, daily = study(h, index)
    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2,
                                       allow_nan=False), encoding="utf-8")
        events.to_csv(args.out.with_name(args.out.stem + "_events.csv"), index=False)
        daily.to_csv(args.out.with_name(args.out.stem + "_daily.csv"), index=False)
    print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
