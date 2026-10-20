"""Taiwan 52-week closing-high / closing-low breadth.

A new high (low) means today's adjusted close strictly exceeds (falls below)
the prior 251 trading observations. Ties are not new records. The adjustment
and unexplained-jump guard match the existing market-risk breadth calculation.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    import ma200_breadth
    import market_risk_alert
except ModuleNotFoundError:
    from scripts import ma200_breadth, market_risk_alert


ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data" / "high_low_breadth.csv"
COLS = ["date", "new_high", "new_low", "eligible", "high_pct", "low_pct"]
WINDOW = 252


def load() -> pd.DataFrame:
    if PATH.exists():
        return pd.read_csv(PATH, dtype={"date": str})[COLS]
    return pd.DataFrame(columns=COLS)


def calculate(history: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    h, unmatched = market_risk_alert.price_history(history, actions, True)
    g = h.groupby("code", sort=False)
    prior_high = g.price.transform(
        lambda s: s.shift(1).rolling(WINDOW - 1, min_periods=WINDOW - 1).max())
    prior_low = g.price.transform(
        lambda s: s.shift(1).rolling(WINDOW - 1, min_periods=WINDOW - 1).min())
    jumps = g.price.pct_change().abs().gt(.105).fillna(False) | unmatched
    clean = jumps.groupby(h["code"], sort=False).transform(
        lambda s: s.rolling(WINDOW, min_periods=WINDOW).sum().eq(0))
    # An earlier price is needed to check the jump on the first day of the
    # 252-session window. Keep the observation span within 280 market days.
    sessions = h["date"].map({d: i for i, d in enumerate(sorted(h["date"].unique()))})
    before = sessions.groupby(h["code"], sort=False).shift(WINDOW)
    eligible = (prior_high.notna() & prior_low.notna() & h.price.gt(0) & clean &
                before.notna() & (sessions - before < 280))
    h["eligible"] = eligible
    h["new_high"] = eligible & h.price.gt(prior_high)
    h["new_low"] = eligible & h.price.lt(prior_low)
    out = h.groupby("date").agg(eligible=("eligible", "sum"),
                                new_high=("new_high", "sum"),
                                new_low=("new_low", "sum")).reset_index()
    out = out[out["eligible"] > 0].copy()
    out["high_pct"] = (out.new_high / out.eligible * 100).round(2)
    out["low_pct"] = (out.new_low / out.eligible * 100).round(2)
    return out[COLS].reset_index(drop=True)


def update(history: pd.DataFrame, backfill: bool = False) -> tuple[pd.DataFrame, str]:
    if backfill:
        history = ma200_breadth.history_for_backfill(history)
    latest = str(history["date"].max())
    actions, meta = market_risk_alert.load_actions(ROOT / "data", latest, True)
    fresh = calculate(history, actions)
    if fresh.empty or str(fresh["date"].iloc[-1]) != latest:
        raise ValueError("No eligible 52-week high/low breadth for latest trading date")
    return ma200_breadth.save_series(fresh, load(), PATH, backfill), meta.get("end", "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", action="store_true")
    args = parser.parse_args()
    history = pd.read_csv(ROOT / "data" / "history.csv.gz", dtype={"code": str})
    series, through = update(history, backfill=args.backfill)
    print(json.dumps({"date": series.date.iloc[-1], "new_high": int(series.new_high.iloc[-1]),
                      "new_low": int(series.new_low.iloc[-1]),
                      "eligible": int(series.eligible.iloc[-1]),
                      "days": len(series), "actions_through": through},
                     ensure_ascii=False, allow_nan=False))
