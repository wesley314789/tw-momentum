"""Daily percentage of Taiwan stocks above their own 200-day moving average.

This measures broad trend participation, separately from the momentum screen.
The price adjustment and eligibility rules are shared with the 60MA risk alert.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    import market_risk_alert
except ModuleNotFoundError:
    from scripts import market_risk_alert


ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data" / "ma200_breadth.csv"
COLS = ["date", "count", "eligible", "pct"]


def load() -> pd.DataFrame:
    if PATH.exists():
        return pd.read_csv(PATH, dtype={"date": str})[COLS]
    return pd.DataFrame(columns=COLS)


def calculate(history: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Only complete 200-session histories with a clean adjusted-price window count."""
    p = market_risk_alert.participation(history, actions, True, window=200,
                                        max_universe_days=280)
    p = p[p["eligible"] > 0].rename(columns={"above": "count", "pct200": "pct"})
    p = p[COLS].copy()
    p["pct"] = p["pct"].round(2)
    return p.reset_index(drop=True)


def update(history: pd.DataFrame, backfill: bool = False) -> tuple[pd.DataFrame, str]:
    """Append missing dates without rewriting older points from a shorter window."""
    if backfill:
        long_path = ROOT / "data" / "history_long.csv.gz"
        if long_path.exists():
            older = pd.read_csv(long_path, dtype={"code": str})
            history = pd.concat([older, history], ignore_index=True).drop_duplicates(
                ["date", "code"], keep="last")
            # Enough overlap for 200 prior sessions plus 240 comparison days.
            keep = sorted(history["date"].unique())[-440:]
            history = history[history["date"].isin(keep)]

    latest = str(history["date"].max())
    actions, meta = market_risk_alert.load_actions(ROOT / "data", latest, True)
    fresh = calculate(history, actions)
    if fresh.empty or str(fresh["date"].iloc[-1]) != latest:
        raise ValueError("No eligible 200MA breadth for latest trading date")

    previous = load()
    if backfill:
        combined = fresh
    else:
        # Today's 280-session snapshot has less warmup history for prior dates
        # than the snapshot available on those dates. Keep existing results;
        # refresh only the latest date, e.g. after a new corporate action.
        additions = fresh[~fresh["date"].isin(previous["date"]) |
                          fresh["date"].eq(latest)]
        combined = pd.concat([previous, additions], ignore_index=True).drop_duplicates(
            "date", keep="last").sort_values("date")
    combined = combined.tail(280).reset_index(drop=True)
    PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(PATH, index=False, encoding="utf-8", float_format="%.2f")
    return combined, meta.get("end", "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", action="store_true",
                        help="Use the local long history, if available, to seed past dates")
    args = parser.parse_args()
    history = pd.read_csv(ROOT / "data" / "history.csv.gz", dtype={"code": str})
    series, through = update(history, backfill=args.backfill)
    print(json.dumps({"date": series.date.iloc[-1],
                      "count": int(series["count"].iloc[-1]),
                      "eligible": int(series.eligible.iloc[-1]),
                      "pct": float(series.pct.iloc[-1]),
                      "days": len(series), "actions_through": through},
                     ensure_ascii=False))
