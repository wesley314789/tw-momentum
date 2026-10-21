"""上市／上櫃個股距各自 52 週最高收盤價下跌至少 20% 的每日比例。

每檔股票等權計數；不是加權指數或櫃買指數自己的回檔幅度。
使用與 high_low_breadth 相同的 252 筆完整歷史、除權息校正和跳價排除。
"""

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    import high_low_breadth
    import ma200_breadth
    import market_risk_alert
except ModuleNotFoundError:
    from scripts import high_low_breadth, ma200_breadth, market_risk_alert


ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data" / "drawdown20_breadth.csv"
LATEST_PATH = ROOT / "docs" / "data" / "latest.json"
COLS = ["date", "market", "count", "eligible", "pct"]
MARKETS = ("上市", "上櫃")
WINDOW = high_low_breadth.WINDOW
THRESHOLD = 0.80
KEEP_DAYS = 120


def load() -> pd.DataFrame:
    if PATH.exists():
        return pd.read_csv(PATH, dtype={"date": str})[COLS]
    return pd.DataFrame(columns=COLS)


def calculate(history: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    h, unmatched = market_risk_alert.price_history(history, actions, True)
    h = h[h.market.isin(MARKETS)].copy()
    unmatched = unmatched.loc[h.index]
    g = h.groupby("code", sort=False)
    high = g.price.transform(lambda s: s.rolling(WINDOW, min_periods=WINDOW).max())
    jumps = g.price.pct_change(fill_method=None).abs().gt(.105).fillna(False) | unmatched
    clean = jumps.groupby(h.code, sort=False).transform(
        lambda s: s.rolling(WINDOW, min_periods=WINDOW).sum().eq(0))
    sessions = h.date.map({d: i for i, d in enumerate(sorted(h.date.unique()))})
    prior = sessions.groupby(h.code, sort=False).shift(WINDOW)
    eligible = (high.notna() & h.price.gt(0) & clean & prior.notna() &
                (sessions - prior < 280))
    h["eligible"] = eligible
    h["down20"] = eligible & h.price.le(high * THRESHOLD)
    out = h.groupby(["date", "market"]).agg(
        count=("down20", "sum"), eligible=("eligible", "sum")).reset_index()
    out = out[out.eligible.gt(0)].copy()
    out["pct"] = (out["count"] / out.eligible * 100).round(2)
    return out[COLS].sort_values(["date", "market"]).reset_index(drop=True)


def save_series(fresh: pd.DataFrame, previous: pd.DataFrame,
                path: Path, backfill: bool) -> pd.DataFrame:
    latest = str(fresh.date.max())
    if backfill:
        combined = fresh
    else:
        additions = fresh[~fresh.date.isin(previous.date) | fresh.date.eq(latest)]
        combined = pd.concat([previous, additions], ignore_index=True)
        combined = combined.drop_duplicates(["date", "market"], keep="last")
    keep = set(sorted(combined.date.unique())[-KEEP_DAYS:])
    combined = combined[combined.date.isin(keep)].sort_values(
        ["date", "market"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False, encoding="utf-8", float_format="%.2f")
    return combined


def update(history: pd.DataFrame, backfill: bool = False) -> tuple[pd.DataFrame, str]:
    if backfill:
        history = ma200_breadth.history_for_backfill(history)
    latest = str(history.date.max())
    actions, meta = market_risk_alert.load_actions(ROOT / "data", latest, True)
    fresh = calculate(history, actions)
    if set(fresh.loc[fresh.date.eq(latest), "market"]) != set(MARKETS):
        raise ValueError("最新交易日的上市與上櫃資料未同時達到 52 週有效樣本條件")
    return save_series(fresh, load(), PATH, backfill), meta.get("end", "")


def publish_latest(series: pd.DataFrame, actions_through: str) -> None:
    """只更新本指標欄位，供手動回補後立刻預覽；每日流程仍由 update_data 寫入。"""
    payload = json.loads(LATEST_PATH.read_text(encoding="utf-8"))
    latest = str(series.date.max())
    if payload.get("trade_date") != latest:
        raise ValueError(f"latest.json 日期 {payload.get('trade_date')} 與廣度 {latest} 不一致")
    payload["drawdown20_breadth"] = series.tail(KEEP_DAYS * len(MARKETS)).to_dict("records")
    payload["drawdown20_actions_through"] = actions_through
    LATEST_PATH.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False),
                           encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backfill", action="store_true", help="用本機加長歷史回補曲線")
    parser.add_argument("--publish", action="store_true", help="同步寫入網站的 latest.json")
    args = parser.parse_args()
    history = pd.read_csv(ROOT / "data" / "history.csv.gz", dtype={"code": str})
    series, through = update(history, backfill=args.backfill)
    if args.publish:
        publish_latest(series, through)
    last = series[series.date.eq(series.date.max())]
    print(json.dumps({"date": str(series.date.max()),
                      "markets": last.set_index("market")[["count", "eligible", "pct"]]
                      .to_dict("index"), "days": int(series.date.nunique()),
                      "actions_through": through}, ensure_ascii=True, allow_nan=False))
