#!/usr/bin/env python3
"""Official TWSE/TPEx ex-right and ex-dividend reference prices for research.

The reference-price ratio is used for a *total-return approximation* in the
backtest. It is not a record of actual cash payments or a broker execution
price. Raw OHLC stays untouched in the market-data pipeline.
"""
import datetime as dt
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

import update_data as u

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data" / "corporate_actions.csv"
META = ROOT / "data" / "corporate_actions_meta.json"
TWSE_URL = "https://www.twse.com.tw/rwd/zh/exRight/TWT49U"
TPEX_URL = "https://www.tpex.org.tw/www/zh-tw/bulletin/exDailyQ"
COLUMNS = ["date", "code", "market", "prev_close", "ref_price", "kind"]


def _date(value: str) -> str:
    nums = [int(x) for x in re.findall(r"\d+", value)[:3]]
    if len(nums) != 3:
        raise ValueError(f"Bad official ex-date: {value!r}")
    y, m, d = nums
    if y < 1911:
        y += 1911
    return dt.date(y, m, d).isoformat()


def _number(value):
    return float(str(value).replace(",", ""))


def _request(session, method, url, **kwargs):
    for attempt in range(3):
        try:
            r = session.request(method, url, timeout=60, **kwargs)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def fetch_range(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Fetch official ex-date events; both exchanges accept year-sized ranges."""
    rows = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        session.verify = u.ca_bundle()
        for year in range(start.year, end.year + 1):
            lo = max(start, dt.date(year, 1, 1))
            hi = min(end, dt.date(year, 12, 31))
            tw = _request(session, "GET", TWSE_URL,
                          params={"response": "json", "startDate": lo.strftime("%Y%m%d"),
                                  "endDate": hi.strftime("%Y%m%d")})
            if tw.get("stat", "").lower() != "ok":
                raise ValueError(f"TWSE {year}: {tw.get('stat')}")
            for item in tw.get("data", []):
                rows.append((_date(item[0]), str(item[1]).strip(), "上市",
                             _number(item[3]), _number(item[4]), str(item[6]).strip()))

            tp = _request(session, "POST", TPEX_URL,
                          data={"startDate": lo.strftime("%Y/%m/%d"),
                                "endDate": hi.strftime("%Y/%m/%d")})
            if tp.get("stat", "").lower() != "ok" or not tp.get("tables"):
                raise ValueError(f"TPEx {year}: {tp.get('stat')}")
            table = tp["tables"][0]
            if len(table.get("data", [])) != int(table.get("totalCount", -1)):
                raise ValueError(f"TPEx {year}: incomplete response")
            for item in table["data"]:
                rows.append((_date(item[0]), str(item[1]).strip(), "上櫃",
                             _number(item[3]), _number(item[4]), str(item[8]).strip()))
            print(f"  {year}: TWSE {len(tw.get('data', []))}, TPEx {len(table['data'])}", flush=True)

    out = pd.DataFrame(rows, columns=COLUMNS)
    if out.empty:
        raise ValueError("Official ex-right APIs returned no events")
    out = out[(out.prev_close > 0) & (out.ref_price > 0)].copy()
    out["ratio"] = out.ref_price / out.prev_close
    if ((out.ratio < .05) | (out.ratio > 10)).any():
        raise ValueError("Official event contains implausible reference-price ratio")
    return out.sort_values(["date", "code", "market"]).drop_duplicates(
        ["date", "code", "market"], keep="last").reset_index(drop=True)


def ensure_actions(start: str, end: str, force: bool = False) -> pd.DataFrame:
    """Refresh the cache if its recorded range does not cover the backtest."""
    if not force and PATH.exists() and META.exists():
        meta = json.loads(META.read_text(encoding="utf-8"))
        if meta.get("start", "9") <= start and meta.get("end", "") >= end:
            return pd.read_csv(PATH, dtype={"date": str, "code": str, "market": str})
    print(f"Updating official corporate actions {start}..{end}", flush=True)
    out = fetch_range(dt.date.fromisoformat(start), dt.date.fromisoformat(end))
    PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PATH, index=False)
    META.write_text(json.dumps({"start": start, "end": end,
                                "source": [TWSE_URL, TPEX_URL],
                                "fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                "events": len(out)}, indent=2), encoding="utf-8")
    return out


def adjust_history(hist: pd.DataFrame, actions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Backward-adjust OHLC on official ex-dates; keep raw OHLC for market cap.

    A mismatch between the official preceding close and our last observed close
    means this factor cannot be safely applied (e.g. a trading halt). Such
    events are counted, then the residual price jump remains visible to the
    backtest's unknown-action guard.
    """
    h = hist.sort_values(["code", "date"]).drop_duplicates(["date", "code"], keep="last").copy()
    a = actions[actions["code"].isin(h["code"].unique())].copy()
    a = a.drop_duplicates(["date", "code", "market"], keep="last")
    h = h.merge(a[["date", "code", "market", "prev_close", "ratio"]],
                on=["date", "code", "market"], how="left", validate="one_to_one")
    prev = h.groupby("code", sort=False)["close"].shift(1)
    event = h["ratio"].notna()
    matched = event & prev.notna() & (abs(prev / h["prev_close"] - 1) <= .03)
    h["_ratio"] = h["ratio"].where(matched, 1.0)
    h["_factor"] = h.groupby("code", sort=False)["_ratio"].transform(
        lambda s: pd.Series(s.to_numpy()[::-1].cumprod()[::-1] / s.to_numpy(), index=s.index))
    for c in ("open", "high", "low", "close"):
        h["adj_" + c] = h[c] * h["_factor"]
    h["signal_close"] = h["adj_close"]
    residual = h.groupby("code", sort=False)["adj_close"].pct_change().abs() > .105
    h["_corp"] = residual.fillna(False) | (event & ~matched)
    audit = {"events_seen": int(event.sum()), "events_applied": int(matched.sum()),
             "events_mismatched": int((event & ~matched).sum()),
             "residual_jumps": int(residual.sum())}
    return h, audit


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    print(f"Saved {len(ensure_actions(a.start, a.end, a.force))} events")
