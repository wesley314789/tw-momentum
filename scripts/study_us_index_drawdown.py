#!/usr/bin/env python3
"""Exploratory US index breadth study using *current* ETF holdings.

SPY, QQQ and SOXQ proxy the S&P 500, Nasdaq-100 and PHLX SOX membership.
Historical breadth reconstructed from today's membership has survivorship bias.
Do not present these results as a point-in-time backtest or a trading signal.
"""

import argparse
import datetime as dt
import io
import json
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
LIVE_CSV = ROOT / "data" / "us_index_drawdown_breadth.csv"
LIVE_JSON = ROOT / "docs" / "data" / "us_index_drawdown_latest.json"
SPY_XLSX = ("https://www.ssga.com/us/en/individual/library-content/products/"
            "fund-data/etfs/us/holdings-daily-us-en-spy.xlsx")
INVESCO = ("https://dng-api.invesco.com/cache/v1/accounts/en_US/shareclasses/"
           "{}/holdings/fund?idType=cusip&productType=ETF")
YAHOO = "https://query2.finance.yahoo.com/v8/finance/chart/{}?range=5y&interval=1d"
INDICES = {"SP500": ("^GSPC", None),
           "NDX": ("^NDX", "46090E103"),
           "SOX": ("^SOX", "46138G615")}
DEPTHS = (5, 10, 15, 20, 25, 30)
BREADTH_TRIGGER = 35.0
WINDOW = 252
HORIZON = 20
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def get_bytes(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=35) as response:
                return response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def memberships() -> tuple[dict[str, list[str]], dict[str, str]]:
    spy_bytes = get_bytes(SPY_XLSX)
    raw = pd.read_excel(io.BytesIO(spy_bytes), header=4)
    raw = raw[pd.to_numeric(raw.Weight, errors="coerce").notna()]
    sp = sorted({s.replace(".", "-") for s in raw.Ticker.astype(str)
                 if re.fullmatch(r"[A-Z]+(?:\.[A-Z]+)?", s)})
    names, dates = {"SP500": sp}, {}
    # The file's third row is the stated holdings date, rather than download time.
    header = pd.read_excel(io.BytesIO(spy_bytes), header=None, nrows=3)
    dates["SP500"] = str(header.iloc[2, 1]).removeprefix("As of ")
    for index in ("NDX", "SOX"):
        payload = json.loads(get_bytes(INVESCO.format(INDICES[index][1])))
        names[index] = sorted({x["ticker"].replace(".", "-") for x in payload["holdings"]
                               if (x.get("securityTypeName") or "").startswith(
                                   ("Common Stock", "American Depository Receipt"))
                               and isinstance(x.get("ticker"), str)
                               and re.fullmatch(r"[A-Z]+(?:\.[A-Z]+)?", x["ticker"])})
        dates[index] = payload["effectiveDate"]
    for name, symbols in names.items():
        if len(symbols) < {"SP500": 450, "NDX": 90, "SOX": 25}[name]:
            raise ValueError(f"{name}: implausibly short holdings list ({len(symbols)})")
    return names, dates


def yahoo_frame(symbol: str, cache: Path, refresh: bool = False) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (symbol.replace("^", "index_") + ".csv.gz")
    if path.exists() and not refresh:
        return pd.read_csv(path, dtype={"date": str})
    payload = json.loads(get_bytes(YAHOO.format(urllib.parse.quote(symbol, safe="")), 4))
    result = payload["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    adjusted = result["indicators"].get("adjclose")
    adj = adjusted[0]["adjclose"] if adjusted else quote["close"]
    frame = pd.DataFrame({
        "date": pd.to_datetime(result["timestamp"], unit="s", utc=True)
                  .tz_convert("America/New_York").strftime("%Y-%m-%d"),
        "close": quote["close"], "low": quote["low"], "adjclose": adj,
    }).dropna(subset=["close", "adjclose"])
    ny = pd.Timestamp.now(tz="America/New_York")
    if ny.hour * 60 + ny.minute < 16 * 60 + 20:
        frame = frame[frame.date.ne(ny.strftime("%Y-%m-%d"))]
    frame.to_csv(path, index=False, compression="gzip")
    return frame


def stock_breadth(bars: dict[str, pd.DataFrame], members: list[str],
                  calendar: list[str]) -> pd.DataFrame:
    """Rolling own-stock 252-close high; today's adjusted close is included."""
    rows = []
    cal = pd.Index(calendar, name="date")
    for symbol in members:
        if symbol not in bars:
            continue
        b = bars[symbol].drop_duplicates("date", keep="last").set_index("date")
        close = pd.to_numeric(b.adjclose, errors="coerce").reindex(cal)
        # The calendar is the *index* trading calendar. Missing stock sessions
        # break eligibility rather than silently comparing stale closing prices.
        high = close.rolling(WINDOW, min_periods=WINDOW).max()
        valid = close.gt(0) & high.gt(0)
        for depth in DEPTHS:
            rows.append(pd.DataFrame({"date": calendar, "symbol": symbol,
                                      "depth": depth, "eligible": valid.to_numpy(),
                                      "below": (valid & close.le(high * (1-depth/100))).to_numpy()}))
    if not rows:
        raise ValueError("No stock price history")
    return pd.concat(rows, ignore_index=True).groupby(["date", "depth"], as_index=False).agg(
        count=("below", "sum"), eligible=("eligible", "sum"))


def first_crossings(active: pd.Series, cooldown: int = HORIZON) -> list[int]:
    a = active.fillna(False).to_numpy(dtype=bool)
    rising = np.flatnonzero(a & ~np.r_[False, a[:-1]])
    picked, last = [], -cooldown
    for i in rising:
        if i and i - last >= cooldown:
            picked.append(int(i))
            last = int(i)
    return picked


def study_index(index_bars: pd.DataFrame, breadth: pd.DataFrame,
                expected: int) -> list[dict]:
    idx = index_bars.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    idx["high252"] = idx.close.rolling(WINDOW, min_periods=WINDOW).max()
    idx["index_dd_pct"] = (idx.close / idx.high252 - 1) * 100
    idx["worst_next20_pct"] = [
        100 * (idx.low.iloc[i+1:i+HORIZON+1].min() / idx.close.iloc[i] - 1)
        if i + HORIZON < len(idx) else np.nan for i in range(len(idx))]
    out = []
    for depth in DEPTHS:
        frame = idx.merge(breadth[breadth.depth.eq(depth)], on="date", how="left")
        frame["pct"] = frame["count"] / frame["eligible"] * 100
        complete = (frame.eligible.ge(expected * 0.85) & frame.index_dd_pct.gt(-3) &
                    frame.worst_next20_pct.notna())
        signal = complete & frame.pct.ge(BREADTH_TRIGGER)
        base = frame.loc[complete].copy()
        active = frame.eligible.ge(expected * 0.85) & frame.pct.ge(BREADTH_TRIGGER)
        events = [i for i in first_crossings(active) if bool(complete.iloc[i])]
        hits = frame.worst_next20_pct.le(-5 + 1e-10)
        hits10 = frame.worst_next20_pct.le(-10 + 1e-10)
        matched = None
        if signal.any():
            base["band"] = pd.cut(base.index_dd_pct, [-3, -2, -1, .01])
            rates = base.groupby("band", observed=True).worst_next20_pct.apply(
                lambda x: x.le(-5 + 1e-10).mean())
            s = base[base.pct.ge(BREADTH_TRIGGER)]
            matched = float(s.band.map(rates).astype(float).mean() * 100)
        out.append({"depth": depth,
                    "eligible_dates": int(complete.sum()),
                    "first_eligible_date": str(frame.loc[complete, "date"].iloc[0]) if complete.any() else None,
                    "last_eligible_date": str(frame.loc[complete, "date"].iloc[-1]) if complete.any() else None,
                    "baseline_risk_pct": round(float(hits[complete].mean()*100), 1) if complete.any() else None,
                    "baseline_risk10_pct": round(float(hits10[complete].mean()*100), 1) if complete.any() else None,
                    "signal_days": int(signal.sum()),
                    "signal_risk_pct": round(float(hits[signal].mean()*100), 1) if signal.any() else None,
                    "signal_risk10_pct": round(float(hits10[signal].mean()*100), 1) if signal.any() else None,
                    "matched_risk_pct": round(matched, 1) if matched is not None else None,
                    "onset_events": len(events), "onset_hits": int(hits.iloc[events].sum()),
                    "onset_hits10": int(hits10.iloc[events].sum()),
                    "onset_dates": [str(frame.date.iloc[i]) for i in events],
                    "latest_pct": round(float(frame.pct.dropna().iloc[-1]), 1) if frame.pct.notna().any() else None,
                    "latest_eligible": int(frame.eligible.dropna().iloc[-1]) if frame.eligible.notna().any() else 0})
    return out


def write_daily(bars: dict[str, pd.DataFrame], members: dict[str, list[str]],
                holdings_dates: dict[str, str]) -> None:
    """Append one observed ETF-membership snapshot; never backfill past dates."""
    end_dates = {name: bars[index_symbol].date.iloc[-1]
                 for name, (index_symbol, _) in INDICES.items()}
    if len(set(end_dates.values())) != 1:
        raise RuntimeError(f"Index dates disagree: {end_dates}")
    date = next(iter(end_dates.values()))
    rows = []
    for name, (index_symbol, _) in INDICES.items():
        source_day = pd.to_datetime(holdings_dates[name]).date()
        if not 0 <= (dt.date.fromisoformat(date) - source_day).days <= 4:
            raise RuntimeError(f"{name}: holdings date {source_day} is stale for {date}")
        calendar = bars[index_symbol].date.tolist()
        breadth = stock_breadth(bars, members[name], calendar)
        last = breadth[breadth.depth.eq(20) & breadth.date.eq(date)].iloc[0]
        if last.eligible < len(members[name]) * 0.85:
            raise RuntimeError(f"{name}: only {last.eligible}/{len(members[name])} stocks eligible")
        rows.append({"date": date, "index": name, "count": int(last["count"]),
                     "eligible": int(last.eligible),
                     "pct": round(float(last["count"] / last.eligible * 100), 2),
                     "holdings": len(members[name]),
                     "holdings_asof": source_day.isoformat(),
                     "basis": "observed_etf_holdings"})
    current = pd.DataFrame(rows)
    if LIVE_CSV.exists():
        existing = pd.read_csv(LIVE_CSV, dtype={"date": str, "holdings_asof": str})
        current = pd.concat([existing[existing.date.ne(date)], current], ignore_index=True)
    current = current.drop_duplicates(["date", "index"], keep="last").sort_values(["date", "index"])
    LIVE_CSV.parent.mkdir(parents=True, exist_ok=True)
    current.to_csv(LIVE_CSV, index=False)
    # Keep the observed series intact. Truncating it would eventually create
    # a gap between the fixed reconstruction and newer daily snapshots.
    records = current.to_dict("records")
    LIVE_JSON.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(LIVE_JSON.read_text(encoding="utf-8")) if LIVE_JSON.exists() else {}
    LIVE_JSON.write_text(json.dumps({"updated": dt.datetime.now(dt.timezone.utc).isoformat(),
                                    "depth_pct": 20, "rows": records,
                                    "reconstructed_history": previous.get("reconstructed_history")},
                                   ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Saved observed US index breadth for {date}: "
          + ", ".join(f"{r['index']} {r['pct']:.2f}%" for r in rows), flush=True)


def seed_history(bars: dict[str, pd.DataFrame], members: dict[str, list[str]],
                 holdings_dates: dict[str, str], days: int = 252) -> None:
    """One-time illustrative history from a fixed membership snapshot.

    This is deliberately kept separate from observed daily snapshots. It must
    never be used as point-in-time history for event studies or alerts.
    """
    if not LIVE_JSON.exists():
        raise RuntimeError("Create the first observed daily snapshot before seeding history")
    payload = json.loads(LIVE_JSON.read_text(encoding="utf-8"))
    if payload.get("reconstructed_history"):
        raise RuntimeError("Reconstructed history already exists; refusing to change its membership anchor")
    observed = payload.get("rows") or []
    first_observed = min(r["date"] for r in observed)
    historical = []
    for name, (index_symbol, _) in INDICES.items():
        calendar = bars[index_symbol].date.tolist()
        breadth = stock_breadth(bars, members[name], calendar)
        frame = breadth[(breadth.depth.eq(20)) & (breadth.date.lt(first_observed))]
        frame = frame[frame.eligible.ge(len(members[name]) * 0.85)].tail(days)
        historical.extend({"date": r.date, "index": name,
                           "count": int(r["count"]), "eligible": int(r.eligible),
                           "pct": round(float(r["count"] / r.eligible * 100), 2)}
                          for _, r in frame.iterrows())
    common = set.intersection(*(set(r["date"] for r in historical if r["index"] == name)
                                for name in INDICES))
    common = set(sorted(common)[-days:])
    rows = sorted((r for r in historical if r["date"] in common),
                  key=lambda r: (r["date"], r["index"]))
    if len(common) < days * 0.9:
        raise RuntimeError(f"Too few common dates for reconstructed history: {len(common)}")
    payload["reconstructed_history"] = {
        "basis": "fixed_current_etf_holdings_reconstruction",
        "holdings_asof": {name: pd.to_datetime(value).date().isoformat()
                          for name, value in holdings_dates.items()},
        "first_observed": first_observed,
        "rows": rows,
    }
    LIVE_JSON.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Saved {len(common)} reconstructed dates before {first_observed}; "
          "observed rows unchanged", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="refetch cached Yahoo bars")
    ap.add_argument("--daily", action="store_true",
                    help="append today's observed ETF-membership breadth to the website")
    ap.add_argument("--seed-history", action="store_true",
                    help="one-time reconstructed chart history using current ETF holdings")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "reports" / "us_index_drawdown_study.json")
    args = ap.parse_args()
    cache = Path(tempfile.gettempdir()) / "tw-momentum-us-index-drawdown"
    if args.daily and LIVE_CSV.exists() and LIVE_JSON.exists():
        # The Taiwan schedule runs three times per day. Only the morning slot
        # normally sees a newly closed US session; avoid 529 redundant calls.
        latest_index = yahoo_frame("^GSPC", cache, refresh=True).date.iloc[-1]
        existing = pd.read_csv(LIVE_CSV, usecols=["date", "index"], dtype=str)
        if set(existing.loc[existing.date.eq(latest_index), "index"]) == set(INDICES):
            print(f"US index breadth already current through {latest_index}", flush=True)
            return 0
    members, dates = memberships()
    symbols = sorted(set().union(*map(set, members.values()), set(INDICES[k][0] for k in INDICES)))
    print(f"Holdings: {[(k, len(v)) for k, v in members.items()]}; unique symbols {len(symbols)}", flush=True)
    bars, errors = {}, {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(yahoo_frame, s, cache, args.refresh): s for s in symbols}
        for i, future in enumerate(as_completed(futures), 1):
            sym = futures[future]
            try:
                bars[sym] = future.result()
            except Exception as exc:
                errors[sym] = f"{type(exc).__name__}: {exc}"
            if i % 100 == 0:
                print(f"Fetched {i}/{len(symbols)}; errors={len(errors)}", flush=True)
    if any(INDICES[k][0] not in bars for k in INDICES):
        raise RuntimeError(f"Index price series missing: {errors}")
    if args.daily:
        write_daily(bars, members, dates)
        return 0
    if args.seed_history:
        seed_history(bars, members, dates)
        return 0
    results = {}
    for name, (index_symbol, _) in INDICES.items():
        index_bars = bars[index_symbol]
        calendar = index_bars.date.tolist()
        breadth = stock_breadth(bars, members[name], calendar)
        coverage = sum(s in bars for s in members[name]) / len(members[name])
        if coverage < 0.85:
            raise RuntimeError(f"{name}: Yahoo coverage {coverage:.1%} too low")
        results[name] = {"membership_date": dates[name], "holdings": len(members[name]),
                         "fetch_success_pct": round(coverage*100, 1),
                         "index_price_end": calendar[-1],
                         "study_start": calendar[WINDOW-1],
                         "study_end_with_20d_outcome": calendar[-HORIZON-1],
                         "thresholds": study_index(index_bars, breadth, len(members[name]))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"method": "reconstructed_current_etf_holdings",
                                    "bias": "Survivorship and membership look-ahead; exploratory only",
                                    "threshold_stock_drawdown_pct": list(DEPTHS),
                                    "breadth_share_trigger_pct": BREADTH_TRIGGER,
                                    "index_near_52w_high_pct": 3,
                                    "outcome": "next 20 sessions' low <= signal close -5%",
                                    "indexes": results, "missing_symbols": errors},
                                   indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Saved {args.out}; missing {len(errors)} symbols", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
