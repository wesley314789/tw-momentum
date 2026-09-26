#!/usr/bin/env python3
"""Fixed-horizon new-listing study with entry-open aligned TAIEX benchmark."""
import argparse
import datetime as dt
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import backtest_momentum as b

BENCH_PATH = b.ROOT / "data" / "taiex_ohlc.csv"
BENCH_META = b.ROOT / "data" / "taiex_ohlc_meta.json"
BENCH_URL = "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST"


def ensure_benchmark(start, end):
    cols = ["date", "open", "high", "low", "close"]
    old = pd.read_csv(BENCH_PATH) if BENCH_PATH.exists() else pd.DataFrame(columns=cols)
    meta = json.loads(BENCH_META.read_text(encoding="utf-8")) if BENCH_META.exists() else {}
    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        for month in pd.period_range(start[:7], end[:7], freq="M"):
            key = str(month)
            part = old[old.date.str.startswith(key)] if len(old) else old
            needed = min(end, month.end_time.date().isoformat())
            if len(part) and (meta.get(key, {}).get("through", "") >= needed or part.date.max() >= needed):
                continue
            for attempt in range(3):
                try:
                    resp = session.get(BENCH_URL, params={"response": "json", "date": key.replace("-", "") + "01"}, timeout=45)
                    resp.raise_for_status()
                    payload = resp.json()
                    if payload.get("stat", "").lower() != "ok" or payload.get("fields") != ["日期", "開盤指數", "最高指數", "最低指數", "收盤指數"]:
                        raise ValueError(f"Invalid TAIEX response: {key}")
                    rows = []
                    for row in payload["data"]:
                        year, mon, day = map(int, row[0].split("/"))
                        date = dt.date(year + 1911 if year < 1911 else year, mon, day).isoformat()
                        if not date.startswith(key):
                            raise ValueError("Benchmark API ignored requested month")
                        rows.append([date] + [float(str(x).replace(",", "")) for x in row[1:5]])
                    new = pd.DataFrame(rows, columns=cols)
                    if new.empty or not np.isfinite(new.iloc[:, 1:].to_numpy(dtype=float)).all():
                        raise ValueError("Empty/nonfinite benchmark response")
                    break
                except (requests.RequestException, ValueError, KeyError):
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
            old = pd.concat([old, new], ignore_index=True).drop_duplicates("date", keep="last").sort_values("date")
            BENCH_PATH.parent.mkdir(parents=True, exist_ok=True)
            old.to_csv(BENCH_PATH, index=False)
            now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
            meta[key] = {"through": min((now.date()-dt.timedelta(days=1)).isoformat(), month.end_time.date().isoformat()),
                         "fetched_utc": now.astimezone(dt.timezone.utc).isoformat(), "source": BENCH_URL}
            BENCH_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"TAIEX {key}: {len(new)} rows", flush=True)
            time.sleep(.5)
    return old


def evaluate(px, members, dates, bench, horizon, max_horizon=20, initial_prev=None, no_overlap=True):
    """Entry is signal+1 open, exit is signal+horizon close (entry counts as day 1).

    Reserve a scheduled holding period before inspecting future data so missing
    outcomes do not retroactively free a position for another entry.
    """
    if not 1 <= horizon <= max_horizon:
        raise ValueError("Require 1 <= horizon <= max_horizon")
    rows, audit, busy = [], Counter(), {}
    previous = initial_prev or set()
    for s, day in enumerate(dates):
        current = members.get(day, set())
        new = current - previous
        previous = current
        audit["new_events"] += len(new)
        if s + max_horizon >= len(dates):
            audit["right_censored"] += len(new)
            continue
        entry_day, exit_day = dates[s+1], dates[s+horizon]
        for code in sorted(new):
            if no_overlap and busy.get(code, "") >= entry_day:
                audit["overlap"] += 1
                continue
            p = px.get(code)
            ei = p["i"].get(entry_day) if p is not None else None
            if ei is None or not np.isfinite(p["o"][ei]) or p["o"][ei] <= 0:
                audit["missing_entry"] += 1
                continue
            busy[code] = exit_day
            xi = p["i"].get(exit_day)
            if xi is None or not np.isfinite(p["c"][xi]) or p["c"][xi] <= 0:
                audit["missing_exit"] += 1
                continue
            if p["x"][ei:xi+1].any():
                audit["unknown_action"] += 1
                continue
            if xi - ei + 1 != horizon or not np.isfinite(p["c"][ei:xi+1]).all():
                audit["missing_path"] += 1
                continue
            entry, exit = float(p["o"][ei]), float(p["c"][xi])
            gross = exit / entry - 1
            sell = b.POSITION * (1 + gross)
            cost = (b.fees(b.POSITION, False, .001425, 20.) + b.fees(sell, True, .001425, 20.)) / b.POSITION
            low_cost = (b.fees(b.POSITION, False, .001425*.6, 1.) + b.fees(sell, True, .001425*.6, 1.)) / b.POSITION
            market = float(bench.loc[exit_day, "close"] / bench.loc[entry_day, "open"] - 1)
            rows.append(dict(signal=day, code=code, entry_date=entry_day, exit_date=exit_day,
                             gross=gross, net=gross-cost, net_low=gross-low_cost, market=market,
                             excess=gross-cost-market, excess_low=gross-low_cost-market,
                             dividend=bool(p["ex"][ei+1:xi+1].any())))
    audit["valid"] = len(rows)
    return pd.DataFrame(rows), dict(audit)


def summarize(t):
    if t.empty:
        return {"n": 0, **{k: np.nan for k in ["gross", "net", "net_low", "market", "excess", "excess_low", "median_net", "win", "beat"]}}
    return {"n": len(t), **{k: float(t[k].mean()*100) for k in ["gross", "net", "net_low", "market", "excess", "excess_low"]},
            "median_net": float(t.net.median()*100), "win": float((t.net>0).mean()*100),
            "beat": float((t.excess>0).mean()*100)}


def cohort_interval(t, calendar, block=40, repeats=2000):
    """Circular moving-block bootstrap of equal-weight signal-day excess means.

    All calendar days are retained, including zero-event days. This avoids
    treating correlated stocks on one date as independent observations.
    """
    if t.empty:
        return (0, np.nan, np.nan, np.nan)
    daily = t.groupby("signal").excess.mean().reindex(calendar).to_numpy(dtype=float)
    valid = np.isfinite(daily)
    if not valid.any():
        return (0, np.nan, np.nan, np.nan)
    rng = np.random.default_rng(20260926)
    n = len(daily)
    starts = rng.integers(0, n, size=(repeats, int(np.ceil(n/block))))
    sample = ((starts[:,:,None] + np.arange(block)) % n).reshape(repeats, -1)[:,:n]
    counts = valid[sample].sum(axis=1)
    sums = np.nan_to_num(daily, nan=0.)[sample].sum(axis=1)
    means = sums[counts>0] / counts[counts>0]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (int(valid.sum()), float(daily[valid].mean()*100), float(lo*100), float(hi*100))


def study(args, bench):
    hist = b.load_prices()
    hist = hist[hist.date <= args.end]
    digest = hashlib.sha256(b.ca.PATH.read_bytes()).hexdigest()[:10]
    b.MEM_CACHE = b.ROOT / "data" / f"_bt_members_{b.u.SCREEN_SIG}_ca1_{digest}.csv.gz"
    all_dates = sorted(hist.date.unique())
    dates = [d for d in all_dates[199:] if args.start <= d <= args.end]
    if len(dates) <= 20:
        raise ValueError("Need more than 20 eligible trading days")
    bench = bench.set_index("date")
    missing = set(dates)-set(bench.index)
    if missing:
        raise ValueError(f"Benchmark missing trading dates: {sorted(missing)}")
    if (bench.loc[dates, ["open", "close"]] <= 0).any().any():
        raise ValueError("Nonpositive index prices")
    check = b.IDX.merge(bench.reset_index(), on="date", suffixes=("_old", "_official"))
    diffs = (check.close_old-check.close_official).abs()
    delta = float(diffs.max())
    # Legacy index files used %.4g. Accept only demonstrable rounding, not an
    # arbitrary wider tolerance that could hide a mismatched date or index.
    rounded = np.array([float(format(v, ".4g")) for v in check.close_official])
    explained = (diffs <= .1) | np.isclose(check.close_old, rounded, atol=.001, rtol=0)
    if not explained.all():
        raise ValueError(f"Unexplained index mismatch: {check.loc[~explained, 'date'].tolist()}")
    shares = b.u.load_shares()
    mem = b.membership(hist, shares, dates)
    prev_date = max((d for d in all_dates if d < dates[0]), default=None)
    prior = b.u.momentum_screen(hist[hist.date <= prev_date], shares, idx=b.IDX) if prev_date else pd.DataFrame()
    initial = set(prior.code) if len(prior) else set()
    px = b.build_px(hist)
    for code, g in hist.groupby("code", sort=False):
        px[code]["ex"] = g.sort_values("date")._ratio.ne(1.).to_numpy()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summaries, years, audits, cohorts, robustness = [], [], [], [], []
    for horizon in [5, 10, 20]:
        t, audit = evaluate(px, mem, dates, bench, horizon, initial_prev=initial)
        summaries.append({"horizon": horizon, **summarize(t)})
        audits.append({"horizon": horizon, **audit})
        n_days, mean, lo, hi = cohort_interval(t, dates[:-20])
        cohorts.append({"horizon": horizon, "days": n_days, "mean": mean, "lo": lo, "hi": hi})
        if not t.empty:
            t.to_csv(out.with_name(out.stem + f"_events_{horizon}.csv"), index=False)
            for year, g in t.groupby(t.entry_date.str[:4]):
                years.append({"horizon": horizon, "year": year, **summarize(g)})
            robustness.append({"horizon": horizon, "sample": "持有期無除權息", **summarize(t[~t.dividend])})
        all_events, _ = evaluate(px, mem, dates, bench, horizon, initial_prev=initial, no_overlap=False)
        robustness.append({"horizon": horizon, "sample": "全部事件含重疊", **summarize(all_events)})
        print(f"H={horizon}: {summaries[-1]}, cohort 95% interval [{lo:.3f}, {hi:.3f}]", flush=True)
    pd.DataFrame(summaries).to_csv(out.with_suffix(".csv"), index=False)
    pd.DataFrame(years).to_csv(out.with_name(out.stem + "_yearly.csv"), index=False)
    pd.DataFrame(audits).to_csv(out.with_name(out.stem + "_audit.csv"), index=False)
    lines = ["# 新上榜訊號：固定持有期間研究", "",
             f"行情區間：{dates[0]}～{dates[-1]}；{len(dates)}個交易日。"
             f"共同訊號截止日：{dates[-21]}（三組都必須有完整20日後續行情）。", "",
             "## 方法", "",
             "- 沿用校正後的動能名單。第一次研究日先與前一交易日比較，避免把原已上榜者全算成新訊號。",
             "- 上榜隔日開盤買進，進場當天算第1天，第5／10／20個交易日收盤賣出。沒有停損、保本或移動停利。",
             "- 每次投入10,000元，原價手續費0.1425%、每筆最低20元，賣出交易稅0.3%；另列六折最低1元。",
             "- 主表同一股票持有期間不重複加碼；資料缺漏／未明公司行為的持有期仍鎖住，不因事後排除而重新開放買入。",
             "- 個股和加權指數都從相同交易日開盤到相同交易日收盤；指數不扣成本，僅作不可直接交易的比較基準。",
             "- 個股報酬為除權息近似校正，加權為價格指數，股利口徑對個股有利。因此另列持有期沒有除權息的子樣本。",
             "- 未涵蓋完整持有期的近期訊號不計；缺失價格、停牌造成不完整路徑與未明公司行為另列排除數，並非視為零報酬。",
             "- 每筆等權重是訊號研究，不是固定本金帳戶回測。沒有資金上限、滑價、漲跌停成交或整數零股限制；股數沿用現值。",
             "- 同日股票和相鄰持有期不獨立：另以每個上榜日等權平均，40交易日區塊重抽2,000次估95%區間。區間是探索性，未作多重比較校正，不能當作因果或穩定獲利證明。", "",
             f"官方大盤收盤與既有資料最大差異：{delta:.4f}點，已確認均為舊檔四捨五入精度。"
             f"為隔離進場訊號問題，名單沿用原有篩選精度，後續大盤報酬採完整官方精度。名單指紋：{b.u.SCREEN_SIG}；除權息快取：{digest}。", "",
             "## 主表（原價手續費，同股不重疊）", "",
             "超額以百分點表示＝個股扣成本報酬−同期間大盤報酬。", "",
             "|持有日數|筆數|個股毛報酬|個股淨報酬|同期大盤|平均淨超額|淨報酬中位數|淨勝率|勝過大盤比例|",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in summaries:
        lines.append(f"|{x['horizon']}|{x['n']:,}|{x['gross']:+.2f}%|{x['net']:+.2f}%|{x['market']:+.2f}%|{x['excess']:+.2f}|{x['median_net']:+.2f}%|{x['win']:.1f}%|{x['beat']:.1f}%|")
    lines += ["", "## 逐年（以進場年分組）", "", "2023與2026非完整年度。跨年交易全部歸到進場年。", "",
              "|年度|持有日數|筆數|個股淨報酬|同期大盤|平均淨超額|", "|---|---:|---:|---:|---:|---:|"]
    for x in years:
        lines.append(f"|{x['year']}|{x['horizon']}|{x['n']:,}|{x['net']:+.2f}%|{x['market']:+.2f}%|{x['excess']:+.2f}|")
    lines += ["", "## 不把同日多檔當成獨立證據", "", "|持有日數|上榜日數|每上榜日等權淨超額|40日區塊95%區間|", "|---:|---:|---:|---:|"]
    for x in cohorts:
        lines.append(f"|{x['horizon']}|{x['days']}|{x['mean']:+.2f}|[{x['lo']:+.2f}, {x['hi']:+.2f}]|")
    lines += ["", "## 成本及樣本敏感度", "", "|持有日數|原價淨超額|六折最低1元淨超額|", "|---:|---:|---:|"]
    for x in summaries:
        lines.append(f"|{x['horizon']}|{x['excess']:+.2f}|{x['excess_low']:+.2f}|")
    lines += ["", "|樣本|持有日數|筆數|平均淨超額（原價）|", "|---|---:|---:|---:|"]
    for x in robustness:
        lines.append(f"|{x['sample']}|{x['horizon']}|{x['n']:,}|{x['excess']:+.2f}|")
    lines += ["", "## 排除與重複持倉稽核", "", "|日數|原始新訊號|近期未滿20日|同股仍持有|缺進場|缺出場|未明公司行為|缺中途行情|有效|", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in audits:
        lines.append("|"+"|".join(str(x.get(k,0)) for k in ["horizon","new_events","right_censored","overlap","missing_entry","missing_exit","unknown_action","missing_path","valid"])+"|")
    lines += ["", f"大盤資料來源：[證交所歷史指數]({BENCH_URL}?response=html)。", "",
              f"重跑：`python scripts/signal_entry_study.py --start {args.start} --end {args.end} --out _signal_entry_study.md`。需要本機加長歷史。", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2023-11-06")
    ap.add_argument("--end", default="2026-09-24")
    ap.add_argument("--out", default="_signal_entry_study.md")
    ap.add_argument("--benchmark-only", action="store_true")
    args = ap.parse_args()
    bench = ensure_benchmark(args.start, args.end)
    if args.benchmark_only:
        return
    study(args, bench)


if __name__ == "__main__":
    main()
