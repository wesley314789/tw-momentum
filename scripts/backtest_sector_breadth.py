#!/usr/bin/env python3
"""比較動能新上榜原策略與「當日官方產業 Strong% 前五」附加濾網。

沿用 backtest_momentum 的校正價、隔日開盤、每檔一萬元、7% R 階梯停利與
交易成本。產業對照只有目前的官方快照，歷史年度結果有分類存活偏差，僅供研究。
"""

import argparse
import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import backtest_momentum as bt
    import corporate_actions as ca
    import industry_mapping
    import sector_breadth
    import update_data as u
except ModuleNotFoundError:
    from scripts import backtest_momentum as bt
    from scripts import corporate_actions as ca
    from scripts import industry_mapping, sector_breadth, update_data as u


def _index_values(dates, idx_dates, idx_close):
    """與 update_data.index_lookup 一樣，缺日只往前找最近的指數收盤。"""
    positions = np.searchsorted(idx_dates, dates, side="right") - 1
    values = np.full(len(dates), np.nan)
    good = positions >= 0
    values[good] = idx_close[positions[good]]
    return values


def historical_scores(hist: pd.DataFrame, sector_map: dict, idx: pd.DataFrame,
                      start: str, end: str) -> pd.DataFrame:
    """以每檔過去 61 根校正日 K 計算六項 Strong Score；沒有看隔日價格。"""
    p = sector_breadth.PIVOT_DAYS
    r = sector_breadth.RS_DAYS
    series = idx.dropna(subset=["close"]).drop_duplicates("date").sort_values("date")
    ix_dates = series["date"].to_numpy(dtype=str)
    ix_close = series["close"].to_numpy(dtype=float)
    chunks = []
    for code, group in hist.groupby("code", sort=False):
        if code not in sector_map:
            continue
        group = group.sort_values("date")
        if len(group) < p + 1:
            continue
        dates = group["date"].to_numpy(dtype=str)
        close = group["adj_close"].to_numpy(dtype=float)
        high = group["adj_high"].to_numpy(dtype=float)
        volume = group["volume"].to_numpy(dtype=float)
        invalid = (~np.isfinite(close) | ~np.isfinite(high) | ~np.isfinite(volume)
                   | (close <= 0) | (high <= 0) | (volume < 0)
                   | group["_corp"].to_numpy(dtype=bool))
        clean = pd.Series(invalid.astype(int)).rolling(p + 1, min_periods=p + 1).sum().to_numpy() == 0
        c = pd.Series(close)
        ma20 = c.rolling(20).mean().to_numpy()
        ma50 = c.rolling(50).mean().to_numpy()
        pivot = pd.Series(high).shift(1).rolling(p).max().to_numpy()
        prevvol20 = pd.Series(volume).shift(1).rolling(20).mean().to_numpy()
        base_close = np.full(len(close), np.nan)
        base_close[r:] = close[:-r]
        ix_today = _index_values(dates, ix_dates, ix_close)
        ix_base = np.full(len(close), np.nan)
        ix_base[r:] = _index_values(dates[:-r], ix_dates, ix_close)
        prior_close = np.full(len(close), np.nan)
        prior_close[1:] = close[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            rs20 = (close / base_close - ix_today / ix_base) * 100
            distance = (close / pivot - 1) * 100
            vol_ratio = volume / prevvol20
        fresh = ((prior_close <= pivot) & (close > pivot) & (close <= pivot * 1.03)
                 & (prevvol20 > 0)
                 & (vol_ratio >= sector_breadth.MIN_BREAKOUT_VOLUME_RATIO))
        eps = 1e-9
        score = ((close > ma20).astype(int) + (close > ma50).astype(int)
                 + (ma20 > ma50).astype(int) + (rs20 >= 5).astype(int)
                 + ((distance >= -5-eps) & (distance <= 3+eps)).astype(int)
                 + fresh.astype(int))
        valid = (clean & np.isfinite(rs20) & np.isfinite(pivot) & (pivot > 0)
                 & (ix_today > 0) & (ix_base > 0)
                 & (dates >= start) & (dates <= end))
        if valid.any():
            chunks.append(pd.DataFrame({"date": dates[valid], "code": code,
                                        "sector": sector_map[code],
                                        "strong": score[valid] >= 4}))
    return (pd.concat(chunks, ignore_index=True) if chunks else
            pd.DataFrame(columns=["date", "code", "sector", "strong"]))


def top_five(scores: pd.DataFrame) -> tuple[dict[str, set[str]], pd.DataFrame]:
    """當天 Strong% 前五；分母與網站同為有效樣本且每組至少三檔。"""
    if scores.empty:
        return {}, pd.DataFrame()
    counts = scores.groupby(["date", "sector"], as_index=False).agg(
        total=("strong", "size"), strong_count=("strong", "sum"))
    counts = counts[counts["total"] >= sector_breadth.MIN_SECTOR_STOCKS].copy()
    counts["strong_pct"] = counts["strong_count"] / counts["total"] * 100
    # 百分比相同時以產業名稱排序，固定取五組；不看隔日變化或價格。
    counts = counts.sort_values(["date", "strong_pct", "sector"],
                                ascending=[True, False, True], kind="stable")
    leaders = counts.groupby("date", sort=False).head(5)
    top = {day: set(g["sector"]) for day, g in leaders.groupby("date")}
    return top, leaders


def filtered_new_signals(members: dict[str, set[str]], dates: list[str],
                         initial_prev: set[str], sector_map: dict[str, str],
                         top: dict[str, set[str]]) -> tuple[dict[str, set[str]], dict]:
    """先求原始新上榜，再以當日產業篩選；防止舊榜股票因產業轉強而誤買。"""
    signals, total, mapped = {}, 0, 0
    prior = initial_prev
    for day in dates:
        today = members.get(day, set())
        new = today - prior
        leaders = top.get(day, set())
        signals[day] = {code for code in new if sector_map.get(code) in leaders}
        total += len(new)
        mapped += sum(code in sector_map for code in new)
        prior = today
    return signals, {"new": total, "mapped_new": mapped,
                     "selected": sum(map(len, signals.values()))}


def metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0, "excluded": 0, "open": 0, "pnl": 0., "avg": 0.,
                "win": 0., "pf": None, "cost": 0., "peak": 0, "peak_roi": None}
    excluded = int((trades["reason"] == "未明公司行為排除").sum())
    valid = trades[trades["reason"] != "未明公司行為排除"]
    if valid.empty:
        return {**metrics(pd.DataFrame()), "excluded": excluded}
    gp = valid.loc[valid.pnl > 0, "pnl"].sum()
    gl = -valid.loc[valid.pnl <= 0, "pnl"].sum()
    events = pd.concat([pd.Series(1, index=pd.to_datetime(valid.entry_date)),
                        pd.Series(-1, index=pd.to_datetime(valid.exit_date))])
    peak = int(events.groupby(level=0).sum().sort_index().cumsum().max())
    return {"n": len(valid), "excluded": excluded,
            "open": int((valid["reason"] == "仍持有").sum()),
            "pnl": float(valid.pnl.sum()), "avg": float(valid.ret.mean() * 100),
            "win": float((valid.pnl > 0).mean() * 100),
            "pf": float(gp / gl) if gl else None,
            "cost": float(valid.cost.sum()), "peak": peak,
            "peak_roi": float(valid.pnl.sum() / (peak * bt.POSITION) * 100) if peak else None}


def month_block_difference(baseline: pd.DataFrame, selected: pd.DataFrame,
                           repeats=5000, seed=7):
    """以入場月份為區塊，探索兩組平均每筆報酬差；保留同月行情共振。"""
    if baseline.empty or selected.empty:
        return None
    frames = []
    for trades in (baseline, selected):
        usable = trades[trades["reason"] != "未明公司行為排除"].copy()
        usable["month"] = usable["entry_date"].str[:7]
        frames.append(usable.groupby("month")["pnl"].agg(["sum", "count"]))
    months = sorted(set(frames[0].index) | set(frames[1].index))
    if len(months) < 3:
        return None
    base, filt = [f.reindex(months, fill_value=0) for f in frames]
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(months), size=(repeats, len(months)))
    bn = base["count"].to_numpy()[draw].sum(axis=1)
    fn = filt["count"].to_numpy()[draw].sum(axis=1)
    good = (bn > 0) & (fn > 0)
    if not good.any():
        return None
    diffs = (filt["sum"].to_numpy()[draw[good]].sum(axis=1) / fn[good]
             - base["sum"].to_numpy()[draw[good]].sum(axis=1) / bn[good])
    diffs = diffs / bt.POSITION * 100
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)),
            float((diffs <= 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default="_sector_backtest.txt")
    args = ap.parse_args()
    bt.R = 0.07
    bt.SMOOTH = False
    hist = bt.load_prices()
    digest = hashlib.sha256(ca.PATH.read_bytes()).hexdigest()[:10]
    bt.MEM_CACHE = bt.ROOT / "data" / f"_bt_members_{u.SCREEN_SIG}_ca1_{digest}.csv.gz"
    all_dates = sorted(hist["date"].unique())
    dates = [d for d in all_dates[max(200, u.BR_HIGH52_DAYS)-1:]
             if d >= args.start and (args.end is None or d <= args.end)]
    if len(dates) < 2:
        raise ValueError("歷史資料不足，無法回測")
    members = bt.membership(hist, u.load_shares(), dates)
    previous = max((d for d in all_dates if d < dates[0]), default=None)
    prior = (u.momentum_screen(hist[hist["date"] <= previous], u.load_shares(), idx=bt.IDX)
             if previous else pd.DataFrame())
    initial_prev = set(prior["code"]) if len(prior) else set()
    sector_map = industry_mapping.load()
    if len(sector_map) < 1000:
        raise ValueError("目前官方產業對照不足，不能跑歷史產業濾網")
    print(f"sector score {dates[0]}..{dates[-1]}", flush=True)
    scores = historical_scores(hist, sector_map, bt.IDX, dates[0], dates[-1])
    top, leaders = top_five(scores)
    signals, counts = filtered_new_signals(members, dates, initial_prev, sector_map, top)
    print(f"signals: {counts}, ranked days: {len(top)}", flush=True)
    px = bt.build_px(hist)
    baseline = bt.run(px, members, dates, initial_prev=initial_prev)
    selected = bt.run(px, members, dates, initial_prev=initial_prev, entry_signals=signals)
    trade_prefix = str(Path(args.out).with_suffix(""))
    baseline.to_csv(trade_prefix + "_baseline_trades.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(trade_prefix + "_filtered_trades.csv", index=False, encoding="utf-8-sig")

    with io.open(args.out, "w", encoding="utf-8") as out:
        out.write(f"期間 {dates[0]} ~ {dates[-1]}；1R=7%，每筆 10,000 元，隔日開盤進場。\n")
        index_at = u.index_lookup(bt.IDX)
        index_start, index_end = index_at(dates[0]), index_at(dates[-1])
        if index_start and index_end:
            out.write(f"同期加權指數收盤 {index_start:,.1f} → {index_end:,.1f}，"
                      f"價格變化 {(index_end / index_start - 1) * 100:+.2f}%（不含股息）。\n")
        out.write("手續費買賣各 0.1425%，每筆最低 20 元；賣出交易稅 0.3%。\n")
        out.write("原策略＝目前動能篩選首次上榜（含距52週最高收盤價25%內）。\n")
        out.write("新策略＝原策略當天新上榜，且所屬官方產業 Strong% 排名前五；當天盤後判斷。\n")
        out.write("產業名錄是目前快照回填歷史，歷史分類更動／退市股票可能產生存活偏差。\n")
        out.write("動能市值使用目前股數；校正價供技術訊號，原價供買賣／市值。\n")
        out.write("資料有未明公司行為者不計績效；最後仍持有者以期末收盤評價。\n")
        out.write(f"全期間原始新上榜 {counts['new']} 筆，現有產業對照 {counts['mapped_new']} 筆，"
                  f"通過前五 {counts['selected']} 筆；有產業排名 {len(top)}/{len(dates)} 天。\n")
        if not leaders.empty:
            out.write(f"前五產業有效檔數中位 {leaders['total'].median():.0f}，"
                      f"少於 10 檔的比例 {(leaders['total'] < 10).mean() * 100:.1f}%。\n")
        for label, trades in (("原動能", baseline), ("產業前五", selected)):
            m = metrics(trades)
            out.write(f"\n{label}: {m}\n")
            if not trades.empty:
                out.write(f"  只看已平倉: {metrics(trades[trades['reason'] != '仍持有'])}\n")
                for year, group in trades.groupby(trades["entry_date"].str[:4]):
                    out.write(f"  入場年 {year}: {metrics(group)}\n")
        interval = month_block_difference(baseline, selected)
        if interval:
            lo, hi, nonpositive = interval
            out.write(f"\n平均每筆報酬改善的月份區塊重抽樣 95% 區間: "
                      f"{lo:+.3f} ~ {hi:+.3f} 個百分點；差異≤0 的抽樣比例 "
                      f"{nonpositive * 100:.1f}%。此為探索性估計，非前瞻保證。\n")
        # 診斷：各年排名有效樣本數，避免把缺資料日當作弱勢。
        coverage = scores.groupby("date").size()
        for year, group in coverage.groupby(coverage.index.str[:4]):
            out.write(f"  {year} 產業有效股票每日中位數 {group.median():.0f}\n")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
