#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_gex.py — 驗證 GEX 位階到底有沒有預測力

結論(2026-08 於 279 個交易日上跑出來的)寫在 README。簡短版:除了「gamma 集中
在哪裡」這種事實陳述之外,位階對交易決策幾乎沒有幫助 —— 牆的彈回率跟隨機
位階一樣、flip 當觸發點沒有方向性、唯一顯著的波動預測已經被 IV 定價完。

保留成可重跑的腳本是因為當時的樣本全部落在同一個大多頭(指數 14 個月 +102%),
空頭環境下結論未必相同。之後資料累積更多可以再跑一次:

    python scripts/validate_gex.py

需要 scipy。會重抓選擇權資料算 IV, 跑一次約 5~10 分鐘。
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.txo_gex as g   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N_SHUFFLE = 300          # 隨機對照的重複次數
SHIFT = 0.30             # 對照組平移幅度(佔牆距比例)


def fetch_ohlc(start: dt.date, end: dt.date) -> pd.DataFrame:
    """台指期主力月每日高/低/收(現有 pipeline 只留結算價, 這裡要高低點)。"""
    rows, cur = [], start
    while cur <= end:
        nxt = min(cur + dt.timedelta(days=27), end)
        raw = g._post_csv(g.TAIFEX_FUT_URL, "TX", cur, nxt, index_col=False)
        if not raw.empty:
            d = raw[~raw["到期月份(週別)"].astype(str).str.contains("/")].copy()
            d["_date"] = pd.to_datetime(d["交易日期"], errors="coerce").dt.date
            d = d[d["交易時段"].astype(str).str.strip() == "一般"]
            for c in ("最高價", "最低價", "收盤價", "未沖銷契約數"):
                d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", ""),
                                     errors="coerce")
            for day, grp in d.groupby("_date"):
                m = grp.loc[grp["未沖銷契約數"].idxmax()]      # 主力月 = OI 最大
                rows.append({"date": day.isoformat(), "high": m["最高價"],
                             "low": m["最低價"], "close": m["收盤價"]})
        cur = nxt + dt.timedelta(days=1)
    return pd.DataFrame(rows).drop_duplicates("date")


def atm_iv(df: pd.DataFrame) -> pd.DataFrame:
    """每日最近到期(剩 1~7 天)的價平隱含波動率。"""
    rows = []
    for day, sub in df.groupby("_trade_date"):
        chain, F = g.build_chain(sub, day)
        if not chain or F is None:
            continue
        near = sorted({round(c["T"] * 365) for c in chain})
        near = [d for d in near if 1 <= d <= 7]
        if not near:
            continue
        band = [c for c in chain if round(c["T"] * 365) == near[0]]
        atm = min(band, key=lambda c: abs(c["K"] - F))
        rows.append({"date": day.isoformat(), "iv": atm["iv"]})
    return pd.DataFrame(rows)


def head(title):
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def main():
    h = pd.read_csv(ROOT / "data" / "gex_history.csv")
    h = h.sort_values("date").reset_index(drop=True)
    start = dt.date.fromisoformat(h.date.min())
    end = dt.date.fromisoformat(h.date.max())

    print("抓取台指期 OHLC…", file=sys.stderr)
    d = h.merge(fetch_ohlc(start, end), on="date", how="inner")
    for c in ("high", "low", "close"):
        d["n_" + c] = d[c].shift(-1)
    d["ret"] = (d.txf_settle.shift(-1) / d.txf_settle - 1) * 100
    d["today"] = (d.txf_settle / d.txf_settle.shift(1) - 1) * 100
    d = d.dropna(subset=["ret", "today", "n_close", "call_wall", "put_wall",
                         "micro_flip", "txf_settle"]).reset_index(drop=True)
    band = d.call_wall - d.put_wall
    print(f"樣本 {len(d)} 個交易日 ({d.date.min()} ~ {d.date.max()})")
    print(f"期間指數 {d.txf_settle.iloc[0]:.0f} -> {d.txf_settle.iloc[-1]:.0f}"
          f" ({(d.txf_settle.iloc[-1]/d.txf_settle.iloc[0]-1)*100:+.0f}%)")

    # ---------------- 1. 牆:碰到之後會不會被彈回 ----------------
    head("1. 牆能不能當支撐壓力?(碰到後退回牆內 = 成功)")

    def reject(cw, pw):
        tc, tp = d.n_high >= cw, d.n_low <= pw
        ok = (tc & (d.n_close < cw)).sum() + (tp & (d.n_close > pw)).sum()
        n = tc.sum() + tp.sum()
        return n, ok / n if n else np.nan

    def contain(cw, pw):
        return ((d.n_close >= pw) & (d.n_close <= cw)).mean()

    rng = np.random.default_rng(0)
    shifts = [rng.uniform(-SHIFT, SHIFT, len(d)) * band for _ in range(N_SHUFFLE)]
    n_t, r_act = reject(d.call_wall, d.put_wall)
    r_null = np.array([reject(d.call_wall + s, d.put_wall + s)[1] for s in shifts])
    c_act = contain(d.call_wall, d.put_wall)
    c_null = np.array([contain(d.call_wall + s, d.put_wall + s) for s in shifts])

    print(f"  觸及牆 {n_t} 次")
    print(f"  觸及後退回率  實際 {r_act*100:.1f}%  隨機位階 {r_null.mean()*100:.1f}%"
          f"  -> 第 {(r_null < r_act).mean()*100:.0f} 百分位")
    print(f"  隔日收在牆內  實際 {c_act*100:.1f}%  隨機位階 {c_null.mean()*100:.1f}%"
          f"  -> 第 {(c_null < c_act).mean()*100:.0f} 百分位")
    pen = ((d.n_high - d.call_wall) / band)[d.n_high >= d.call_wall]
    print(f"  觸及上牆後的穿透深度(佔牆距): 中位 {pen.median()*100:.0f}%"
          f" / 90分位 {pen.quantile(.9)*100:.0f}%   <- 真障礙不該這麼容易穿過")

    # ---------------- 2. Micro Flip 當觸發點 ----------------
    head("2. 跌破 Micro Flip 會加速嗎?方向可預測嗎?")
    d["above"] = d.txf_settle > d.micro_flip
    d["n_above"] = d.n_close > d.micro_flip
    d["r2"] = (d.close.shift(-2) / d.n_close - 1) * 100
    brk = d[d.above & ~d.n_above].dropna(subset=["r2"])
    keep = d[d.above & d.n_above].dropna(subset=["r2"])
    if len(brk) >= 5 and len(keep) >= 5:
        pv = stats.mannwhitneyu(brk.r2.abs(), keep.r2.abs(), alternative="greater").pvalue
        print(f"  跌破組 {len(brk)} 天 後續|報酬| 中位 {brk.r2.abs().median():.2f}%")
        print(f"  維持組 {len(keep)} 天 後續|報酬| 中位 {keep.r2.abs().median():.2f}%")
        print(f"  是否加速: p = {pv:.3f} {'顯著' if pv < 0.05 else '<- 不顯著'}")
        up = (brk.r2 > 0).mean()
        pb = stats.binomtest(int((brk.r2 > 0).sum()), len(brk), 0.5).pvalue
        print(f"  跌破後方向: 反彈 {up*100:.0f}% / 續跌 {(1-up)*100:.0f}%"
              f"  p = {pb:.3f} {'有方向性' if pb < 0.05 else '<- 隨機'}")

    # ---------------- 3. regime 對隔日波動 ----------------
    head("3. 負 gamma 環境是否預示較大波動?(控制波動延續)")
    pos, neg = d[d.regime == "positive"].ret.abs(), d[d.regime == "negative"].ret.abs()
    print(f"  positive n={len(pos):3d} 隔日|報酬|中位 {pos.median():.2f}%")
    print(f"  negative n={len(neg):3d} 隔日|報酬|中位 {neg.median():.2f}%")
    if len(pos) >= 5 and len(neg) >= 5:
        print(f"  未控制: p = {stats.mannwhitneyu(pos, neg, alternative='less').pvalue:.4f}")
    d["bucket"] = pd.qcut(d.today.abs(), 3, labels=["小", "中", "大"])
    for b in d.bucket.cat.categories:
        s = d[d.bucket == b]
        p_, n_ = s[s.regime == "positive"].ret.abs(), s[s.regime == "negative"].ret.abs()
        if len(p_) >= 5 and len(n_) >= 5:
            pv = stats.mannwhitneyu(p_, n_, alternative="less").pvalue
            print(f"  當日波動{b}: 正 {p_.median():.2f}%(n={len(p_)}) vs "
                  f"負 {n_.median():.2f}%(n={len(n_)})  p = {pv:.3f}")

    # ---------------- 4. 決定性測試:是否已被 IV 定價 ----------------
    head("4. 決定性測試:這個波動訊號是不是早就反映在 IV 裡?")
    print("重算每日 ATM IV…", file=sys.stderr)
    df = g.prepare(g.fetch_chunked(start, end))
    v = d.merge(atm_iv(df), on="date", how="inner").dropna(subset=["iv"])
    v["exp_move"] = v.iv / np.sqrt(252) * np.sqrt(2 / np.pi) * 100
    v["ratio"] = v.ret.abs() / v.exp_move

    for r in ("positive", "neutral", "negative"):
        s = v[v.regime == r]
        if len(s) >= 5:
            print(f"  {r:9} n={len(s):3d}  當日 ATM IV 中位 {s.iv.median()*100:5.1f}%"
                  f"   實現/隱含比 {s.ratio.median():.2f}")
    pi, ni = v[v.regime == "positive"].iv, v[v.regime == "negative"].iv
    if len(pi) >= 5 and len(ni) >= 5:
        pv = stats.mannwhitneyu(pi, ni, alternative="less").pvalue
        print(f"\n  IV 是否已反映 gamma 結構: p = {pv:.4f} "
              f"{'<- 已反映' if pv < 0.05 else '<- 未反映'}")
    a, b_ = v[v.regime == "negative"].ratio, v[v.regime == "positive"].ratio
    if len(a) >= 5 and len(b_) >= 5:
        pv = stats.mannwhitneyu(a, b_, alternative="greater").pvalue
        print(f"  扣掉 IV 後還有預測力嗎: p = {pv:.3f} "
              f"{'<- 有' if pv < 0.05 else '<- 沒有, 訊號已被定價(無經濟價值)'}")
    rho, pv = stats.spearmanr(v.gex_now, v.ratio)
    print(f"  gex_now vs 實現/隱含比: rho = {rho:+.3f}, p = {pv:.3f}")
    print(f"\n  (全樣本實現/隱含比中位 {v.ratio.median():.2f} —— 選擇權長期偏貴,"
          f" 但這是全市場現象, 與 GEX 無關)")

    head("提醒")
    print("  本腳本共跑約十個檢定, 沒有做多重比較校正 —— 出現一兩個 p<0.05")
    print("  是預期內的。判讀時請以效果量與對照組分布為主, 不要只看 p 值。")


if __name__ == "__main__":
    main()
