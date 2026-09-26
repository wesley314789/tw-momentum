#!/usr/bin/env python3
"""Compare fixed-percent risk with signal-day Wilder ATR risk on identical data."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_momentum as b


def stats(t):
    excluded = 0 if t.empty else int((t.reason == "未明公司行為排除").sum())
    keep = t if t.empty else t[t.reason != "未明公司行為排除"]
    if keep.empty:
        return {"trades": 0, "excluded": excluded, "skipped_atr": t.attrs.get("skipped_atr", 0),
                "net": 0., "gross": 0., "cost": 0., "win_pct": np.nan,
                "pf": np.nan, "risk_median_pct": np.nan, "hold_median": np.nan}
    loss = -keep.loc[keep.pnl < 0, "pnl"].sum()
    gain = keep.loc[keep.pnl > 0, "pnl"].sum()
    return {"trades": len(keep), "excluded": excluded, "skipped_atr": t.attrs.get("skipped_atr", 0),
            "net": keep.pnl.sum(), "gross": keep.gross.sum(), "cost": keep.cost.sum(),
            "win_pct": (keep.pnl > 0).mean() * 100, "pf": gain / loss if loss else np.nan,
            "risk_median_pct": keep.risk_pct.median() * 100,
            "hold_median": (pd.to_datetime(keep.exit_date) - pd.to_datetime(keep.entry_date)).dt.days.median()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2023-11-06")
    ap.add_argument("--end")
    ap.add_argument("--period", type=int, default=14)
    ap.add_argument("--atr", default="1.5,2,3")
    ap.add_argument("--fixed", default="0.07,0.10", help="Fixed-risk controls, as fractions")
    ap.add_argument("--out", default="_atr_study.md")
    args = ap.parse_args()
    multiples = [float(s) for s in args.atr.split(",")]
    fixed = [float(s) for s in args.fixed.split(",")]
    if args.period < 1 or any(not np.isfinite(x) or x <= 0 for x in multiples):
        ap.error("ATR period and multiples must be positive")
    if any(not np.isfinite(x) or not 0 < x < 1 for x in fixed):
        ap.error("Fixed-risk controls must be between 0 and 1")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hist = b.load_prices()
    digest = hashlib.sha256(b.ca.PATH.read_bytes()).hexdigest()[:10]
    b.MEM_CACHE = b.ROOT / "data" / f"_bt_members_{b.u.SCREEN_SIG}_ca1_{digest}.csv.gz"
    shares = b.u.load_shares()
    all_dates = sorted(hist.date.unique())
    dates = [d for d in all_dates[199:] if d >= args.start and (not args.end or d <= args.end)]
    if len(dates) < 2:
        ap.error("Need at least two eligible trading days")
    members = b.membership(hist, shares, dates)
    previous = max((d for d in all_dates if d < dates[0]), default=None)
    prior = b.u.momentum_screen(hist[hist.date <= previous], shares, idx=b.IDX) if previous else pd.DataFrame()
    initial_prev = set(prior.code) if len(prior) else set()
    px = b.build_px(hist, atr_period=args.period)
    b.SMOOTH = False
    models = [(f"固定 {x*100:g}%", x, None) for x in fixed] + [
        (f"ATR{args.period} × {x:g}", .07, x) for x in multiples]
    fees = [("原價最低20", dict(rate=.001425, minimum=20.)),
            ("六折最低1", dict(rate=.001425*.6, minimum=1.)),
            ("無交易成本", dict(with_fees=False))]
    rows, years = [], []
    with out.with_suffix(".txt").open("w", encoding="utf-8") as detail:
        for model_id, (model, fixed_r, multiple) in enumerate(models):
            b.R = fixed_r
            for fee_id, (fee, kw) in enumerate(fees):
                t = b.run(px, members, dates, initial_prev=initial_prev, atr_multiple=multiple, **kw)
                row = {"model": model, "fees": fee, **stats(t)}
                rows.append(row)
                if not t.empty:
                    b.report(detail, t, f"{model} | {fee}", dates)
                if fee_id == 0 and not t.empty:
                    t.to_csv(out.with_name(out.stem + f"_trades_{model_id}.csv"), index=False, encoding="utf-8-sig")
                    for year, group in t.groupby(t.exit_date.str[:4]):
                        years.append({"model": model, "year": year, **stats(group)})
                print(f"{model} | {fee}: {row['trades']} trades, net {row['net']:+,.0f}, "
                      f"risk median {row['risk_median_pct']:.2f}%, excluded {row['excluded']}", flush=True)
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(years).to_csv(out.with_name(out.stem + "_yearly.csv"), index=False, encoding="utf-8-sig")
    lines = ["# ATR 與固定百分比回測", "",
             f"期間：{dates[0]}～{dates[-1]}，{len(dates)} 個交易日。", "",
             f"ATR 使用 {args.period} 日 Wilder 平滑，首值為 {args.period} 筆有效真實區間的平均。"
             "只讀上榜當日及以前的調整後 OHLC；缺值或未明公司行為會重設暖機期。",
             "1R＝上榜日 ATR × 倍數，隔日開盤進場後固定該距離。初始停損為進場價−1R；"
             "達1R後保本，達2R後按原有整數R階梯停利。日內順序仍先檢查原停損，再更新當日最高R，"
             "新停損下一交易日生效；不是每日用最新ATR重設追蹤停損。", "",
             "每次投入10,000元，不按ATR調整部位；因此每筆承擔的金額風險不同。"
             "下列為可評價交易的損益加總（含期末持倉收盤估值），不是固定資金帳戶報酬率。"
             "公司行為校正仍為近似、股數仍用現值，無法評價交易排除；未建模滑價、漲跌停成交限制或零股整數股數。",
             "這是同一歷史樣本的參數比較，沒有獨立樣本外驗證，不能因某倍數較好就認定其可靠。", "",
             "## 全期比較", "",
             "|模型|成本|交易筆數|淨損益（元）|勝率|PF|初始停損中位|排除|ATR缺值跳過|",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for x in rows:
        lines.append(f"|{x['model']}|{x['fees']}|{x['trades']:,}|{x['net']:+,.0f}|"
                     f"{x['win_pct']:.1f}%|{x['pf']:.2f}|{x['risk_median_pct']:.2f}%|"
                     f"{x['excluded']}|{x['skipped_atr']}|")
    lines += ["", "## 逐年（原價手續費，最低20元；依出場年分組）", "",
              "2023及2026不是完整年度，跨年部位全筆歸入出場年。", "",
              "|模型|年份|交易筆數|淨損益（元）|勝率|PF|", "|---|---|---:|---:|---:|---:|"]
    for x in years:
        lines.append(f"|{x['model']}|{x['year']}|{x['trades']:,}|{x['net']:+,.0f}|{x['win_pct']:.1f}%|{x['pf']:.2f}|")
    lines += ["", "ATR 定義：[Fidelity](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/atr)。", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
