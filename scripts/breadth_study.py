#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
breadth_study.py — 市場廣度跌破門檻之後, 大盤會不會變差?

問題: 通過動能篩選的檔數從 100 以上掉到 100 以下, 接下來指數的表現是否較差。

方法上有三件事必須做對, 否則很容易得到看起來漂亮但沒有意義的結果:

1. **跟「無條件」比, 不是跟零比。** 2026 大盤漲了 57%, 任何一段隨機的
   20 天期望報酬都是正的。要問的是「穿越之後的報酬, 有沒有比隨便挑一天更差」,
   所以每個統計量都附上同期間所有交易日的基準線。

2. **事件會連在一起。** 廣度在門檻附近會來回穿越 —— 2026 年 7 月兩週內就穿了
   四次。把它們當成四個獨立樣本會嚴重高估顯著性。這裡用 COOLDOWN 個交易日的
   不應期, 同一段只算一次, 並且同時報告原始次數與去重後次數。

3. **門檻是絕對檔數, 會隨全市場檔數漂移。** 所以同時用「佔全市場的百分比」
   再跑一次當作穩健性檢查。

用法:
    python scripts/breadth_study.py --threshold 100 --horizons 5,10,20
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import update_data as u

ROOT = Path(__file__).resolve().parent.parent
MEM_CACHE = ROOT / "data" / f"_bt_members_{u.SCREEN_SIG}.csv.gz"
LONG_HIST = ROOT / "data" / "history_long.csv.gz"
IDX_LONG = ROOT / "data" / "index_long.csv.gz"
# 篩選條件是相對大盤的, 回溯到 2023 要用涵蓋得到的指數序列 —— 流水線的
# index.csv.gz 只有 280 天
IDX = pd.read_csv(IDX_LONG) if IDX_LONG.exists() else u.load_index_history()

COOLDOWN = 10          # 事件不應期(交易日), 避免同一段來回穿越被當成多個樣本


def build_breadth(hist: pd.DataFrame, shares: pd.DataFrame, dates: list) -> pd.DataFrame:
    """逐日算通過動能篩選的檔數。有快取就用快取。"""
    cached = {}
    if MEM_CACHE.exists():
        c = pd.read_csv(MEM_CACHE, dtype={"date": str, "code": str})
        cached = c.groupby("date")["code"].apply(set).to_dict()
    rows, new = [], {}
    for i, d in enumerate(dates, 1):
        if d in cached:
            codes = cached[d]
        else:
            picks = u.momentum_screen(hist[hist["date"] <= d], shares, idx=IDX)
            codes = set(picks["code"]) if len(picks) else set()
            new[d] = codes
        uni = int((hist["date"] == d).sum())
        rows.append({"date": d, "count": len(codes), "universe": uni,
                     "pct": len(codes) / uni * 100 if uni else 0.0})
        if i % 50 == 0:
            print(f"  廣度 {i}/{len(dates)} ({d}: {len(codes)} 檔)", flush=True)
    if new:
        allsets = {**cached, **new}
        pd.DataFrame([{"date": d, "code": c} for d, cs in allsets.items() for c in cs]) \
          .to_csv(MEM_CACHE, index=False, compression="gzip")
    return pd.DataFrame(rows)


def dedup(idx: list, cooldown: int = COOLDOWN) -> list:
    """同一段內來回穿越只留第一次。"""
    out = []
    for i in idx:
        if not out or i - out[-1] > cooldown:
            out.append(i)
    return out


def forward(close: np.ndarray, i: int, h: int):
    if i + h >= len(close):
        return np.nan
    return (close[i + h] / close[i] - 1) * 100


def study(o, b: pd.DataFrame, col: str, thr: float, horizons: list, label: str):
    b = b.dropna(subset=["close"]).reset_index(drop=True)
    close = b["close"].to_numpy()
    above = b[col].to_numpy() >= thr
    raw = [i for i in range(1, len(b)) if above[i - 1] and not above[i]]
    ev = dedup(raw)

    o.write(f"\n{'=' * 70}\n{label}(門檻 {thr})\n{'=' * 70}\n")
    o.write(f"樣本 {len(b)} 個交易日 {b.date.iloc[0]} ~ {b.date.iloc[-1]}\n")
    o.write(f"跌破門檻 原始 {len(raw)} 次 -> 去重後 {len(ev)} 次"
            f"(不應期 {COOLDOWN} 個交易日)\n\n")
    if len(ev) < 5:
        o.write("事件太少, 無法下任何結論。\n")
        return

    o.write(f"{'期間':>6s}{'事件後平均':>12s}{'中位數':>10s}{'基準線平均':>12s}"
            f"{'差':>9s}{'勝率':>8s}{'基準勝率':>10s}{'n':>5s}\n")
    o.write("-" * 74 + "\n")
    for h in horizons:
        ev_r = np.array([forward(close, i, h) for i in ev])
        ev_r = ev_r[~np.isnan(ev_r)]
        base = np.array([forward(close, i, h) for i in range(len(b))], dtype=float)
        base = base[~np.isnan(base)]
        if len(ev_r) < 5:
            continue
        # 隨機抽同樣多的日子, 看事件平均落在什麼位置(單尾: 事件後是否更差)
        rng = np.random.default_rng(0)
        draws = np.array([rng.choice(base, len(ev_r), replace=False).mean()
                          for _ in range(20000)])
        p = float((draws <= ev_r.mean()).mean())
        o.write(f"{h:5d} 日{ev_r.mean():11.2f}%{np.median(ev_r):9.2f}%"
                f"{base.mean():11.2f}%{ev_r.mean() - base.mean():+8.2f}"
                f"{(ev_r > 0).mean() * 100:7.0f}%{(base > 0).mean() * 100:9.0f}%"
                f"{len(ev_r):5d}\n")
        o.write(f"        隨機抽同樣多天數 20000 次, 事件平均落在第 {p * 100:.0f} 百分位"
                f"{'  <- 顯著偏差' if p < 0.05 else ''}\n")

    o.write("\n各次事件的後續表現:\n")
    o.write(f"{'日期':<12s}{'廣度':>6s}" + "".join(f"{h:>8d}日" for h in horizons) + "\n")
    for i in ev:
        o.write(f"{b.date.iloc[i]:<12s}{b[col].iloc[i]:6.0f}"
                + "".join(f"{forward(close, i, h):8.1f}%" if not np.isnan(forward(close, i, h))
                          else f"{'—':>9s}" for h in horizons) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=100)
    ap.add_argument("--horizons", default="5,10,20")
    ap.add_argument("--start", default=None)
    ap.add_argument("--out", default="_breadth_study.txt")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]

    hist = pd.read_csv(LONG_HIST, dtype={"code": str}) if LONG_HIST.exists() else u.load_history()
    idx = pd.read_csv(IDX_LONG) if IDX_LONG.exists() else u.load_index_history()
    shares = u.load_shares()

    all_dates = sorted(hist["date"].unique())
    dates = all_dates[199:]                      # 前 200 天拿來算 SMA200
    if args.start:
        dates = [d for d in dates if d >= args.start]
    print(f"可用 {len(dates)} 個交易日 {dates[0]} ~ {dates[-1]}", flush=True)

    b = build_breadth(hist, shares, dates).merge(idx[["date", "close"]], on="date", how="left")
    miss = b["close"].isna().sum()
    if miss:
        print(f"  指數缺 {miss} 天, 這些日子不納入", flush=True)

    with io.open(args.out, "w", encoding="utf-8") as o:
        study(o, b, "count", args.threshold, horizons, "廣度檔數跌破門檻")
        # 穩健性: 門檻改用佔全市場的百分比, 避免全市場檔數漂移的影響
        pct_thr = round(float(b.loc[b["count"] >= args.threshold, "pct"].min()), 2) \
            if (b["count"] >= args.threshold).any() else 5.0
        study(o, b, "pct", pct_thr, horizons, "穩健性檢查:改用佔全市場百分比")
    b.to_csv("_breadth_series.csv", index=False, encoding="utf-8-sig")
    print(f"寫到 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
