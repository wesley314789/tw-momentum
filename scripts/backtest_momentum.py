#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_momentum.py — 「新上榜就買、R 倍數移動停利」的回測

規則(依使用者指定):
  進場  每當有個股**新進入**動能篩選名單, 隔一個交易日以開盤價買進 POSITION 元
        的零股(可買零股所以不取整股)
  停損  -7%。1R = 7%
  保本  漲幅曾超過 +1R 之後, 停損上移到成本價
  移動  再往上每過一個整數 R, 停利就鎖在那個 R —— 使用者給的例子: 漲到 +25%
        (3.57R), 回檔到 3R = +21% 時出場

  兩句描述合起來只有一種一致的解釋:
      曾達最高 maxR < 1   -> 停損 -1R (-7%)
      1 <= maxR < 2       -> 停損 0R (成本價, 保本)
      maxR >= 2           -> 停損 floor(maxR) * R
  2R 那一階會從 0R 直接跳到 2R, 是這組規則本身的性質, 不是實作上的取捨。
  --smooth 可切換成 (floor(maxR)-1)*R 的平滑版, 用來確認結論不是卡在這個邊界。

出場價的保守處理: 同一天先檢查是否觸及停損(用當日最低), 觸及就以停損價出場;
沒觸及才用當日最高更新 maxR。這樣不會出現「同一天先創高把停利拉上去、再用新的
停利價出場」這種事後諸葛。跳空低於停損價時以開盤價出場, 不假設拿得到停損價。

用法:
    python scripts/backtest_momentum.py --start 2026-01-01
"""
import argparse
import hashlib
import io
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import update_data as u
import corporate_actions as ca

ROOT = Path(__file__).resolve().parent.parent
LONG_HIST = ROOT / "data" / "history_long.csv.gz"
IDX_LONG = ROOT / "data" / "index_long.csv.gz"
# 篩選條件是相對大盤的, 回測區間比流水線的 280 天長, 要用涵蓋得到的指數序列
IDX = (pd.concat([pd.read_csv(IDX_LONG), u.load_index_history()], ignore_index=True)
       .drop_duplicates("date", keep="last").sort_values("date")
       if IDX_LONG.exists() else u.load_index_history())

POSITION = 10_000.0     # 每檔買進金額(元)
R = 0.07                # 1R = 7%

# 台股交易成本。零股與整股費率相同, 但多數券商有每筆最低手續費 20 元 ——
# 一萬元的單子光手續費就佔 0.2%, 遠高於費率本身, 對這個策略影響很大。
FEE_RATE = 0.001425     # 手續費, 買賣各一次
FEE_MIN = 20.0          # 每筆最低手續費(元); 0 表示不收最低
TAX_RATE = 0.003        # 證交稅, 賣出時收

SMOOTH = False

# 官方除權息參考價可校正一般現金與股票股利；減資、面額變更等未涵蓋事件
# 仍以殘差跳價守門，並列出排除數量。
LIMIT = 0.105


# 整數 R 的邊界要容忍浮點誤差: 0.21/0.07 = 2.9999999999999996, 直接 floor 會
# 變成 2R, 剛好把使用者舉的「漲到 3R」那個例子算錯一階。
EPS = 1e-9


def stop_level(max_r: float, risk_r: float | None = None) -> float:
    """回傳當下的停損/停利價位, 以報酬率表示(-0.07 = -7%)。"""
    risk_r = R if risk_r is None else risk_r
    if max_r < 1 - EPS:
        return -risk_r
    if SMOOTH:
        return (math.floor(max_r + EPS) - 1) * risk_r
    return 0.0 if max_r < 2 - EPS else math.floor(max_r + EPS) * risk_r


def load_prices(refresh_actions: bool = False) -> pd.DataFrame:
    if LONG_HIST.exists():
        df = pd.concat([pd.read_csv(LONG_HIST, dtype={"code": str}),
                        u.load_history()], ignore_index=True)
    else:
        df = u.load_history()
    df = df.drop_duplicates(["date", "code"], keep="last").sort_values(["code", "date"])
    actions = ca.ensure_actions(str(df.date.min()), str(df.date.max()), refresh_actions)
    adjusted, audit = ca.adjust_history(df, actions)
    print(f"官方除權息: 匹配 {audit['events_applied']}/{audit['events_seen']} 筆, "
          f"不符 {audit['events_mismatched']} 筆, 校正後仍異常跳價 "
          f"{audit['residual_jumps']} 筆", flush=True)
    return adjusted


MEM_CACHE = ROOT / "data" / f"_bt_members_{u.SCREEN_SIG}_ca1.csv.gz"


def membership(hist: pd.DataFrame, shares: pd.DataFrame, dates: list) -> dict:
    """
    {date: set(code)} —— 每個交易日通過動能篩選的名單。

    算一次要十分鐘(逐日重跑 momentum_screen), 但 R、停利規則、手續費都不影響
    名單, 所以存起來重複用。快取涵蓋不到要求的區間時才重算。
    """
    out = {}
    if MEM_CACHE.exists():
        c = pd.read_csv(MEM_CACHE, dtype={"date": str, "code": str})
        g = c[c["date"].isin(dates)].groupby("date")["code"].apply(
            lambda s: {x for x in s.dropna() if x})
        out = {d: g[d] for d in g.index}
        if out:
            print(f"  用名單快取 ({len(out)}/{len(dates)} 天)", flush=True)
    missing = [d for d in dates if d not in out]
    for i, d in enumerate(missing, 1):
        picks = u.momentum_screen(hist[hist["date"] <= d], shares, idx=IDX)
        out[d] = set(picks["code"]) if len(picks) else set()
        if i % 20 == 0:
            print(f"  篩選 {i}/{len(missing)} ({d}: {len(out[d])} 檔)", flush=True)
    # Keep an empty marker for zero-pick days; otherwise every rerun
    # needlessly recalculates those dates.
    rows = [{"date": d, "code": c} for d, cs in out.items()
            for c in (cs if cs else {""})]
    pd.DataFrame(rows).to_csv(MEM_CACHE, index=False, compression="gzip")
    return out


def fees(amount: float, is_sell: bool, rate=None, minimum=None) -> float:
    rate = FEE_RATE if rate is None else rate
    minimum = FEE_MIN if minimum is None else minimum
    f = max(amount * rate, minimum) if minimum else amount * rate
    return f + (amount * TAX_RATE if is_sell else 0.0)


def wilder_atr(high, low, close, invalid, period=14):
    """Causal Wilder ATR, seeded by period valid true ranges.

    Missing/invalid bars and unexplained corporate actions reset the seed.
    The first valid bar in a segment uses high-low as its true range.
    """
    if period < 1:
        raise ValueError("ATR period must be positive")
    out = np.full(len(close), np.nan)
    seed, prior, value = [], None, None
    for i, (hi, lo, cl, bad) in enumerate(zip(high, low, close, invalid)):
        if bad or not np.isfinite([hi, lo, cl]).all() or lo <= 0 or hi < lo:
            seed, prior, value = [], None, None
            continue
        tr = hi - lo if prior is None else max(hi - lo, abs(hi - prior), abs(lo - prior))
        if value is None:
            seed.append(tr)
            if len(seed) == period:
                value = sum(seed) / period
        else:
            value = ((period - 1) * value + tr) / period
        if value is not None:
            out[i] = value
        prior = cl
    return out


def build_px(hist: pd.DataFrame, atr_period: int | None = None) -> dict:
    """
    把日線攤成每檔一組 numpy 陣列 + 日期索引。

    原本每跑一個情境就用 groupby + set_index 重建一次, 再用 .loc 逐日查 ——
    掃描多個 R 值與成本情境時要重建十幾次, 期間拉到三年後這一步就變成主要成本。
    名單、價格都跟情境無關, 建一次共用即可。
    """
    px = {}
    for code, g in hist.groupby("code", sort=False):
        g = g.sort_values("date")
        px[code] = {
            "i": {d: k for k, d in enumerate(g["date"].to_numpy())},
            "o": g["adj_open"].to_numpy(dtype=float),
            "h": g["adj_high"].to_numpy(dtype=float),
            "l": g["adj_low"].to_numpy(dtype=float),
            "c": g["adj_close"].to_numpy(dtype=float),
            "x": g["_corp"].to_numpy(dtype=bool),
            "name": g["name"].iloc[-1],
        }
        if atr_period is not None:
            p = px[code]
            p["atr"] = wilder_atr(p["h"], p["l"], p["c"], p["x"], atr_period)
    return px


def run(px: dict, members: dict, dates: list,
        rate=None, minimum=None, with_fees: bool = True,
        initial_prev: set | None = None, atr_multiple: float | None = None) -> pd.DataFrame:
    if atr_multiple is not None and (not np.isfinite(atr_multiple) or atr_multiple <= 0):
        raise ValueError("ATR multiple must be positive and finite")
    trades, open_pos = [], {}
    skipped_atr = 0

    for i, d in enumerate(dates):
        # --- 先處理已持有的部位 ---
        for code in list(open_pos):
            pos = open_pos[code]
            p = px[code]
            k = p["i"].get(d)
            if k is None:
                continue                      # 停牌, 續抱
            if p["x"][k]:  # 官方資料無法校正的跳價；績效無法可信估算
                pos.update(exit_date=d, exit_px=pos["last_px"], reason="未明公司行為排除")
                trades.append(pos)
                del open_pos[code]
                continue
            lo, hi, op, cl = p["l"][k], p["h"][k], p["o"][k], p["c"][k]
            # 開高低收偶爾有缺值(當日無成交等)。NaN 比較永遠是 False, 停損會被
            # 靜悄悄跳過, 而 max() 碰到 NaN 會把 max_r 整個汙染成 NaN, 之後的
            # 停利階梯就全錯。缺值就當這天沒資料, 續抱。
            if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(op)):
                continue
            stop_px = pos["entry"] * (1 + stop_level(pos["max_r"], pos["risk_pct"]))
            if lo <= stop_px:                 # 先判停損, 不讓當日新高先把停利拉上去
                exit_px = min(op, stop_px)    # 跳空就以開盤價出場
                reason = ("停損" if pos["max_r"] < 1 else
                          "保本" if pos["max_r"] < 2 else "移動停利")
                pos.update(exit_date=d, exit_px=exit_px, reason=reason)
                trades.append(pos)
                del open_pos[code]
                continue
            pos["max_r"] = max(pos["max_r"], (hi / pos["entry"] - 1) / pos["risk_pct"])
            if np.isfinite(cl):
                pos["last_px"] = cl

        # --- 今天新進榜的, 明天開盤買 ---
        if i + 1 >= len(dates):
            continue
        prev = members.get(dates[i - 1], set()) if i else (initial_prev or set())
        new = members.get(d, set()) - prev
        nxt = dates[i + 1]
        for code in sorted(new):
            if code in open_pos or code not in px:
                continue
            p = px[code]
            k = p["i"].get(nxt)
            if k is None:
                continue
            if p["x"][k]:
                continue
            entry = p["o"][k]
            if not np.isfinite(entry) or entry <= 0:   # 缺開盤價就買不進去
                continue
            risk_pct, atr_signal = R, np.nan
            if atr_multiple is not None:
                signal_k = p["i"].get(d)
                atr_signal = p["atr"][signal_k] if signal_k is not None else np.nan
                risk_pct = atr_multiple * atr_signal / entry
                if not np.isfinite(risk_pct) or not 0 < risk_pct < 1:
                    skipped_atr += 1
                    continue
            open_pos[code] = dict(code=code, name=p["name"],
                                  listed=d, entry_date=nxt, entry=float(entry),
                                  risk_pct=float(risk_pct), atr_signal=float(atr_signal),
                                  max_r=0.0, last_px=float(entry),
                                  exit_date=None, exit_px=None, reason=None)

    for code, pos in open_pos.items():        # 還沒出場的以最後一天收盤計
        pos.update(exit_date=dates[-1], exit_px=pos["last_px"], reason="仍持有")
        trades.append(pos)

    t = pd.DataFrame(trades)
    t.attrs["skipped_atr"] = skipped_atr
    if t.empty:
        return t
    t["shares"] = POSITION / t["entry"]
    t["gross"] = (t["exit_px"] - t["entry"]) * t["shares"]
    if with_fees:
        buy = t["entry"] * t["shares"]
        sell = t["exit_px"] * t["shares"]
        t["cost"] = [fees(b, False, rate, minimum) + fees(s, True, rate, minimum)
                     for b, s in zip(buy, sell)]
    else:
        t["cost"] = 0.0
    t["pnl"] = t["gross"] - t["cost"]
    t["ret"] = t["pnl"] / POSITION
    t["r_mult"] = t["ret"] / t["risk_pct"]
    result = t.sort_values("entry_date").reset_index(drop=True)
    result.attrs["skipped_atr"] = skipped_atr
    return result


def report(o, t: pd.DataFrame, label: str, dates: list):
    o.write(f"\n{'=' * 64}\n{label}  ({dates[0]} ~ {dates[-1]})\n{'=' * 64}\n")
    if t.empty:
        o.write("沒有任何交易\n")
        return
    dropped = t[t["reason"] == "未明公司行為排除"]
    t = t[t["reason"] != "未明公司行為排除"]
    if len(dropped):
        o.write(f"(另有 {len(dropped)} 筆遇到官方資料無法校正的跳價，"
                f"已排除；此績效只代表其餘交易)\n")
    if t.empty:
        o.write("沒有可評價的交易\n")
        return
    n = len(t)
    win = int((t["pnl"] > 0).sum())
    invested = n * POSITION
    o.write(f"交易筆數      {n}\n")
    o.write(f"累計投入      {invested:,.0f} 元 (每檔 {POSITION:,.0f})\n")
    o.write(f"總損益        {t['pnl'].sum():+,.0f} 元   ({t['pnl'].sum() / invested * 100:+.2f}% / 累計投入)\n")
    o.write(f"交易成本      {t['cost'].sum():,.0f} 元   (佔累計投入 {t['cost'].sum() / invested * 100:.2f}%)\n")
    o.write(f"勝率          {win}/{n} = {win / n * 100:.1f}%\n")
    o.write(f"平均每筆      {t['ret'].mean() * 100:+.2f}%  ({t['r_mult'].mean():+.3f}R)\n")
    up = t.loc[t["pnl"] > 0, "r_mult"]
    dn = t.loc[t["pnl"] <= 0, "r_mult"]
    o.write(f"賺的平均      {up.mean():+.2f}R ({len(up)} 筆)   賠的平均 {dn.mean():+.2f}R ({len(dn)} 筆)\n")
    gp = t.loc[t["pnl"] > 0, "pnl"].sum()
    gl = -t.loc[t["pnl"] <= 0, "pnl"].sum()
    o.write(f"獲利因子      {gp / gl:.2f}\n" if gl else "獲利因子      —\n")
    o.write(f"最好 / 最差   {t['r_mult'].max():+.2f}R / {t['r_mult'].min():+.2f}R\n")
    # 累計投入不是實際要準備的錢 —— 部位會滾動, 真正需要的是同時在倉的最大檔數
    ev = pd.concat([pd.Series(1, index=pd.to_datetime(t["entry_date"])),
                    pd.Series(-1, index=pd.to_datetime(t["exit_date"]))])
    conc = ev.groupby(level=0).sum().sort_index().cumsum()
    peak = int(conc.max())
    avg = float(conc.mean())
    cap = peak * POSITION
    o.write(f"同時在倉      最多 {peak} 檔, 平均 {avg:.0f} 檔\n")
    o.write(f"實際佔用本金  {cap:,.0f} 元 (最多同時在倉 × 每檔 {POSITION:,.0f})\n")
    o.write(f"對佔用本金    {t['pnl'].sum() / cap * 100:+.2f}%\n")
    hold = (pd.to_datetime(t["exit_date"]) - pd.to_datetime(t["entry_date"])).dt.days
    o.write(f"持有天數      中位數 {hold.median():.0f} 天, 平均 {hold.mean():.1f} 天\n")
    m = t.copy()
    m["月"] = pd.to_datetime(m["exit_date"]).dt.strftime("%Y-%m")
    o.write("\n逐月損益(以出場日計):\n")
    for mo, g in m.groupby("月"):
        o.write(f"  {mo}  {len(g):4d} 筆  {g['pnl'].sum():+9,.0f} 元  "
                f"平均 {g['r_mult'].mean():+.2f}R\n")
    o.write("\n出場原因:\n")
    for r, g in t.groupby("reason"):
        o.write(f"  {r:<6s} {len(g):4d} 筆  平均 {g['r_mult'].mean():+6.2f}R  "
                f"合計 {g['pnl'].sum():+9,.0f} 元\n")
    o.write("\n最賺的 8 筆:\n")
    for _, x in t.nlargest(8, "pnl").iterrows():
        o.write(f"  {x['code']} {str(x['name'])[:8]:<9s} {x['entry_date']} -> {x['exit_date']} "
                f"{x['r_mult']:+6.2f}R {x['pnl']:+8,.0f} 元  {x['reason']}\n")
    o.write("\n最賠的 5 筆:\n")
    for _, x in t.nsmallest(5, "pnl").iterrows():
        o.write(f"  {x['code']} {str(x['name'])[:8]:<9s} {x['entry_date']} -> {x['exit_date']} "
                f"{x['r_mult']:+6.2f}R {x['pnl']:+8,.0f} 元  {x['reason']}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default="_bt_out.txt")
    ap.add_argument("--r", default="0.07",
                    help="停損幅度, 可給多個做比較, 例如 --r 0.05,0.07,0.10,0.15")
    ap.add_argument("--smooth", action="store_true",
                    help="移動停利改用 (floor(maxR)-1)*R, 確認結論對規則邊界不敏感")
    ap.add_argument("--refresh-actions", action="store_true",
                    help="重新抓證交所與櫃買中心官方除權息參考價")
    args = ap.parse_args()

    global SMOOTH, R
    SMOOTH = args.smooth
    r_list = [float(x) for x in str(args.r).split(",")]

    hist = load_prices(args.refresh_actions)
    # A refreshed action series can change historical membership; never reuse
    # a cache built with earlier adjustment factors.
    global MEM_CACHE
    digest = hashlib.sha256(ca.PATH.read_bytes()).hexdigest()[:10]
    MEM_CACHE = ROOT / "data" / f"_bt_members_{u.SCREEN_SIG}_ca1_{digest}.csv.gz"
    shares = u.load_shares()
    all_dates = sorted(hist["date"].unique())
    usable = all_dates[199:]          # 前 200 個交易日拿來算 SMA200
    dates = [d for d in usable if d >= args.start and (not args.end or d <= args.end)]
    if not dates:
        print(f"沒有可回測的交易日。歷史 {all_dates[0]} ~ {all_dates[-1]}, "
              f"最早可篩選日 {usable[0] if usable else '—'}", file=sys.stderr)
        return 1
    print(f"回測 {dates[0]} ~ {dates[-1]} ({len(dates)} 個交易日)"
          f"{' [平滑版移動停利]' if SMOOTH else ''}", flush=True)

    mem = membership(hist, shares, dates)
    # The first requested date is not an automatic "new listing": compare
    # it with the last trading day before the requested research window.
    previous = max((d for d in all_dates if d < dates[0]), default=None)
    prior = (u.momentum_screen(hist[hist["date"] <= previous], shares, idx=IDX)
             if previous else pd.DataFrame())
    initial_prev = set(prior["code"]) if len(prior) else set()
    px = build_px(hist)          # 建一次, 所有情境共用
    fee_scen = [
        ("原價手續費 + 每筆最低 20 元", dict(rate=0.001425, minimum=20.0)),
        ("六折手續費 + 最低 1 元", dict(rate=0.001425 * 0.6, minimum=1.0)),
        ("完全不計交易成本", dict(with_fees=False)),
    ]
    with io.open(args.out, "w", encoding="utf-8") as o:
        o.write("資料品質: 用證交所/櫃買中心官方除權息參考價近似校正歷史 OHLC；"
                "技術訊號用校正價，市值仍用原始價 × 目前股數。\n")
        o.write(f"未能校正的異常跳價 {int(hist['_corp'].sum())} 筆；含事件的交易"
                "另行排除並計數，績效不含其未知結果。\n")
        summary, yearly = [], []
        for r in r_list:
            R = r
            for j, (label, kw) in enumerate(fee_scen):
                t = run(px, mem, dates, initial_prev=initial_prev, **kw)
                report(o, t, f"1R = {r*100:.0f}%  |  {label}", dates)
                if j == 0:                       # 逐年只看原價手續費那組
                    y = t[t["reason"] != "未明公司行為排除"].copy()
                    y["年"] = y["exit_date"].str[:4]
                    for yr, gg in y.groupby("年"):
                        gp = gg.loc[gg.pnl > 0, "pnl"].sum()
                        gl = -gg.loc[gg.pnl <= 0, "pnl"].sum()
                        yearly.append((r, yr, len(gg), gg.pnl.sum(),
                                       (gg.pnl > 0).mean() * 100,
                                       gp / gl if gl else float("nan")))
                    if r == r_list[0]:
                        t.to_csv("_bt_trades.csv", index=False, encoding="utf-8-sig")
                keep = t[t["reason"] != "未明公司行為排除"]
                gp = keep.loc[keep.pnl > 0, "pnl"].sum()
                gl = -keep.loc[keep.pnl <= 0, "pnl"].sum()
                ev = pd.concat([pd.Series(1, index=pd.to_datetime(keep["entry_date"])),
                                pd.Series(-1, index=pd.to_datetime(keep["exit_date"]))])
                peak = int(ev.groupby(level=0).sum().sort_index().cumsum().max())
                summary.append((r, label, len(keep), keep.pnl.sum(),
                                keep.pnl.sum() / (peak * POSITION) * 100,
                                (keep.pnl > 0).mean() * 100, gp / gl if gl else float("nan"),
                                (pd.to_datetime(keep["exit_date"]) -
                                 pd.to_datetime(keep["entry_date"])).dt.days.median()))
        o.write("\n\n" + "=" * 78 + "\n逐年(原價手續費 + 每筆最低 20 元)\n" + "=" * 78 + "\n")
        o.write(f"{'1R':>5s}{'年份':>7s}{'筆數':>7s}{'總損益':>11s}{'勝率':>8s}{'PF':>7s}\n")
        for r, yr, n, pnl, wr, pf in yearly:
            o.write(f"{r * 100:4.0f}%{yr:>7s}{n:7d}{pnl:+11,.0f}{wr:7.1f}%{pf:7.2f}\n")
        o.write("\n\n" + "=" * 78 + "\n總覽\n" + "=" * 78 + "\n")
        o.write(f"{'1R':>5s}  {'成本情境':<22s}{'筆數':>7s}{'總損益':>11s}"
                f"{'對佔用本金':>11s}{'勝率':>7s}{'PF':>6s}{'持有中位':>9s}\n")
        for r, lab, n, pnl, roc, wr, pf, hd in summary:
            o.write(f"{r * 100:4.0f}%  {lab:<22s}{n:7d}{pnl:+11,.0f}{roc:+10.2f}%"
                    f"{wr:6.1f}%{pf:6.2f}{hd:7.0f} 天\n")
    print(f"寫到 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
