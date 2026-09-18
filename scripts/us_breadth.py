#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
us_breadth.py — 美股版動能篩選與市場廣度

條件與台股那組相同(scripts/update_data.py 的 momentum_screen):
    收盤 > SMA200、SMA10 > SMA20、市值 > MIN_MCAP、成交值 > MIN_TURNOVER、
    近 PERF_DAYS 個交易日漲幅 - 同期 S&P 500 漲幅 > MIN_EXCESS 個百分點

最後一條 2026-09-18 起從「漲幅 > 20%」改成相對大盤, 理由同台股:絕對門檻跟著
大盤走, 廣度會混進大量「大盤本身漲多少」的成分。

用法:
    python scripts/us_breadth.py --backfill   # 連同歷史廣度與名單一起重算(抓兩年)
    python scripts/us_breadth.py              # 每日更新(抓一年)

資料來源都是免費的:
  * 宇宙與市值 —— Nasdaq screener, 一次回傳全美約 7,000 檔
  * 每日 K 線 —— Yahoo chart API, 逐檔但一次給一整年

因為 Yahoo 一趟就給一年份, 回補與每日更新的成本相同, 所以不另外存價格歷史
(台股那邊要按月切片抓, 才需要累積檔案)。只有廣度的每日檔數會存下來。

⚠ Yahoo 這支是非官方 API, 沒有服務保證, 隨時可能改格式或擋人 —— 與台股用
證交所官方資料的性質完全不同。抓不到的個股會被跳過並計數, 失敗率過高時會警告。
"""
import argparse
import datetime as dt
import re
import json
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
BREADTH_PATH = ROOT / "data" / "us_breadth.csv"
# 每個交易日通過篩選的名單 —— 上榜天數與族群日增減都靠它

OUTPUT_PATH = ROOT / "docs" / "data" / "us_latest.json"

NASDAQ_URL = "https://api.nasdaq.com/api/screener/stocks"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{}"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

MIN_MCAP = 5e8        # 市值下限(美元)
MIN_TURNOVER = 5e6    # 成交值下限(美元)
MIN_EXCESS = 10.0     # 近一個月超額報酬下限(百分點, 個股漲幅 - S&P 500 漲幅)
BENCH = "^GSPC"       # 比較基準: S&P 500
PERF_DAYS = 21        # 「一個月」取 21 個交易日
SMA_LONG = 200
WORKERS = 4           # 併發數。Yahoo 沒有公開額度, 保守一點
KEEP_DAYS = 280


# Nasdaq 的 name 欄位帶著一長串證券類別樣板("Class A Common Stock"、"Ordinary
# Shares"、"American Depositary Shares"…), 對顯示和搜尋都是雜訊。原本直接切
# 前 40 字, 結果是把名字切在字中間 —— "Palantir Technologies Commo"、
# "Iovance Biotherapeutics, Inc. Common Sto" —— 網站上顯示難看, 拿去查新聞更
# 是查不到(精確片語比對不到半個殘缺字)。改成把樣板整段拿掉, 留完整公司名。
SEC_TYPE = re.compile(
    r"[,\s]*\b(class\s+[a-c]\b|series\s+[a-z]\b|"
    r"common\s+stock|common\s+shares?|ordinary\s+shares?|"
    r"subordinate\s+voting\s+shares?|"
    r"american\s+depositary\s+shares?|depositary\s+shares?|"
    r"perpetual\b|units?\b|warrants?\b|ads\b|adr\b).*$", re.I)


def clean_name(name: str) -> str:
    n = SEC_TYPE.sub("", (name or "").strip())
    n = re.sub(r"\s+", " ", n).strip(" ,.")
    return (n or (name or ""))[:60]


def fetch_universe() -> pd.DataFrame:
    """Nasdaq screener:一次拿全美股清單與市值。"""
    r = requests.get(NASDAQ_URL, headers=HEADERS, timeout=90,
                     params={"tableonly": "true", "limit": 10000, "offset": 0})
    r.raise_for_status()
    rows = r.json()["data"]["table"]["rows"]
    df = pd.DataFrame(rows)
    df["mcap"] = pd.to_numeric(df.marketCap.str.replace(",", ""), errors="coerce")
    df = df[df.mcap > MIN_MCAP].copy()
    # Yahoo 的代號用 '-' 不是 '/'(BRK/B -> BRK-B)
    df["yahoo"] = df.symbol.str.replace("/", "-", regex=False)
    return df[["symbol", "yahoo", "name", "mcap"]].reset_index(drop=True)


def _one(session: requests.Session, sym: str, retries: int = 2, rng: str = "1y"):
    for attempt in range(retries + 1):
        try:
            # quote: 指數代號帶 ^(^GSPC), 放進路徑前要編碼
            r = session.get(YAHOO_URL.format(urllib.parse.quote(sym)), timeout=25,
                            params={"range": rng, "interval": "1d"})
            if r.status_code == 429:            # 被限流就退一步再試
                time.sleep(2 * (attempt + 1))
                continue
            res = r.json()["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            df = pd.DataFrame({
                "date": pd.to_datetime(res["timestamp"], unit="s", utc=True)
                          .tz_convert("America/New_York").date,
                "close": q["close"],
                "volume": q["volume"],
            }).dropna()
            # 盤中執行時 Yahoo 會回一根還沒收完的「今天」。每日更新遇到已經有
            # 廣度的日子會跳過, 所以盤中算出來的半根 K 棒會永久留在紀錄裡 ——
            # 排程都在美股收盤後跑所以沒踩過, 手動執行就會。收盤(16:00 ET)加
            # 點緩衝之前, 今天那根一律不要。
            ny = pd.Timestamp.now(tz="America/New_York")
            if ny.hour * 60 + ny.minute < 16 * 60 + 20:
                df = df[df["date"] != ny.date()]
            return sym, df if len(df) >= SMA_LONG else None
        except Exception:
            if attempt == retries:
                return sym, None
            time.sleep(1)
    return sym, None


def fetch_bench(rng: str = "1y"):
    """
    S&P 500 收盤, 回傳 (日期序數陣列, 收盤陣列)。抓不到就直接失敗 —— 沒有基準
    就算不出相對強弱, 硬算下去會讓每一檔都被判不合格, 廣度變成 0。
    """
    with requests.Session() as s:
        s.headers.update(HEADERS)
        _, df = _one(s, BENCH, retries=4, rng=rng)
    if df is None or df.empty:
        raise RuntimeError(f"抓不到基準 {BENCH}, 無法計算相對大盤的漲幅")
    df = df.sort_values("date")
    return (np.array([d.toordinal() for d in df["date"]]),
            df["close"].to_numpy(dtype=float))


def fetch_bars(symbols: list[str], rng: str = "1y") -> dict:
    """逐檔抓日 K(預設一年)。回傳 {symbol: DataFrame}, 抓不到的不放進去。"""
    out, done = {}, 0
    with requests.Session() as s:
        s.headers.update(HEADERS)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for sym, df in ex.map(lambda x: _one(s, x, rng=rng), symbols):
                done += 1
                if df is not None:
                    out[sym] = df
                if done % 500 == 0:
                    print(f"  已抓 {done}/{len(symbols)} 檔 (成功 {len(out)})",
                          flush=True)
    miss = len(symbols) - len(out)
    print(f"  完成: {len(out)}/{len(symbols)} 檔 (缺 {miss}, "
          f"{miss/len(symbols)*100:.1f}%)")
    if miss / len(symbols) > 0.25:
        print("  警告: 缺漏超過四分之一, Yahoo 可能在擋人, 結果不可信。",
              file=sys.stderr)
    return out


def to_arrays(bars: dict) -> dict:
    """把各檔的日 K 轉成 numpy 陣列, 供依日期反覆切片。"""
    out = {}
    for sym, df in bars.items():
        out[sym] = (np.array([d.toordinal() for d in df["date"]]),
                    df["close"].to_numpy(dtype=float),
                    df["volume"].to_numpy(dtype=float))
    return out


def trading_days(arrs: dict) -> list:
    """交易日曆:取出現次數最多的那些日期(個股會因停牌等缺日)。"""
    from collections import Counter
    cnt = Counter()
    for d, _, _ in arrs.values():
        cnt.update(d.tolist())
    top = max(cnt.values())
    return sorted(o for o, n in cnt.items() if n > top * 0.5)


def screen(arrs: dict, uni: pd.DataFrame, as_of: int | None = None,
           bench=None) -> pd.DataFrame:
    """
    套用五個條件。as_of 是日期序數(date.toordinal()), 不是位置索引 ——
    每檔的 K 棒數量不同(新上市、停牌), 用「從尾端往回數 n 根」會讓同一個
    偏移量對到不同日期。市值用當下的值, Nasdaq 只給現值沒有歷史。

    bench = fetch_bench() 的 (序數, 收盤)。S&P 500 的漲幅用**這檔自己的視窗**
    算(它 21 根 K 棒前那天到今天), 跟台股一樣, 要比就比同一段。
    """
    if bench is None:
        raise ValueError("screen() 需要 bench(S&P 500 收盤)才能算相對強弱")
    b_ord, b_close = bench

    def b_at(o: int):
        i = int(np.searchsorted(b_ord, o, "right")) - 1
        return float(b_close[i]) if i >= 0 else None

    mc = dict(zip(uni.yahoo, uni.mcap))
    nm = dict(zip(uni.yahoo, uni.name))
    rows = []
    for sym, (dates, c, v) in arrs.items():
        end = len(dates) if as_of is None else int(np.searchsorted(dates, as_of, "right"))
        if end < SMA_LONG + 1:
            continue
        if as_of is not None and dates[end - 1] != as_of:
            continue                       # 該日這檔沒有交易
        cc, vv = c[:end], v[:end]
        close = cc[-1]
        turnover = close * vv[-1]
        perf = (close / cc[-PERF_DAYS - 1] - 1) * 100
        b_now, b_base = b_at(int(dates[end - 1])), b_at(int(dates[end - PERF_DAYS - 1]))
        if not b_now or not b_base:
            continue
        b_perf = (b_now / b_base - 1) * 100
        excess = perf - b_perf
        if not (close > cc[-SMA_LONG:].mean()
                and cc[-10:].mean() > cc[-20:].mean()
                and turnover > MIN_TURNOVER
                and excess > MIN_EXCESS):
            continue
        rows.append({"symbol": sym, "name": clean_name(nm.get(sym)),
                     "close": round(float(close), 2),
                     "perf_1m": round(float(perf), 1),
                     "idx_1m": round(float(b_perf), 1),     # 同期 S&P 500 漲幅
                     "excess_1m": round(float(excess), 1),  # 超額 = 個股 - S&P 500
                     "turnover": round(turnover / 1e6, 1),
                     "mcap": round(mc.get(sym, 0) / 1e9, 2),
                     "date": dt.date.fromordinal(int(dates[end - 1])).isoformat()})
    df = pd.DataFrame(rows)
    return df.sort_values("perf_1m", ascending=False) if len(df) else df


def count_universe(arrs: dict, as_of: int) -> int:
    """該日有足夠歷史可供評估的檔數(廣度的分母)。"""
    n = 0
    for dates, _, _ in arrs.values():
        end = int(np.searchsorted(dates, as_of, "right"))
        if end >= SMA_LONG + 1 and dates[end - 1] == as_of:
            n += 1
    return n


def load_breadth() -> pd.DataFrame:
    if BREADTH_PATH.exists():
        return pd.read_csv(BREADTH_PATH, dtype={"date": str})
    return pd.DataFrame(columns=["date", "count", "universe", "pct"])


def save_breadth(df: pd.DataFrame) -> pd.DataFrame:
    BREADTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    df = df[df.date.isin(sorted(df.date.unique())[-KEEP_DAYS:])]
    df.to_csv(BREADTH_PATH, index=False)
    return df


MEMBER_PATH = ROOT / "data" / "us_momentum_members.csv.gz"
KEEP_MEMBER_DAYS = 280


def load_members() -> pd.DataFrame:
    if MEMBER_PATH.exists():
        df = pd.read_csv(MEMBER_PATH, dtype={"date": str, "symbol": str})
    else:
        df = pd.DataFrame(columns=["date", "symbol", "theme"])
    # 整欄空的時候 read_csv 會給 float64, 之後寫字串進去會 TypeError
    df["theme"] = df["theme"].astype("object") if "theme" in df else None
    return df


def save_members(df: pd.DataFrame) -> pd.DataFrame:
    MEMBER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if "theme" not in df:
        df = df.assign(theme=None)
    df = df.copy()
    df["theme"] = df["theme"].astype("object")
    # 去重讓「有題材」的那筆贏, 否則回補的空題材會蓋掉當天標好的結果
    df["_has"] = df["theme"].notna() & (df["theme"].astype(str) != "")
    df = (df.sort_values(["date", "symbol", "_has"])
            .drop_duplicates(subset=["date", "symbol"], keep="last")
            .drop(columns="_has"))
    keep = sorted(df["date"].unique())[-KEEP_MEMBER_DAYS:]
    df = df[df["date"].isin(keep)].sort_values(["date", "symbol"])
    df.to_csv(MEMBER_PATH, index=False, compression="gzip")
    return df


def streaks(members: pd.DataFrame, dates: list) -> dict:
    """{symbol: (連續交易日數, 本段起算日, 累計天數)}。隔週末不算中斷。"""
    if members.empty or not dates:
        return {}
    pos = {d: i for i, d in enumerate(dates)}
    out = {}
    for sym, g in members.groupby("symbol"):
        idx = sorted({pos[d] for d in g["date"] if d in pos})
        if not idx or idx[-1] != len(dates) - 1:
            continue
        run = 1
        while run < len(idx) and idx[-run - 1] == idx[-1] - run:
            run += 1
        out[sym] = (run, dates[idx[-run]], len(idx))
    return out


def prev_theme_counts(members: pd.DataFrame, dates: list, today_theme: dict) -> dict:
    """前一個交易日各題材的檔數。兩天用同一套標籤才比得準,理由同台股。"""
    if members.empty or len(dates) < 2:
        return {}
    last = {}
    for sym, g in members.groupby("symbol"):
        k = g.sort_values("date")["theme"].dropna()
        k = k[k != ""]
        if len(k):
            last[sym] = k.iloc[-1]
    label = {**last, **{k: v for k, v in today_theme.items() if v}}
    counts = {}
    for sym in members[members["date"] == dates[-2]]["symbol"]:
        t = label.get(sym) or "未歸類"
        counts[t] = counts.get(t, 0) + 1
    return counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true",
                    help="連同歷史廣度與名單一起重算。抓兩年日 K, 扣掉 SMA200 的"
                         "回看約可重算 280 天(抓一年只剩 50 天左右)")
    args = ap.parse_args()
    rng = "2y" if args.backfill else "1y"

    uni = fetch_universe()
    print(f"宇宙: {len(uni)} 檔 (市值 > ${MIN_MCAP/1e6:.0f}M)")
    bench = fetch_bench(rng)
    bars = fetch_bars(uni.yahoo.tolist(), rng=rng)
    if not bars:
        print("沒有取得任何日 K,中止。", file=sys.stderr)
        return 1

    arrs = to_arrays(bars)
    cal = trading_days(arrs)
    picks = screen(arrs, uni, cal[-1], bench)
    rows = load_breadth().to_dict("records")
    mem = load_members()
    mem_rows = mem.to_dict("records")
    have_mem = set(mem["date"])
    # 回補是「用現在的條件重算」, 所以該日的舊名單要整批換掉; 但同一天同一檔
    # 若還在新名單裡, 當時標好的題材要留著
    old_theme = {(r["date"], r["symbol"]): r["theme"] for r in mem_rows
                 if isinstance(r.get("theme"), str) and r["theme"]}
    rebuilt, new_mem = set(), []

    have = {r["date"] for r in rows}
    days = cal if args.backfill else cal[-1:]
    for i, o in enumerate(days, 1):
        d = dt.date.fromordinal(o).isoformat()
        if d in have and not args.backfill:
            continue
        n_uni = count_universe(arrs, o)
        if n_uni < 500:            # 該日可評估的檔數太少, 不具代表性
            continue
        p = screen(arrs, uni, o, bench)
        rows.append({"date": d, "count": len(p), "universe": n_uni,
                     "pct": round(len(p) / n_uni * 100, 2)})
        if args.backfill:
            rebuilt.add(d)
            new_mem.extend({"date": d, "symbol": sym, "theme": old_theme.get((d, sym))}
                           for sym in (p["symbol"] if len(p) else []))
        elif len(p) and d not in have_mem:
            # 每日更新:已經有名單的日子不重建, 否則會把那天標好的題材洗掉
            mem_rows.extend({"date": d, "symbol": sym, "theme": None}
                            for sym in p["symbol"])
        if args.backfill and i % 20 == 0:
            print(f"  廣度回補 {i}/{len(days)} ({d}: {len(p)} 檔)", flush=True)
    if args.backfill:
        mem_rows = [r for r in mem_rows if r["date"] not in rebuilt] + new_mem
        # 重算不到的舊日子(超出兩年視窗)還是舊條件, 留著會讓序列前後定義不一致
        stale = have - rebuilt
        if stale:
            print(f"  丟掉 {len(stale)} 天無法用新條件重算的舊廣度", flush=True)
            rows = [r for r in rows if r["date"] not in stale]
            mem_rows = [r for r in mem_rows if r["date"] not in stale]
    breadth = save_breadth(pd.DataFrame(rows))
    members = save_members(pd.DataFrame(mem_rows))

    recs = picks.drop(columns=["date"]).to_dict("records")

    # 題材標註。跟台股一樣不讓它拖垮主流程 —— Google News 掛掉或格式變了,
    # 廣度數字還是要照樣產出。
    groups = []
    us_themes_mod = None
    try:
        try:
            import us_themes
        except ModuleNotFoundError:
            from scripts import us_themes
        us_themes_mod = us_themes
        us_themes.annotate(recs)
        groups = us_themes.summarize(recs)
        named = sum(1 for r in recs if r.get("theme"))
        print(f"  題材: {named}/{len(recs)} 檔已歸類, {len(groups)} 組")
    except Exception as e:
        print(f"  題材判斷失敗({e.__class__.__name__}),略過。")

    # 上榜天數
    mem_dates = sorted(members["date"].unique())
    st = streaks(members, mem_dates)
    today = mem_dates[-1] if mem_dates else None
    for r in recs:
        run, since, total = st.get(r["symbol"], (1, today, 1))
        r["days"], r["since"], r["days_total"] = run, since, total

    # 把今天判斷出來的題材寫回名單, 下次才有前一天的族群分布可比。
    # 覆寫檔當後備 —— 昨天在榜、今天掉出去的個股今天不會被標註。
    try:
        ov = us_themes_mod.T.load_overrides(us_themes_mod.OVERRIDES_PATH)              if us_themes_mod else {}
    except Exception:
        ov = {}
    today_theme = {**ov, **{r["symbol"]: r.get("theme") for r in recs if r.get("theme")}}
    if mem_dates:
        cur = members["date"] == today
        members.loc[cur, "theme"] = members.loc[cur, "symbol"].map(today_theme)
        members = save_members(members)

    prev = prev_theme_counts(members, mem_dates, today_theme)
    if prev:
        for g in groups:
            g["prev"] = prev.get(g["theme"], 0)
            g["delta"] = g["count"] - g["prev"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({
        "updated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "trade_date": picks.date.mode()[0] if len(picks) else None,
        "universe": len(bars),
        "picks": recs,
        "themes": groups,
        "member_days": len(mem_dates),
        "breadth": breadth.tail(120).to_dict("records"),
    }, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    print(f"完成: 通過 {len(picks)} 檔 / 宇宙 {len(bars)} 檔 | "
          f"廣度序列 {len(breadth)} 天")
    return 0


if __name__ == "__main__":
    sys.exit(main())
