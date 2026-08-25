#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_taifex.py — 探測期交所資料的發布時間(手動用)

原本掛在排程上跑, 為了回答兩個只能實測的問題。答案量到了, 排程已移除,
腳本留著供日後手動查驗(例如懷疑期交所改了發布時間)。

    python scripts/probe_taifex.py

── 2026-08 的實測結果(排程據此訂定) ─────────────────────────────
1. 當日選擇權 OI 幾點發布?
     台北 07:18  尚未 —— 「有列但無日盤資料」(日盤還沒開, 本來就不會有)
     台北 14:39  已發布(14:39 / 14:49 兩次量到 OI 與結算價都齊全)
     14:39 之後各時段皆正常
   -> 日盤 13:45 收盤後不到一小時就有。每日排程的第一班訂在台北 14:30,
      靠 GitHub 本身的延遲(實測中位數 48 分)自然落在安全區。

2. 台指期日盤/夜盤何時可取得?
     夜盤(前日 15:00~當日 05:00)在收完之後才發布 —— 進行中查詢會回空表。
     台北 07:18 時當日夜盤收盤價已可取得(當日日盤與 OI 則還沒有)。
   -> 這就是第二班(台北 07:00)能做出盤前快照的依據。

只讀不寫, 不碰資料檔也不 commit。
"""

import io
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

TPE = timezone(timedelta(hours=8))
OPT_URL = "https://www.taifex.com.tw/cht/3/optDataDown"
FUT_URL = "https://www.taifex.com.tw/cht/3/futDataDown"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://www.taifex.com.tw/cht/3/futDataDown",
}


def fetch(url: str, commodity: str, day) -> pd.DataFrame:
    payload = {
        "down_type": "1",
        "commodity_id": commodity,
        "queryStartDate": day.strftime("%Y/%m/%d"),
        "queryEndDate": day.strftime("%Y/%m/%d"),
    }
    r = requests.post(url, data=payload, headers=HEADERS, timeout=60)
    r.raise_for_status()
    for enc in ("ms950", "big5hkscs", "utf-8-sig"):
        try:
            text = r.content.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("無法解碼回應")
    # 期交所的資料列尾端多一個逗號(20 欄 vs 表頭 19 欄), 不加 index_col=False
    # 會被 pandas 當成索引欄而整排位移。
    df = pd.read_csv(io.StringIO(text), index_col=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")


def session(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "交易時段" not in df.columns:
        return df.iloc[0:0]
    return df[df["交易時段"].astype(str).str.strip() == name]


def probe_options(day) -> str:
    try:
        df = fetch(OPT_URL, "TXO", day)
    except Exception as e:
        return f"抓取失敗({e.__class__.__name__})"
    if df.empty:
        return "尚未發布(0 列)"
    reg = session(df, "一般")
    if reg.empty:
        return f"有 {len(df)} 列但無日盤資料"
    oi = num(reg["未沖銷契約數"]).notna().sum()
    settle = num(reg["結算價"]).notna().sum()
    if oi == 0:
        return f"日盤 {len(reg)} 列, OI 尚未發布"
    return f"OK — 日盤 {len(reg)} 序列, OI {oi} 有值, 結算價 {settle} 有值"


def probe_futures(day) -> tuple[str, str]:
    try:
        df = fetch(FUT_URL, "TX", day)
    except Exception as e:
        return f"抓取失敗({e.__class__.__name__})", "—"
    if df.empty:
        return "尚未發布(0 列)", "尚未發布"
    # 排除價差組合(到期月份含 '/'), 只留單式契約, 取最近月
    single = df[~df["到期月份(週別)"].astype(str).str.contains("/")].copy()
    single["_m"] = single["到期月份(週別)"].astype(str).str.strip()

    reg = session(single, "一般")
    if reg.empty:
        day_txt = "無日盤資料"
    else:
        front = reg.sort_values("_m").iloc[0]
        s, oi = num(pd.Series([front["結算價"]]))[0], num(pd.Series([front["未沖銷契約數"]]))[0]
        if pd.isna(s):
            day_txt = f"日盤 {len(reg)} 列, 結算價尚未發布"
        else:
            day_txt = f"OK — 近月 {front['_m']} 結算 {s:.0f}, OI {oi:.0f}"

    aft = session(single, "盤後")
    if aft.empty:
        night_txt = "無夜盤資料"
    else:
        front = aft.sort_values("_m").iloc[0]
        c = num(pd.Series([front["收盤價"]]))[0]
        night_txt = ("無收盤價" if pd.isna(c)
                     else f"OK — 近月 {front['_m']} 收盤 {c:.0f}")
    return day_txt, night_txt


def main():
    now = datetime.now(TPE)
    day = now.date()
    prev = day - timedelta(days=1)
    while prev.weekday() >= 5:          # 跳過週末, 找上一個可能的交易日
        prev -= timedelta(days=1)

    print(f"[probe] 台北時間 {now:%Y-%m-%d %H:%M (%a)}")
    print(f"— 當日 {day} —")
    print(f"  TXO 選擇權 OI : {probe_options(day)}")
    d, n = probe_futures(day)
    print(f"  TX  期貨日盤   : {d}")
    print(f"  TX  期貨夜盤   : {n}")
    print(f"— 前一交易日 {prev} —")
    print(f"  TXO 選擇權 OI : {probe_options(prev)}")
    pd_, pn = probe_futures(prev)
    print(f"  TX  期貨日盤   : {pd_}")
    print("  (交易日定義: 夜盤在前(前日15:00~當日05:00), 日盤在後(08:45~13:45)。")
    print("   早上七點那班要回答的是: 當日夜盤(05:00 剛收)的收盤價拿不拿得到)")


if __name__ == "__main__":
    sys.exit(main())
