#!/usr/bin/env python3
"""
台股強勢股掃描 — 資料管線
1. 每日抓取上市(TWSE)+ 上櫃(TPEx)全市場收盤資料
2. 累積歷史資料到 data/history.csv.gz(滾動保留 280 個交易日)
3. 計算 SEPA Trend Template 與當日強勢清單
4. 輸出 docs/data/latest.json 給前端

首次使用請先跑 backfill(證交所/櫃買中心免費歷史 API,無需金鑰):
    python scripts/update_data.py --backfill
之後每日更新(GitHub Actions 自動執行):
    python scripts/update_data.py
"""

import argparse
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = ROOT / "data" / "history.csv.gz"
OUTPUT_PATH = ROOT / "docs" / "data" / "latest.json"
KEEP_DAYS = 280  # 滾動保留的交易日數(> 252 即可算 52 週)

HEADERS = {"User-Agent": "Mozilla/5.0 (tw-momentum-scanner)"}


# ---------------------------------------------------------------- utilities

def is_common_stock(code: str) -> bool:
    """只保留普通股:4 碼數字且非 00 開頭(排除 ETF、權證、特別股等)。"""
    return len(code) == 4 and code.isdigit() and not code.startswith("00")


def to_float(x):
    try:
        v = float(str(x).replace(",", ""))
        return v if v > 0 or str(x).strip() in ("0", "0.0") else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------- fetchers

def _get_with_retry(url, *, retries=5, backoff=8, **kwargs):
    """對暫時性連線錯誤(逾時、連線中斷等)重試幾次再放棄。"""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, **kwargs)
            r.raise_for_status()
            return r
        except (requests.exceptions.RequestException,) as e:
            if attempt == retries:
                raise
            print(f"  請求失敗({e.__class__.__name__}),{backoff}s 後重試 "
                  f"({attempt}/{retries})…")
            time.sleep(backoff)


def fetch_twse_today() -> pd.DataFrame:
    """證交所 OpenAPI:上市個股當日收盤(免金鑰)。"""
    url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
    r = _get_with_retry(url, timeout=30)
    rows = []
    for it in r.json():
        code = it.get("Code", "").strip()
        if not is_common_stock(code):
            continue
        close = to_float(it.get("ClosingPrice"))
        if close is None:
            continue
        rows.append({
            "code": code,
            "name": it.get("Name", "").strip(),
            "market": "上市",
            "open": to_float(it.get("OpeningPrice")),
            "high": to_float(it.get("HighestPrice")),
            "low": to_float(it.get("LowestPrice")),
            "close": close,
            "volume": to_float(it.get("TradeVolume")) or 0,   # 股數
            "value": to_float(it.get("TradeValue")) or 0,     # 成交金額
        })
    return pd.DataFrame(rows)


def fetch_tpex_today() -> pd.DataFrame:
    """櫃買中心 OpenAPI:上櫃個股當日收盤(免金鑰)。"""
    url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
    r = _get_with_retry(url, timeout=30)
    rows = []
    for it in r.json():
        code = (it.get("SecuritiesCompanyCode") or "").strip()
        if not is_common_stock(code):
            continue
        close = to_float(it.get("Close"))
        if close is None:
            continue
        rows.append({
            "code": code,
            "name": (it.get("CompanyName") or "").strip(),
            "market": "上櫃",
            "open": to_float(it.get("Open")),
            "high": to_float(it.get("High")),
            "low": to_float(it.get("Low")),
            "close": close,
            "volume": to_float(it.get("TradingShares")) or 0,
            "value": to_float(it.get("TransactionAmount")) or 0,
        })
    return pd.DataFrame(rows)


def fetch_twse_date(d: date) -> pd.DataFrame:
    """證交所:歷史單日上市個股收盤(免金鑰,backfill 用)。"""
    url = "https://www.twse.com.tw/exchangeReport/MI_INDEX"
    params = {"response": "json", "date": d.strftime("%Y%m%d"),
              "type": "ALLBUT0999"}
    r = _get_with_retry(url, params=params, timeout=30)
    table = next((t for t in r.json().get("tables", [])
                  if "每日收盤行情" in t.get("title", "")), None)
    if table is None:
        return pd.DataFrame()
    rows = []
    for it in table["data"]:
        code = it[0].strip()
        if not is_common_stock(code):
            continue
        close = to_float(it[8])
        if close is None:
            continue
        rows.append({
            "date": d.isoformat(),
            "code": code,
            "name": it[1].strip(),
            "market": "上市",
            "open": to_float(it[5]),
            "high": to_float(it[6]),
            "low": to_float(it[7]),
            "close": close,
            "volume": to_float(it[2]) or 0,
            "value": to_float(it[4]) or 0,
        })
    return pd.DataFrame(rows)


def fetch_tpex_date(d: date) -> pd.DataFrame:
    """櫃買中心:歷史單日上櫃個股收盤(免金鑰,backfill 用)。"""
    url = "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes"
    params = {"date": d.strftime("%Y/%m/%d")}
    r = _get_with_retry(url, params=params, timeout=30)
    table = next((t for t in r.json().get("tables", [])
                  if "上櫃股票行情" in t.get("title", "")), None)
    if table is None:
        return pd.DataFrame()
    rows = []
    for it in table["data"]:
        code = it[0].strip()
        if not is_common_stock(code):
            continue
        close = to_float(it[2])
        if close is None:
            continue
        rows.append({
            "date": d.isoformat(),
            "code": code,
            "name": it[1].strip(),
            "market": "上櫃",
            "open": to_float(it[4]),
            "high": to_float(it[5]),
            "low": to_float(it[6]),
            "close": close,
            "volume": to_float(it[8]) or 0,
            "value": to_float(it[9]) or 0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- history

def load_history() -> pd.DataFrame:
    if HISTORY_PATH.exists():
        return pd.read_csv(HISTORY_PATH, dtype={"code": str})
    return pd.DataFrame(columns=["date", "code", "name", "market", "open",
                                 "high", "low", "close", "volume", "value"])


def save_history(df: pd.DataFrame):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 只保留最近 KEEP_DAYS 個交易日
    keep = sorted(df["date"].unique())[-KEEP_DAYS:]
    df = df[df["date"].isin(keep)]
    df.to_csv(HISTORY_PATH, index=False, compression="gzip",
              float_format="%.4g")
    return df


def backfill():
    """回補約 280 個交易日的歷史資料(證交所+櫃買中心,每個交易日 2 次 API 呼叫)。"""
    hist = load_history()
    have = set(hist["date"].unique())
    d = date.today()
    fetched, frames = 0, [hist]
    # 往回掃 420 個日曆日,足夠涵蓋 280 個交易日
    for i in range(1, 421):
        day = d - timedelta(days=i)
        if day.weekday() >= 5 or day.isoformat() in have:
            continue
        try:
            twse = fetch_twse_date(day)
            tpex = fetch_tpex_date(day)
        except requests.exceptions.RequestException as e:
            print(f"  {day} 抓取失敗({e.__class__.__name__}),略過此日。")
            continue
        df = pd.concat([twse, tpex], ignore_index=True)
        if not df.empty:
            frames.append(df)
            fetched += 1
            print(f"  {day} ✓ {len(df)} 檔")
        if fetched >= KEEP_DAYS:
            break
        if fetched % 20 == 0:
            save_history(pd.concat(frames, ignore_index=True)
                         .drop_duplicates(subset=["date", "code"], keep="last"))
        time.sleep(1)  # 尊重伺服器,避免過於頻繁請求
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "code"], keep="last")
    save_history(merged)
    print(f"backfill 完成:{fetched} 個交易日")


# ---------------------------------------------------------------- compute

def rs_score(g: pd.DataFrame) -> float | None:
    """IBD 式加權動能:40% 三個月 + 各 20% 六/九/十二個月報酬。"""
    c = g["close"].to_numpy()
    n = len(c)
    if n < 130:
        return None

    def ret(days):
        return c[-1] / c[-days] - 1 if n >= days else None

    r3, r6, r9, r12 = ret(63), ret(126), ret(189), ret(252)
    parts, weights = [], []
    for r, w in [(r3, 0.4), (r6, 0.2), (r9, 0.2), (r12, 0.2)]:
        if r is not None:
            parts.append(r * w)
            weights.append(w)
    return sum(parts) / sum(weights) if weights else None


def compute(hist: pd.DataFrame) -> dict:
    hist = hist.sort_values(["code", "date"])
    latest_date = hist["date"].max()
    today = hist[hist["date"] == latest_date].set_index("code")

    records = []
    for code, g in hist.groupby("code"):
        if code not in today.index:
            continue
        c = g["close"].to_numpy()
        v = g["volume"].to_numpy()
        n = len(c)
        if n < 60:
            continue
        row = today.loc[code]
        close = c[-1]
        prev = c[-2] if n >= 2 else close
        chg_pct = (close / prev - 1) * 100 if prev else 0

        ma50 = c[-50:].mean() if n >= 50 else None
        ma150 = c[-150:].mean() if n >= 150 else None
        ma200 = c[-200:].mean() if n >= 200 else None
        ma200_prev = c[-221:-21].mean() if n >= 221 else None
        hi52 = c[-252:].max()
        lo52 = c[-252:].min()
        vol20 = v[-21:-1].mean() if n >= 21 else None
        vol_ratio = v[-1] / vol20 if vol20 else None

        rec = {
            "code": code,
            "name": row["name"] or "",
            "market": row["market"] or "",
            "close": round(close, 2),
            "chg_pct": round(chg_pct, 2),
            "value": round(row["value"] / 1e8, 2),  # 億元
            "vol_ratio": round(vol_ratio, 2) if vol_ratio else None,
            "off_high": round((close / hi52 - 1) * 100, 1),
            "above_low": round((close / lo52 - 1) * 100, 1),
            "rs_raw": rs_score(g),
            "n_days": n,
        }

        # SEPA Trend Template(RS 條件另外算完 percentile 再判斷)
        if all(x is not None for x in (ma50, ma150, ma200, ma200_prev)):
            rec["tt"] = [
                close > ma150 and close > ma200,      # 1. 價格在150/200MA之上
                ma150 > ma200,                        # 2. 150MA > 200MA
                ma200 > ma200_prev,                   # 3. 200MA 上升(近一個月)
                ma50 > ma150 > ma200,                 # 4. 均線多頭排列
                close > ma50,                         # 5. 價格在 50MA 之上
                close >= lo52 * 1.30,                 # 6. 高於 52 週低點 30%+
                close >= hi52 * 0.75,                 # 7. 距 52 週高點 25% 內
            ]
        else:
            rec["tt"] = None
        records.append(rec)

    if not records:
        # 歷史資料不足 60 天(例如尚未 backfill),無法計算 RS/SEPA
        return {
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "trade_date": latest_date,
            "universe": 0,
            "sepa": [],
            "daily": [],
        }

    df = pd.DataFrame(records)

    # RS Rating:全市場 percentile 1–99
    valid = df["rs_raw"].notna()
    df.loc[valid, "rs"] = (df.loc[valid, "rs_raw"].rank(pct=True) * 98 + 1
                           ).round().astype(int)
    df["rs"] = df["rs"].where(df["rs"].notna(), None)

    # SEPA 清單:7 條件全過 + RS >= 70
    def sepa_pass(r):
        return (r["tt"] is not None and all(r["tt"])
                and r["rs"] is not None and r["rs"] >= 70)

    sepa = df[df.apply(sepa_pass, axis=1)].copy()
    sepa = sepa.sort_values("rs", ascending=False)

    # 當日強勢:漲幅 >= 4%、量比 >= 1.5、成交值 >= 1 億
    daily = df[(df["chg_pct"] >= 4) & (df["vol_ratio"] >= 1.5)
               & (df["value"] >= 1)].copy()
    daily = daily.sort_values(["chg_pct", "vol_ratio"], ascending=False)

    def pack(sub: pd.DataFrame):
        cols = ["code", "name", "market", "close", "chg_pct", "value",
                "vol_ratio", "off_high", "above_low", "rs"]
        out = sub[cols].where(sub[cols].notna(), None)
        return out.to_dict(orient="records")

    return {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "trade_date": latest_date,
        "universe": int(len(df)),
        "sepa": pack(sepa),
        "daily": pack(daily.head(150)),
    }


# ---------------------------------------------------------------- main

def daily_update():
    twse = fetch_twse_today()
    tpex = fetch_tpex_today()
    today_df = pd.concat([twse, tpex], ignore_index=True)
    if today_df.empty:
        print("今日無資料(假日?),跳過。")
        return False
    today_df["date"] = date.today().isoformat()

    hist = load_history()
    # 名稱/市場別以最新資料為準,回填舊資料
    name_map = today_df.set_index("code")[["name", "market"]]
    hist = hist[hist["date"] != today_df["date"].iloc[0]]
    merged = pd.concat([hist, today_df], ignore_index=True)
    merged["name"] = merged["code"].map(name_map["name"]).fillna(merged["name"])
    merged["market"] = merged["code"].map(name_map["market"]).fillna(merged["market"])
    merged = save_history(merged)

    result = compute(merged)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False),
                           encoding="utf-8")
    print(f"完成:{result['trade_date']} | 全市場 {result['universe']} 檔 | "
          f"SEPA {len(result['sepa'])} 檔 | 當日強勢 {len(result['daily'])} 檔")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true",
                    help="回補約 280 個交易日的歷史資料(證交所+櫃買中心,免金鑰)")
    args = ap.parse_args()

    if args.backfill:
        backfill()
        # 回補完直接算一次
        result = compute(load_history())
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False),
                               encoding="utf-8")
    else:
        daily_update()
