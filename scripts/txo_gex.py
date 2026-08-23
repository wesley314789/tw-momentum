#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
txo_gex.py — 台指選擇權 (TXO) GEX 位階重建

輸出每日六個位階（Call Wall / Put Wall / 山頂 / 山谷 / Micro Flip / Macro Zero），
座標為「期貨/遠期價」，不是加權指數 —— 基差已透過買賣權平價內含。

用法:
    python txo_gex.py --probe 2026-08-11         # 只印欄位, 第一次跑先做這個
    python txo_gex.py --backfill                 # 回補約 280 個交易日
    python txo_gex.py --backfill --start 2025-01-01 --end 2026-08-15
    python txo_gex.py                            # 每日更新(預設動作)

依賴: pandas, numpy, scipy, requests
    pip install pandas numpy scipy requests
"""

import argparse
import io
import json
import sys
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq
from scipy.signal import find_peaks
from scipy.stats import norm

# ---------------------------------------------------------------- 設定

ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = ROOT / "data" / "gex_history.csv"
OUTPUT_PATH = ROOT / "docs" / "data" / "gex_latest.json"
KEEP_DAYS = 280           # 與股票那邊的滾動視窗一致

TPE = dt.timezone(dt.timedelta(hours=8))
TAIFEX_OPT_URL = "https://www.taifex.com.tw/cht/3/optDataDown"
TAIFEX_FUT_URL = "https://www.taifex.com.tw/cht/3/futDataDown"
TXF_COLS = ["date", "txf_settle", "txf_close", "txf_oi", "txf_night"]
MULTIPLIER = 50          # TXO 每點 NT$50
RISK_FREE = 0.015        # 台灣無風險利率, 影響極小
GRID_PCT = 0.07          # flip 掃描範圍 ±7%
GRID_STEP = 5            # 台指用 5 點一格

# 品質過濾 —— 這兩個值會顯著影響結果, 建議自己做敏感度測試
MIN_OI = 30              # OI 低於此值的序列丟棄
MIN_PRICE = 0.5          # 結算價低於此值 IV 反推不可靠

# 山谷偵測
VALLEY_BIN = 100         # 履約價分箱寬度。近月有 50 點檔且量能遠小於百點檔,
                         # 不分箱的話每根 50 點檔都會被誤判成谷
VALLEY_MIN_DEPTH = 0.15  # 谷的深度(prominence)至少要達區間內最大 gross gamma 的比例

# regime 中性帶: 淨 gamma 佔總 gamma 低於此比例就標 neutral, 不硬判正負
NEUTRAL_RATIO = 0.05

# 反推 IV 失敗時的處理: 'drop' 丟棄 / 'atm' 用該到期日 ATM IV 填補
IV_FALLBACK = "atm"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://www.taifex.com.tw/cht/3/optDataDown",
}

# 期交所欄位名稱可能隨時間微調, 集中在這裡方便修
COL = {
    "date":    "交易日期",
    "product": "契約",
    "expiry":  "到期月份(週別)",
    "strike":  "履約價",
    "cp":      "買賣權",
    "settle":  "結算價",
    "oi":      "未沖銷契約數",
    "session": "交易時段",
    "expdate": "契約到期日",       # 2025/12/08 之後才有
}


# ---------------------------------------------------------------- 下載

def taipei_today() -> dt.date:
    """台北當日。runner 跑在 UTC, 排在台北早上時 UTC 日期會落後一天, 用當地
    日期推算交易日才不會受排程時間影響。"""
    return dt.datetime.now(TPE).date()


def _post_csv(url: str, commodity: str, start: dt.date, end: dt.date,
              *, index_col=None, retries: int = 3) -> pd.DataFrame:
    """對期交所下載端點發 POST 並解析 CSV。"""
    payload = {
        "down_type": "1",
        "commodity_id": commodity,
        "queryStartDate": start.strftime("%Y/%m/%d"),
        "queryEndDate": end.strftime("%Y/%m/%d"),
    }
    for attempt in range(retries):
        try:
            r = requests.post(url, data=payload, headers=HEADERS, timeout=60)
            r.raise_for_status()
            # 期交所常見 big5/ms950, 偶爾 utf-8-sig
            for enc in ("ms950", "big5hkscs", "utf-8-sig"):
                try:
                    text = r.content.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("無法解碼回應")
            # low_memory=False: 期交所的到期月份欄位在同一份檔案裡會混著月選
            # 代碼與週選代碼, 分塊推斷型別會噴 DtypeWarning
            df = pd.read_csv(io.StringIO(text), index_col=index_col,
                             low_memory=False)
            df.columns = [str(c).strip() for c in df.columns]
            # 用「有沒有預期欄位」判斷是不是真的被擋, 不能用回應長度 —— 查到
            # 非交易日時期交所會回一份只有表頭的合法 CSV(約 197 bytes), 用長度
            # 門檻會把這種正常的空結果誤判成失敗並重試三次後拋例外。
            if "交易日期" not in df.columns:
                raise ValueError(f"回應不含預期欄位, 可能被擋: {list(df.columns)[:4]}")
            return df
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  重試 {attempt+1}/{retries}: {e}", file=sys.stderr)
            time.sleep(3 * (attempt + 1))


def fetch_range(start: dt.date, end: dt.date, retries: int = 3) -> pd.DataFrame:
    """抓一段期間的 TXO 每日行情。期交所限制單次查詢不超過一個月。"""
    return _post_csv(TAIFEX_OPT_URL, "TXO", start, end, retries=retries)


def fetch_txf_range(start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    抓台指期(TX)近月行情, 每個交易日整理成一列。

    兩個容易踩的地方:
      * 商品代碼是 TX 不是 TXF —— 用 TXF 會回一個合法的 200 但空表, 很容易
        誤判成「當天沒資料」。
      * 資料列尾端多一個逗號(20 欄對上 19 欄的表頭), 不指定 index_col=False
        的話 pandas 會把第一欄當索引, 整排欄位往左位移。

    期交所的交易日: 夜盤在前(前日 15:00~當日 05:00), 日盤在後(08:45~13:45)。
    夜盤那批沒有結算價也沒有 OI, 只有成交價 —— 這也是位階無法盤中更新的原因。
    """
    raw = _post_csv(TAIFEX_FUT_URL, "TX", start, end, index_col=False)
    if raw.empty:
        return pd.DataFrame(columns=TXF_COLS)

    # 排除價差組合(到期月份含 '/'), 只留單式契約
    df = raw[~raw["到期月份(週別)"].astype(str).str.contains("/")].copy()
    df["_date"] = pd.to_datetime(df["交易日期"], errors="coerce").dt.date
    df["_month"] = df["到期月份(週別)"].astype(str).str.strip()
    df["_sess"] = df["交易時段"].astype(str).str.strip()
    for c in ("結算價", "收盤價", "未沖銷契約數"):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")

    rows = []
    for day, grp in df.groupby("_date"):
        if day is None or pd.isna(day):
            continue
        rec = {"date": day.isoformat(), "txf_settle": np.nan,
               "txf_close": np.nan, "txf_oi": np.nan, "txf_night": np.nan}
        reg = grp[grp["_sess"] == "一般"]
        month = None
        if not reg.empty:
            # 取 OI 最大的月份, 不是最近月 —— 轉倉在到期前幾天就發生了(8/17 時
            # 九月 62k 口已經多過八月 51k), 挑近月會抓到正在死掉的那一口, 到期
            # 當天甚至只剩 11k 口且結算價是 0。
            main = reg.loc[reg["未沖銷契約數"].idxmax()]
            month = main["_month"]
            # 到期當天的結算價期交所寫 0, 跟選擇權同一個道理, 當成缺值
            rec["txf_settle"] = main["結算價"] or np.nan
            rec["txf_close"] = main["收盤價"]
            rec["txf_oi"] = main["未沖銷契約數"]

        aft = grp[grp["_sess"] == "盤後"]
        if not aft.empty:
            # 夜盤沒有 OI, 沿用日盤挑出的月份; 挑不到就退而取成交量最大的
            same = aft[aft["_month"] == month] if month else aft.iloc[0:0]
            pick = same.iloc[0] if not same.empty else aft.iloc[0]
            rec["txf_night"] = pick["收盤價"]
        rows.append(rec)
    return pd.DataFrame(rows, columns=TXF_COLS)


def fetch_txf_chunked(start: dt.date, end: dt.date) -> pd.DataFrame:
    """按月切片抓期貨並串接(同 fetch_chunked, 期交所單次查詢有期間上限)。"""
    frames, cur = [], start
    while cur <= end:
        nxt = min(cur + dt.timedelta(days=27), end)
        frames.append(fetch_txf_range(cur, nxt))
        cur = nxt + dt.timedelta(days=1)
        time.sleep(1)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TXF_COLS)
    return out.drop_duplicates(subset=["date"], keep="last")


def fetch_chunked(start: dt.date, end: dt.date) -> pd.DataFrame:
    """按月切片下載並串接。對期交所客氣一點, 每次間隔 1 秒。"""
    frames, cur = [], start
    while cur <= end:
        nxt = min(cur + dt.timedelta(days=27), end)
        print(f"下載 {cur} ~ {nxt}", file=sys.stderr)
        frames.append(fetch_range(cur, nxt))
        cur = nxt + dt.timedelta(days=1)
        time.sleep(1)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- 到期日

def third_wednesday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 1)
    first_wed = d + dt.timedelta(days=(2 - d.weekday()) % 7)
    return first_wed + dt.timedelta(days=14)


def nth_weekday(year: int, month: int, n: int, weekday: int) -> dt.date:
    """weekday: 週一=0 ... 週三=2, 週五=4"""
    d = dt.date(year, month, 1)
    first = d + dt.timedelta(days=(weekday - d.weekday()) % 7)
    return first + dt.timedelta(days=7 * (n - 1))


def resolve_expiry(code: str) -> dt.date | None:
    """
    解析到期代碼。已知格式:
        202608     月選   -> 第三個週三
        202608W1   週選   -> 該月第 1 個週三 (W3 不掛牌)
        202608F1   雙週選 -> 該月第 1 個週五
    遇到未知格式回 None, 由呼叫端決定丟棄或報錯。

    注意: 這裡沒有處理國定假日順延。近期資料請優先用「契約到期日」欄位。
    """
    code = str(code).strip().upper()
    try:
        if len(code) == 6 and code.isdigit():
            return third_wednesday(int(code[:4]), int(code[4:6]))
        if len(code) >= 7 and code[6] in ("W", "F"):
            y, m, n = int(code[:4]), int(code[4:6]), int(code[7:])
            return nth_weekday(y, m, n, 2 if code[6] == "W" else 4)
    except (ValueError, IndexError):
        pass
    return None


# ---------------------------------------------------------------- 定價

def b76_price(F, K, T, sigma, r, is_call):
    if T <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0) if is_call else max(K - F, 0)
        return np.exp(-r * T) * intrinsic
    v = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / v
    d2 = d1 - v
    disc = np.exp(-r * T)
    if is_call:
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


_SQRT_2PI = np.sqrt(2 * np.pi)


def _norm_pdf(x):
    """標準常態 PDF。scipy 的 norm.pdf 純量呼叫很慢, flip 掃描要跑上百萬次, 直接算。"""
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def b76_gamma(F, K, T, sigma, r):
    """對遠期價的 gamma。這正是你交易期貨時的曝險。"""
    if T <= 0 or sigma <= 0 or F <= 0:
        return 0.0
    v = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / v
    return np.exp(-r * T) * _norm_pdf(d1) / (F * v)


def implied_vol(price, F, K, T, r, is_call):
    if price < MIN_PRICE or T <= 0:
        return np.nan
    intrinsic = np.exp(-r * T) * (max(F - K, 0) if is_call else max(K - F, 0))
    if price <= intrinsic:
        return np.nan
    try:
        return brentq(lambda s: b76_price(F, K, T, s, r, is_call) - price,
                      1e-4, 5.0, xtol=1e-6, maxiter=100)
    except (ValueError, RuntimeError):
        return np.nan


def implied_forward(calls: pd.DataFrame, puts: pd.DataFrame, T: float) -> float | None:
    """
    買賣權平價反推遠期價:  F = K + e^{rT}(C - P)
    取 |C-P| 最小的履約價 (最接近價平), 該處平價關係最穩定。

    這是整套腳本的關鍵一步 —— 它讓基差自動內含, 不必另外抓加權指數,
    也不必估股利率。除權息旺季的逆價差問題在這裡就解決了。
    """
    merged = calls.merge(puts, on="K", suffixes=("_c", "_p"))
    merged = merged[(merged["oi_c"] > 0) & (merged["oi_p"] > 0)]
    # 兩腳都要有實際報價。期交所對最後交易日的到期序列不發結算價(整批寫 0),
    # 不擋掉的話 C-P 會全等於 0, |C-P| 最小變成隨便挑一檔, 反推出來的 F 是垃圾,
    # 再往下污染整條鏈的 IV。
    merged = merged[(merged["px_c"] >= MIN_PRICE) & (merged["px_p"] >= MIN_PRICE)]
    if len(merged) < 3:
        return None
    merged["diff"] = (merged["px_c"] - merged["px_p"]).abs()
    row = merged.nsmallest(1, "diff").iloc[0]
    return float(row["K"] + np.exp(RISK_FREE * T) * (row["px_c"] - row["px_p"]))


# ---------------------------------------------------------------- GEX

def build_chain(day_df: pd.DataFrame, trade_date: dt.date) -> list[dict]:
    """把當日行情整理成 [{K, T, iv, oi_c, oi_p}, ...], 並附上該日整體 F。"""
    chain = []
    forwards = {}

    for exp_code, grp in day_df.groupby("_expiry_date"):
        # groupby 對 object-dtype 的日期欄位有時回傳 Timestamp、有時回傳
        # datetime.date,型別不一致會讓相減直接炸掉,統一轉成 Timestamp 再算。
        T = (pd.Timestamp(exp_code) - pd.Timestamp(trade_date)).days / 365.0
        if T <= 0:
            # 當日到期直接排除。結算日那批部位看起來很大(2026-08-19 佔全場 OI
            # 的 75%), 但期交所對最後交易日的到期序列不發結算價(整批 0), 反推
            # 不出 IV; 而且收盤後它就不再交易, 那份 gamma 不會再產生盤中避險。
            continue

        cp = grp[COL["cp"]].astype(str).str.strip()
        c = grp[cp == "買權"]
        p = grp[cp == "賣權"]
        c = c.rename(columns={COL["strike"]: "K", COL["settle"]: "px_c", COL["oi"]: "oi_c"})
        p = p.rename(columns={COL["strike"]: "K", COL["settle"]: "px_p", COL["oi"]: "oi_p"})
        c = c[["K", "px_c", "oi_c"]]
        p = p[["K", "px_p", "oi_p"]]

        F = implied_forward(c, p, T)
        if F is None or not np.isfinite(F) or F <= 0:
            continue
        forwards[exp_code] = (F, T)

        merged = c.merge(p, on="K", how="outer").fillna(0)
        merged = merged[(merged["oi_c"] + merged["oi_p"]) >= MIN_OI]

        ivs = []
        for _, row in merged.iterrows():
            K = float(row["K"])
            iv_c = implied_vol(row["px_c"], F, K, T, RISK_FREE, True)
            iv_p = implied_vol(row["px_p"], F, K, T, RISK_FREE, False)
            pair = [x for x in (iv_c, iv_p) if not np.isnan(x)]
            ivs.append(float(np.mean(pair)) if pair else np.nan)
        merged["iv"] = ivs

        if IV_FALLBACK == "atm":
            atm_iv = merged.loc[(merged["K"] - F).abs().nsmallest(3).index, "iv"].mean()
            merged["iv"] = merged["iv"].fillna(atm_iv)
        merged = merged.dropna(subset=["iv"])

        for _, row in merged.iterrows():
            chain.append(dict(K=float(row["K"]), T=T, iv=float(row["iv"]),
                              oi_c=float(row["oi_c"]), oi_p=float(row["oi_p"])))

    # 用近月的 F 當作全域參考價 (最接近你實際交易的近月期貨)
    if not forwards:
        return [], None
    nearest = min(forwards, key=lambda e: forwards[e][1])
    return chain, forwards[nearest][0]


def chain_arrays(chain: list[dict]):
    """把 chain 打包成 numpy 陣列。flip 掃描要對上千個價位重算全鏈, 逐筆迴圈太慢。"""
    return (
        np.array([c["K"] for c in chain], dtype=float),
        np.array([c["T"] for c in chain], dtype=float),
        np.array([c["iv"] for c in chain], dtype=float),
        np.array([c["oi_c"] - c["oi_p"] for c in chain], dtype=float),
        np.array([c["oi_c"] + c["oi_p"] for c in chain], dtype=float),
    )


def gamma_matrix(F_grid, K, T, iv):
    """對 (模擬價位 × 序列) 一次算完 gamma, 回傳 shape (len(F_grid), len(K)) 的矩陣。"""
    F_col = np.atleast_1d(np.asarray(F_grid, dtype=float))[:, None]
    valid = (T > 0) & (iv > 0)
    v = np.where(valid, iv * np.sqrt(np.where(valid, T, 1.0)), 1.0)
    d1 = (np.log(F_col / K) + 0.5 * iv ** 2 * T) / v
    g = np.exp(-RISK_FREE * T) * _norm_pdf(d1) / (F_col * v)
    return np.where(valid, g, 0.0)


def gex_curve(F_grid, arrays) -> np.ndarray:
    """
    在每個模擬價位重算全鏈 dollar gamma。
    符號採靜態假設: 造市商 long call / short put。
    這是整套最脆弱的假設 —— 改成流量歸屬會得到不同結果。
    """
    K, T, iv, net, _ = arrays
    F_col = np.atleast_1d(np.asarray(F_grid, dtype=float))[:, None]
    g = gamma_matrix(F_grid, K, T, iv)
    return (g * net * MULTIPLIER * F_col ** 2 * 0.01).sum(axis=1)


def gex_at(F_sim: float, chain: list[dict]) -> float:
    """單一價位的全鏈 dollar gamma。"""
    if not chain:
        return 0.0
    return float(gex_curve([F_sim], chain_arrays(chain))[0])


def gex_by_strike(F: float, chain: list[dict]) -> pd.DataFrame:
    K, T, iv, net, gross = chain_arrays(chain)
    base = gamma_matrix(F, K, T, iv)[0] * MULTIPLIER * F ** 2 * 0.01
    return pd.DataFrame({"K": K, "net": base * net, "gross": base * gross}) \
             .groupby("K", as_index=False).sum()


def regime_label(gex_now: float, net_ratio: float) -> str:
    """淨 gamma 太小(多空幾乎抵銷)時標 neutral, 不硬給正負。"""
    if not np.isfinite(net_ratio) or net_ratio < NEUTRAL_RATIO:
        return "neutral"
    return "positive" if gex_now > 0 else "negative"


def find_valley(by_k: pd.DataFrame, F: float) -> float:
    """
    山谷: 現價下方兩個量能區之間的凹陷 —— 跌破後沒有 gamma 承接、容易加速的真空帶。

    不能直接取區間最小值: gross gamma 隨著遠離價平單調衰減, 全域最小幾乎永遠
    落在掃描窗最外緣那根沒人交易的空履約價上, 沒有結構意義(實測 280 天有一半
    貼在 -6% 以外)。這裡改成先分箱抹掉 50 點檔的鋸齒, 再找有足夠深度的局部
    低點, 取最接近現價的那個。找不到夠深的谷就回 NaN, 不硬給數字。
    """
    lo = F * (1 - GRID_PCT)
    sub = by_k[(by_k["K"] < F) & (by_k["K"] > lo)]
    if len(sub) < 3:
        return np.nan

    binned = (sub.assign(_bin=(sub["K"] // VALLEY_BIN) * VALLEY_BIN)
                 .groupby("_bin", as_index=False)["gross"].sum()
                 .sort_values("_bin"))
    if len(binned) < 3:
        return np.nan

    x = binned["_bin"].to_numpy()
    y = binned["gross"].to_numpy()
    troughs, _ = find_peaks(-y, prominence=y.max() * VALLEY_MIN_DEPTH)
    if not len(troughs):
        return np.nan
    return float(x[troughs].max())    # 下方離現價最近的谷 = 履約價最大的那個


def compute_levels(chain: list[dict], F: float) -> dict:
    if not chain:
        return {}

    by_k = gex_by_strike(F, chain)
    above = by_k[by_k["K"] > F]
    below = by_k[by_k["K"] < F]

    lv = {"F": round(F, 1)}
    lv["call_wall"] = float(above.loc[above["net"].idxmax(), "K"]) if len(above) else np.nan
    lv["put_wall"]  = float(below.loc[below["net"].idxmin(), "K"]) if len(below) else np.nan
    lv["peak"]      = float(by_k.loc[by_k["gross"].idxmax(), "K"])

    lv["valley"] = find_valley(by_k, F)

    # Flip: 掃網格找總 GEX 零軸穿越
    grid = np.arange(F * (1 - GRID_PCT), F * (1 + GRID_PCT), GRID_STEP)
    arrays = chain_arrays(chain)
    curve = gex_curve(grid, arrays)
    lv["gex_now"] = float(gex_curve([F], arrays)[0])

    # net_ratio: 淨 gamma 佔總 gamma 的比例, 用來判斷 regime 標籤可不可信。
    # 貼近翻轉點時多空 gamma 幾乎抵銷完, 這個值趨近 0, 此時正負號會被夜盤/日盤
    # 取價這種小差異翻掉(2026-08-19 就是這樣, 我們標正、對照組標負)。
    # 這裡只存原始比值不套門檻, 門檻留給呼叫端調, 免得改一次就要重算整段歷史。
    gross_total = float(by_k["gross"].sum())
    lv["net_ratio"] = abs(lv["gex_now"]) / gross_total if gross_total else np.nan
    lv["regime"] = regime_label(lv["gex_now"], lv["net_ratio"])

    def zero_cross(c):
        """取最接近現價的零軸穿越, 線性內插。"""
        sc = np.where(np.diff(np.sign(c)) != 0)[0]
        if not len(sc):
            return np.nan
        i = sc[np.argmin(np.abs(grid[sc] - F))]
        x0, x1, y0, y1 = grid[i], grid[i + 1], c[i], c[i + 1]
        return float(x0 - y0 * (x1 - x0) / (y1 - y0))

    lv["micro_flip"] = zero_cross(curve)

    # Macro Zero: 只用 >7 天的合約重掃
    macro = [c for c in chain if c["T"] * 365 > 7]
    lv["macro_zero"] = (zero_cross(gex_curve(grid, chain_arrays(macro)))
                        if macro else np.nan)

    return lv


# ---------------------------------------------------------------- 主流程

def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = [k for k in ("date", "expiry", "strike", "cp", "settle", "oi")
               if COL[k] not in df.columns]
    if missing:
        print("\n找不到必要欄位:", [COL[k] for k in missing], file=sys.stderr)
        print("實際欄位:", list(df.columns), file=sys.stderr)
        print("請修改腳本頂端的 COL 字典。\n", file=sys.stderr)
        sys.exit(1)

    # 只留日盤 (未沖銷契約數以日盤收盤為準)
    if COL["session"] in df.columns:
        df = df[df[COL["session"]].astype(str).str.contains("一般", na=True)]

    if COL["product"] in df.columns:
        df = df[df[COL["product"]].astype(str).str.strip() == "TXO"]

    for k in ("strike", "settle", "oi"):
        df[COL[k]] = pd.to_numeric(df[COL[k]].astype(str).str.replace(",", ""),
                                   errors="coerce")
    df = df.dropna(subset=[COL["strike"], COL["settle"], COL["oi"]])
    df = df[df[COL["oi"]] > 0]

    df["_trade_date"] = pd.to_datetime(df[COL["date"]], errors="coerce").dt.date

    # 優先用官方到期日欄位, 沒有才用推算
    if COL["expdate"] in df.columns:
        # 這欄是整數 YYYYMMDD(如 20260717),不能直接丟給 to_datetime——
        # 數字型輸入會被當成 Unix 時間戳記(奈秒),解析成 1970 年附近。
        parsed = pd.to_datetime(df[COL["expdate"]].astype(str).str.strip(),
                                format="%Y%m%d", errors="coerce").dt.date
        df["_expiry_date"] = parsed
        gap = df["_expiry_date"].isna()
        if gap.any():
            df.loc[gap, "_expiry_date"] = df.loc[gap, COL["expiry"]].map(resolve_expiry)
    else:
        df["_expiry_date"] = df[COL["expiry"]].map(resolve_expiry)

    unresolved = df["_expiry_date"].isna().sum()
    if unresolved:
        codes = df.loc[df["_expiry_date"].isna(), COL["expiry"]].unique()[:10]
        print(f"警告: {unresolved} 列無法解析到期日, 已丟棄。範例代碼: {list(codes)}",
              file=sys.stderr)
    return df.dropna(subset=["_trade_date", "_expiry_date"])


HIST_COLS = ["date", "dow", "is_settle", "F", "txf_settle", "txf_close",
             "txf_oi", "txf_night", "call_wall", "put_wall", "peak",
             "valley", "micro_flip", "macro_zero", "gex_now", "net_ratio", "regime"]


def merge_txf(rows: list[dict], txf: pd.DataFrame) -> list[dict]:
    """把台指期價格併進每日位階。缺的日子留空, 不影響位階本身。"""
    if txf is None or txf.empty:
        return rows
    by_date = txf.set_index("date").to_dict("index")
    for r in rows:
        r.update(by_date.get(r["date"], {}))
    return rows


def load_history() -> pd.DataFrame:
    if HISTORY_PATH.exists():
        return pd.read_csv(HISTORY_PATH, dtype={"date": str})
    return pd.DataFrame(columns=HIST_COLS)


def save_history(df: pd.DataFrame) -> pd.DataFrame:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        df = pd.DataFrame(columns=HIST_COLS)
        df.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")
        return df
    df = df.drop_duplicates(subset=["date"], keep="last")
    keep = sorted(df["date"].unique())[-KEEP_DAYS:]
    df = df[df["date"].isin(keep)].sort_values("date")
    df = df[[c for c in HIST_COLS if c in df.columns]]
    df.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")
    return df


def compute_day(day_df: pd.DataFrame, trade_date: dt.date) -> dict | None:
    chain, F = build_chain(day_df, trade_date)
    if not chain or F is None:
        return None
    lv = compute_levels(chain, F)
    if not lv:
        return None
    lv["date"] = trade_date.isoformat()
    lv["dow"] = trade_date.strftime("%a")
    lv["is_settle"] = trade_date.weekday() in (2, 4)   # 粗略, 未計假日順延
    return lv


def backfill(start: dt.date, end: dt.date):
    """回補指定期間的每日 GEX 位階。期交所單次查詢上限約一個月,故按月切片。"""
    hist = load_history()
    have = set(hist["date"].unique())
    rows = hist.to_dict("records")

    raw = fetch_chunked(start, end)
    df = prepare(raw)

    n = 0
    for trade_date, day_df in df.groupby("_trade_date"):
        if trade_date.isoformat() in have:
            continue
        try:
            lv = compute_day(day_df, trade_date)
        except Exception as e:
            print(f"  {trade_date} 失敗: {e}", file=sys.stderr, flush=True)
            continue
        if lv:
            rows.append(lv)
            n += 1
            print(f"  {trade_date} OK F={lv['F']}", flush=True)
            if n % 20 == 0:
                save_history(pd.DataFrame(rows))   # 中途存檔, 中斷不會前功盡棄

    rows = merge_txf(rows, fetch_txf_chunked(start, end))
    save_history(pd.DataFrame(rows))
    print(f"backfill 完成: {n} 個新交易日")


def preopen_snapshot(df: pd.DataFrame, txf: pd.DataFrame,
                     levels_date: str) -> dict:
    """
    盤前快照:當日夜盤已收(05:00)但日盤還沒開(08:45)時, 用夜盤收盤價重算
    「現在落在曲線的哪一側」。

    位階本身不動 —— 它們是 OI 結構的性質, 而當日 OI 要收盤後才有。這裡重算的
    只有 gex_now / net_ratio / regime, 也就是唯一真正隨價格改變的東西
    (實測 ±2% 的價格變動下 micro_flip 完全不動, 牆最多跳一個檔位)。
    """
    if txf.empty:
        return {}
    # 夜盤有價、日盤還沒結算 = 盤前狀態, 且要比位階那天更新
    cand = txf[(txf["date"] > levels_date) & txf["txf_night"].notna()]
    if cand.empty:
        return {}
    row = cand.sort_values("date").iloc[-1]
    price = float(row["txf_night"])

    day = dt.date.fromisoformat(levels_date)
    chain, _ = build_chain(df[df["_trade_date"] == day], day)
    if not chain:
        return {}

    gex = float(gex_curve([price], chain_arrays(chain))[0])
    gross = float(gex_by_strike(price, chain)["gross"].sum())
    ratio = abs(gex) / gross if gross else np.nan
    return {
        "preopen_date": row["date"],
        "preopen_price": price,
        "preopen_gex": gex,
        "preopen_net_ratio": ratio,
        "preopen_regime": regime_label(gex, ratio),
    }


def daily_update():
    """抓最近 7 個日曆日(含今天)並補上尚未計算過的交易日,順便寫出最新一天給前端。"""
    end = taipei_today()
    start = end - dt.timedelta(days=7)
    raw = fetch_range(start, end)
    df = prepare(raw)
    if df.empty:
        print("今日無資料(假日?),跳過。")
        return False

    hist = load_history()
    have = set(hist["date"].unique())
    rows = hist.to_dict("records")
    for trade_date, day_df in df.groupby("_trade_date"):
        if trade_date.isoformat() in have:
            continue
        lv = compute_day(day_df, trade_date)
        if lv:
            rows.append(lv)
            print(f"  {trade_date} OK F={lv['F']}")

    # 台指期價格每天都補一次(不只新算的那幾天) —— 夜盤收盤價會隨著交易日推進
    # 補上來, 舊的列也可能因此從缺值變成有值。
    txf = fetch_txf_range(start, end)
    rows = merge_txf(rows, txf)
    merged = save_history(pd.DataFrame(rows))
    if merged.empty:
        print("尚無任何已計算的交易日。")
        return False

    latest = merged.sort_values("date").iloc[-1].to_dict()
    # 盤前快照不進歷史檔:它是「還沒收盤的那天」的暫時狀態, 等當日 OI 出來
    # 之後就會被真正的位階取代。
    pre = preopen_snapshot(df, txf, latest["date"])
    latest.update(pre)
    latest["updated"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    # 位階可能是 NaN(當天找不到夠深的谷、或沒有零軸穿越)。json.dumps 預設會寫出
    # 裸的 NaN, 那不是合法 JSON, 前端 JSON.parse 會直接爆掉、卡片靜默消失。
    latest = {k: (None if pd.isna(v) else v) for k, v in latest.items()}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(latest, ensure_ascii=False, allow_nan=False),
                           encoding="utf-8")
    msg = f"完成: {latest['date']} | F={latest['F']} | regime={latest['regime']}"
    if pre:
        msg += (f" | 盤前 {pre['preopen_date']} 夜盤收 {pre['preopen_price']:.0f}"
                f" -> regime={pre['preopen_regime']}")
    print(msg)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=str,
                    help="只抓單日並印出欄位, 第一次跑先用這個")
    ap.add_argument("--backfill", action="store_true",
                    help="回補約 280 個交易日的歷史 GEX 資料")
    ap.add_argument("--start", type=str, help="回補起始日(配合 --backfill)")
    ap.add_argument("--end", type=str, help="回補結束日(配合 --backfill,預設今天)")
    args = ap.parse_args()

    if args.probe:
        d = dt.date.fromisoformat(args.probe)
        raw = fetch_range(d, d)
        print("欄位:", list(raw.columns))
        print(raw.head(12).to_string())
        return

    if args.backfill:
        end = dt.date.fromisoformat(args.end) if args.end else taipei_today()
        start = (dt.date.fromisoformat(args.start) if args.start
                 else end - dt.timedelta(days=420))
        backfill(start, end)
        return

    daily_update()


if __name__ == "__main__":
    main()
