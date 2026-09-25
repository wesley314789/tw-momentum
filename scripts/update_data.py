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
import os
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import certifi
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = ROOT / "data" / "history.csv.gz"
INDEX_PATH = ROOT / "data" / "index.csv.gz"
SHARES_PATH = ROOT / "data" / "shares.csv"
BREADTH_PATH = ROOT / "data" / "breadth.csv"
# 每個交易日通過動能篩選的名單(date, code, theme)。廣度只存檔數, 這裡存
# 是誰 —— 上榜天數和族群的日增減都要靠它。
MEMBER_PATH = ROOT / "data" / "momentum_members.csv.gz"
OUTPUT_PATH = ROOT / "docs" / "data" / "latest.json"
KEEP_DAYS = 280  # 滾動保留的交易日數(> 252 即可算 52 週)

# 篩選門檻
MIN_VALUE = 1.0    # 成交值下限(億)。兩份清單共用 —— 沒有這道濾網, SEPA 會塞滿
                   # 一天只成交幾十萬元的個股(實測佔三成), 條件再漂亮也進不去
MIN_RS = 70        # SEPA 的 RS 下限
DAILY_CHG = 4.0    # 當日強勢: 漲幅下限(%)
DAILY_VOL_RATIO = 1.5   # 當日強勢: 量比下限

# 動能篩選(市場廣度用):
#   價格 > SMA200、SMA10 > SMA20、總市值 > 20 億、成交值 > 5000 萬、
#   一個月漲幅 - 大盤一個月漲幅 > 10 個百分點
# 最後一條原本是 TradingView 那組的「一個月漲幅 > 20%」。絕對門檻的問題是它
# 跟著大盤走:大盤一個月漲 15% 時, 只多漲 5% 的股票就能過; 大盤跌 10% 時,
# 逆勢漲 15% 的股票反而過不了。2026-09-18 改成相對大盤, 選出的是「跑贏市場」
# 而不是「剛好在漲的市場裡」。
BR_MCAP = 2e9      # 總市值下限(元)
BR_TURNOVER = 5e7  # 成交值下限(元)
BR_EXCESS = 10.0   # 一個月超額報酬下限(百分點, 個股漲幅 - 加權指數漲幅)
BR_PERF_DAYS = 20  # 「一個月」取 20 個交易日(四週)。原本是 21(= 252/12,
                   # 跟 RS 的 63/126/189/252 同一套換算), 2026-09-18 依使用者偏好
                   # 改成 20。兩者都只是近似 —— TradingView 的「1 個月」是日曆月,
                   # 實際落在 20~23 個交易日之間
# 篩選條件的指紋。回測/研究腳本的名單快取檔名帶著它 —— 條件一改, 快取自然
# 失效, 不會悄悄拿舊定義的名單去算新問題。
SCREEN_SIG = f"x{BR_EXCESS:g}_mc{BR_MCAP:.0e}_tv{BR_TURNOVER:.0e}_d{BR_PERF_DAYS}"

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

# --- TPEx 的 TLS 憑證鏈不完整 ---------------------------------------------
# www.tpex.org.tw 在 2026-09-07 換發憑證後只送葉憑證, 沒附中繼憑證
# (openssl s_client 量到鏈長 1)。瀏覽器會照 AIA 欄位自己去補, Python 不會,
# 所以在乾淨的環境(GitHub runner)上直接 CERTIFICATE_VERIFY_FAILED。
# 把缺的那張跟 certifi 的根憑證合成一個暫時 bundle —— 中繼憑證的簽發者
# TWCA CYBER Root CA 本來就在 certifi 裡, 所以驗證仍是完整一條鏈到受信任的
# 根, 不是 verify=False。詳見 scripts/certs/README.txt。
EXTRA_CA = Path(__file__).resolve().parent / "certs" / "twca_ssl_subca.pem"
_ca_bundle = None


def ca_bundle():
    """回傳 certifi 根 + TPEx 缺的那張中繼憑證 合成的 bundle 路徑。"""
    global _ca_bundle
    if _ca_bundle is None:
        if not EXTRA_CA.exists():
            _ca_bundle = certifi.where()
        else:
            fd, path = tempfile.mkstemp(prefix="ca-", suffix=".pem")
            with os.fdopen(fd, "wb") as f:
                f.write(Path(certifi.where()).read_bytes())
                f.write(b"\n")
                f.write(EXTRA_CA.read_bytes())
            _ca_bundle = path
    return _ca_bundle


def _get_with_retry(url, *, retries=5, backoff=8, **kwargs):
    """對暫時性連線錯誤(逾時、連線中斷等)重試幾次再放棄。"""
    kwargs.setdefault("verify", ca_bundle())
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


def fetch_twse_date(d: date) -> tuple[pd.DataFrame, float | None]:
    """證交所:歷史單日上市個股收盤 + 加權股價指數(免金鑰,backfill/daily 用)。"""
    url = "https://www.twse.com.tw/exchangeReport/MI_INDEX"
    params = {"response": "json", "date": d.strftime("%Y%m%d"),
              "type": "ALLBUT0999"}
    r = _get_with_retry(url, params=params, timeout=30)
    tables = r.json().get("tables", [])

    table = next((t for t in tables if "每日收盤行情" in t.get("title", "")), None)
    rows = []
    if table is not None:
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

    idx_table = next((t for t in tables
                       if "價格指數(臺灣證券交易所)" in t.get("title", "")), None)
    index_close = None
    if idx_table is not None:
        idx_row = next((r for r in idx_table["data"]
                         if "發行量加權股價指數" in r[0]), None)
        if idx_row is not None:
            index_close = to_float(idx_row[1])

    return pd.DataFrame(rows), index_close


def fetch_tpex_date(d: date, with_raw: bool = False):
    """
    櫃買中心:歷史單日上櫃個股收盤(免金鑰,backfill 用)。

    with_raw=True 時額外回傳發行股數(欄位索引 15) —— 上櫃的股數就藏在這份每日
    行情裡, 不必另外打一支 API。
    """
    url = "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes"
    params = {"date": d.strftime("%Y/%m/%d")}
    r = _get_with_retry(url, params=params, timeout=30)
    table = next((t for t in r.json().get("tables", [])
                  if "上櫃股票行情" in t.get("title", "")), None)
    if table is None:
        return (pd.DataFrame(), []) if with_raw else pd.DataFrame()
    rows, shares = [], []
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
        n = to_float(it[15]) if len(it) > 15 else None
        if n:
            shares.append({"code": code, "shares": n})
    return (pd.DataFrame(rows), shares) if with_raw else pd.DataFrame(rows)


def fetch_shares() -> pd.DataFrame:
    """
    發行股數(算總市值用)。上市取證交所公司基本資料 OpenAPI, 上櫃直接在每日行情
    裡就有(欄位「發行股數」)。

    注意: 這兩個來源給的都是**當下**的股數, 沒有歷史。回補歷史廣度時等於用今天
    的股數去乘過去的股價, 遇到現金增資/減資的個股會失真。因為市值門檻只是個粗
    篩(20 億), 多數個股離門檻很遠, 影響有限, 但要知道有這回事。
    """
    rows = []
    r = _get_with_retry("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", timeout=40)
    payload = r.json()
    # 欄位名稱用比對找, 不寫死 —— 它實際叫「已發行普通股數或TDR原股發行股數」,
    # 少抄一個「股」字就會全部解析失敗而且不會報錯(只是安靜地一筆都對不上)。
    key = next((k for k in payload[0]
                if "已發行" in k and "股數" in k), None) if payload else None
    if key is None:
        raise ValueError(f"找不到發行股數欄位, 現有欄位: {list(payload[0])[:12]}")
    for it in payload:
        code = str(it.get("公司代號", "")).strip()
        n = to_float(it.get(key))
        if is_common_stock(code) and n:
            rows.append({"code": code, "shares": n})

    _, tpex_raw = fetch_tpex_date(date.today(), with_raw=True)
    rows.extend(tpex_raw)

    df = pd.DataFrame(rows).drop_duplicates(subset=["code"], keep="first")
    return df[df["shares"] > 0]


def load_shares() -> pd.DataFrame:
    if SHARES_PATH.exists():
        return pd.read_csv(SHARES_PATH, dtype={"code": str})
    return pd.DataFrame(columns=["code", "shares"])


def save_shares(df: pd.DataFrame):
    SHARES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("code").to_csv(SHARES_PATH, index=False, float_format="%.0f")


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
    # %.4g 只留四位有效數字, 四位數以上的股價會被截掉尾數(13925 -> 13920),
    # 之後讀回來算均線/報酬都帶著這個誤差。改成 %.6g, 檔案大小影響很小。
    df.to_csv(HISTORY_PATH, index=False, compression="gzip",
              float_format="%.6g")
    return df


def load_index_history() -> pd.DataFrame:
    """大盤(加權股價指數)歷史收盤,用來算個股相對大盤的超額報酬。"""
    if INDEX_PATH.exists():
        return pd.read_csv(INDEX_PATH)
    return pd.DataFrame(columns=["date", "close"])


def save_index_history(df: pd.DataFrame):
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = df.drop_duplicates(subset=["date"], keep="last")
    keep = sorted(df["date"].unique())[-KEEP_DAYS:]
    df = df[df["date"].isin(keep)].sort_values("date")
    df.to_csv(INDEX_PATH, index=False, float_format="%.4g")
    return df


def backfill():
    """回補約 280 個交易日的歷史資料(證交所+櫃買中心,每個交易日 2 次 API 呼叫)。"""
    hist = load_history()
    have = set(hist["date"].unique())
    idx_hist = load_index_history()
    idx_rows = idx_hist.to_dict("records")
    d = date.today()
    fetched, frames = 0, [hist]
    # 往回掃 420 個日曆日,足夠涵蓋 280 個交易日
    for i in range(1, 421):
        day = d - timedelta(days=i)
        if day.weekday() >= 5 or day.isoformat() in have:
            continue
        try:
            twse, index_close = fetch_twse_date(day)
            tpex = fetch_tpex_date(day)
        except requests.exceptions.RequestException as e:
            print(f"  {day} 抓取失敗({e.__class__.__name__}),略過此日。")
            continue
        df = pd.concat([twse, tpex], ignore_index=True)
        if not df.empty:
            frames.append(df)
            fetched += 1
            print(f"  {day} ✓ {len(df)} 檔")
        if index_close is not None:
            idx_rows.append({"date": day.isoformat(), "close": index_close})
        if fetched >= KEEP_DAYS:
            break
        if fetched % 20 == 0:
            save_history(pd.concat(frames, ignore_index=True)
                         .drop_duplicates(subset=["date", "code"], keep="last"))
            save_index_history(pd.DataFrame(idx_rows))
        time.sleep(1)  # 尊重伺服器,避免過於頻繁請求
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "code"], keep="last")
    save_history(merged)
    save_index_history(pd.DataFrame(idx_rows))
    print(f"backfill 完成:{fetched} 個交易日")


# ---------------------------------------------------------------- 市場廣度

def index_lookup(idx: pd.DataFrame):
    """
    回傳 date -> 加權指數收盤 的查詢函式。查不到當天就用之前最近一天(asof),
    連之前都沒有就回 None。
    """
    s = idx.dropna(subset=["close"]).drop_duplicates("date").sort_values("date")
    ds = s["date"].to_numpy()
    cs = s["close"].to_numpy(dtype=float)

    def at(d: str):
        i = ds.searchsorted(d, side="right") - 1
        return float(cs[i]) if i >= 0 else None
    return at


def momentum_screen(hist: pd.DataFrame, shares: pd.DataFrame,
                    as_of: str | None = None,
                    idx: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    動能篩選,回傳通過的個股。

        價格 > SMA200、SMA10 > SMA20、總市值 > BR_MCAP、成交值 > BR_TURNOVER、
        BR_PERF_DAYS 個交易日漲幅 - 同期間加權指數漲幅 > BR_EXCESS 個百分點

    大盤漲幅用**這檔股票自己的視窗**算(它 BR_PERF_DAYS 根 K 棒前的那一天到今天),
    不是固定的「全市場 BR_PERF_DAYS 個交易日前」—— 中間停牌過的個股, 兩者的起點
    會不一樣, 要比就得比同一段。

    idx 是加權指數收盤(date, close)。沒給就讀 data/index.csv.gz, 但那份只有
    280 天; 回測更早的區間要自己傳涵蓋得到的序列進來。

    需要 200 天以上的歷史才算得出 SMA200, 所以歷史視窗前 200 天無法回補。
    """
    if as_of:
        hist = hist[hist["date"] <= as_of]
    dates = sorted(hist["date"].unique())
    if len(dates) < 200:
        return pd.DataFrame()
    today = dates[-1]

    ix_at = index_lookup(idx if idx is not None else load_index_history())
    ix_today = ix_at(today)
    if ix_today is None:
        # 寧可大聲失敗也不要悄悄全部算不出來 —— 那會變成「今天廣度 0 檔」,
        # 看起來像市場崩了, 其實只是指數沒抓到。
        raise ValueError(f"{today} 沒有加權指數收盤, 無法計算相對大盤的漲幅")

    sh = dict(zip(shares["code"], shares["shares"]))
    rows = []
    for code, g in hist.sort_values("date").groupby("code"):
        if g["date"].iloc[-1] != today:      # 當天沒交易(停牌等)就跳過
            continue
        # Backtests can supply a total-return approximation for technical
        # signals while keeping raw close for actual price and market cap.
        c = g["signal_close"].to_numpy() if "signal_close" in g else g["close"].to_numpy()
        if len(c) < 200:
            continue
        # The historical backtest marks unexplained corporate-action jumps.
        # A contaminated 200-day window makes both SMA and return signals
        # unreliable; the live pipeline has no _corp field and is unchanged.
        if "_corp" in g and g["_corp"].tail(200).any():
            continue
        close = c[-1]
        raw_close = g["close"].iloc[-1]
        n = sh.get(code)
        if not n:
            continue
        turnover = g["value"].iloc[-1]
        if len(c) <= BR_PERF_DAYS:
            continue
        perf = (close / c[-BR_PERF_DAYS - 1] - 1) * 100
        ix_base = ix_at(g["date"].iloc[-BR_PERF_DAYS - 1])
        if not ix_base:
            continue
        ix_perf = (ix_today / ix_base - 1) * 100
        excess = perf - ix_perf
        if not (close > c[-200:].mean()
                and c[-10:].mean() > c[-20:].mean()
                and raw_close * n > BR_MCAP
                and turnover > BR_TURNOVER
                and excess > BR_EXCESS):
            continue
        rows.append({"code": code, "name": g["name"].iloc[-1],
                     "market": g["market"].iloc[-1], "close": round(raw_close, 2),
                     "perf_1m": round(perf, 1),
                     "idx_1m": round(ix_perf, 1),      # 同一段期間的大盤漲幅
                     "excess_1m": round(excess, 1),    # 超額 = 個股 - 大盤
                     "value": round(turnover / 1e8, 2),
                     "mcap": round(raw_close * n / 1e8, 0)})
    df = pd.DataFrame(rows)
    return df.sort_values("perf_1m", ascending=False) if len(df) else df


def load_breadth() -> pd.DataFrame:
    if BREADTH_PATH.exists():
        return pd.read_csv(BREADTH_PATH, dtype={"date": str})
    return pd.DataFrame(columns=["date", "count", "universe", "pct"])


def save_breadth(df: pd.DataFrame) -> pd.DataFrame:
    BREADTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    df = df[df["date"].isin(sorted(df["date"].unique())[-KEEP_DAYS:])]
    df.to_csv(BREADTH_PATH, index=False)
    return df


def update_breadth(hist: pd.DataFrame, shares: pd.DataFrame,
                   backfill: bool = False,
                   idx: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    記錄每日通過動能篩選的檔數 —— 這就是市場廣度 —— 並把名單一起留下來。

    數字本身比名單更有用:同樣是大盤上漲, 通過的檔數在擴張還是萎縮, 代表漲勢
    是全面的還是集中在少數幾檔。名單則是拿來算上榜天數與族群的日增減 ——
    這裡本來就逐日重跑過 momentum_screen 了, 順手留下來幾乎沒有額外成本。
    """
    br = load_breadth()
    mem = load_members()
    if idx is None:
        idx = load_index_history()     # 讀一次就好, 不要每天在迴圈裡重讀
    have = set(br["date"])
    dates = sorted(hist["date"].unique())
    todo = dates[199:] if backfill else dates[-1:]   # 前 200 天算不出 SMA200
    todo = [d for d in todo if backfill or d not in have]

    rows = br.to_dict("records")
    mem_rows = mem.to_dict("records")
    have_mem = set(mem["date"])
    for i, d in enumerate(todo, 1):
        if d in have and not backfill:
            continue
        sub = hist[hist["date"] <= d]
        picks = momentum_screen(sub, shares, idx=idx)
        universe = int((hist["date"] == d).sum())
        rows.append({"date": d, "count": len(picks), "universe": universe,
                     "pct": round(len(picks) / universe * 100, 2) if universe else 0})
        # 回補的日子沒有當天的新聞可查, theme 留空; 之後由當天的標註補上。
        # 已經有名單的日子不重建 —— 名單本身是從歷史價格決定的、不會變,
        # 重建只會把那天標好的題材洗成空的。
        if len(picks) and d not in have_mem:
            mem_rows.extend({"date": d, "code": c, "theme": None}
                            for c in picks["code"])
        if backfill and i % 20 == 0:
            print(f"  廣度回補 {i}/{len(todo)} ({d}: {len(picks)} 檔)", flush=True)
    return save_breadth(pd.DataFrame(rows)), save_members(pd.DataFrame(mem_rows))


# ------------------------------------------------------- 上榜天數 / 族群增減

def load_members() -> pd.DataFrame:
    if MEMBER_PATH.exists():
        df = pd.read_csv(MEMBER_PATH, dtype={"date": str, "code": str})
    else:
        df = pd.DataFrame(columns=["date", "code", "theme"])
    # theme 整欄空的時候 read_csv 會給 float64, 之後寫字串進去會 TypeError。
    # 明確轉成 object, 讓「還沒標註」和「標註過」共用同一個欄位型別。
    df["theme"] = df["theme"].astype("object") if "theme" in df else None
    return df


def save_members(df: pd.DataFrame) -> pd.DataFrame:
    MEMBER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if "theme" not in df:
        df = df.assign(theme=None)
    # 每次都明確轉 object。整欄都是 None 時 pandas 會把它推成 float64
    # (read_csv 讀空欄、或 DataFrame(records) 從全 None 的紀錄重建都會),
    # 之後往裡面寫字串就 TypeError: Invalid value ... for dtype 'float64'。
    df = df.copy()
    df["theme"] = df["theme"].astype("object")
    # 去重時讓「有題材」的那筆贏 —— 直接 keep="last" 的話, 回補產生的空題材
    # 會蓋掉當天標註好的結果, 族群的日增減就永遠比不出來。
    df["_has"] = df["theme"].notna() & (df["theme"].astype(str) != "")
    df = (df.sort_values(["date", "code", "_has"])
            .drop_duplicates(subset=["date", "code"], keep="last")
            .drop(columns="_has"))
    keep = sorted(df["date"].unique())[-KEEP_DAYS:]
    df = df[df["date"].isin(keep)].sort_values(["date", "code"])
    df.to_csv(MEMBER_PATH, index=False, compression="gzip")
    return df


def streaks(members: pd.DataFrame, dates: list[str]) -> dict:
    """
    每檔的上榜天數。回傳 {code: (連續天數, 本段起算日, 視窗內累計天數)}。

    「連續」按**交易日**算而不是日曆天 —— 中間隔週末或連假不算中斷, 但只要有
    一個交易日掉出名單, 連續就歸零重數。dates 必須是實際有資料的交易日序列。
    """
    if members.empty or not dates:
        return {}
    pos = {d: i for i, d in enumerate(dates)}
    by_code = {}
    for code, g in members.groupby("code"):
        idx = sorted({pos[d] for d in g["date"] if d in pos})
        if not idx or idx[-1] != len(dates) - 1:
            continue                       # 今天不在榜上就不用算
        run = 1
        while run < len(idx) and idx[-run - 1] == idx[-1] - run:
            run += 1
        by_code[code] = (run, dates[idx[-run]], len(idx))
    return by_code


def theme_delta(members: pd.DataFrame, dates: list[str],
                today_theme: dict, forced: dict | None = None) -> dict:
    """
    每個題材相對前一個交易日的檔數增減。

    前一天的個股用**最後一次看到的題材**來歸類, 不是重跑一次當天的新聞 ——
    昨天掉出名單的個股今天不會被標註, 沒有標籤可用; 而且要比較的是「這個族群
    的檔數變多還是變少」, 兩天用同一套標籤才比得準。今天有標註的以今天為準,
    這樣人工覆寫或題材改名會同時套用到兩邊, 不會憑空生出一組差值。

    forced 是覆寫檔(load_overrides)的內容, 最後套用而且 None 也算數 —— 覆寫檔
    裡填 "-" 的個股是「查過、確定不屬於任何族群」。today_theme 會把沒有標籤的
    濾掉(那通常代表「還沒查」), 所以不另外處理的話, 被 "-" 蓋掉的個股前一天會
    退回歷史上那個錯的標籤: 只要它還在榜上, 網頁就每天顯示「原題材 -1、未歸類
    +1」, 其實什麼都沒變。
    """
    if members.empty or len(dates) < 2:
        return {}
    last = {}
    for code, g in members.groupby("code"):
        g = g.sort_values("date")
        known = g["theme"].dropna()
        known = known[known != ""]
        if len(known):
            last[code] = known.iloc[-1]
    label = {**last, **{k: v for k, v in today_theme.items() if v}, **(forced or {})}

    prev_day = dates[-2]
    prev = members[members["date"] == prev_day]["code"]
    counts = {}
    for code in prev:
        counts[label.get(code) or "未歸類"] = counts.get(label.get(code) or "未歸類", 0) + 1
    return counts

# ---------------------------------------------------------------- compute

def period_returns(c) -> dict:
    """3/6/9/12 個月(63/126/189/252 個交易日)報酬率。"""
    n = len(c)

    def ret(days):
        return c[-1] / c[-days] - 1 if n >= days else None

    return {3: ret(63), 6: ret(126), 9: ret(189), 12: ret(252)}


def rs_score(g: pd.DataFrame, idx_rets: dict) -> float | None:
    """IBD 式加權動能,但用相對大盤(加權指數)的超額報酬取代絕對報酬:
    40% 三個月 + 各 20% 六/九/十二個月超額報酬。"""
    c = g["close"].to_numpy()
    if len(c) < 130:
        return None

    rets = period_returns(c)
    parts, weights = [], []
    for months, w in [(3, 0.4), (6, 0.2), (9, 0.2), (12, 0.2)]:
        r = rets[months]
        if r is None:
            continue
        idx_r = idx_rets.get(months)
        excess = r - idx_r if idx_r is not None else r
        parts.append(excess * w)
        weights.append(w)
    return sum(parts) / sum(weights) if weights else None


def compute(hist: pd.DataFrame, idx_hist: pd.DataFrame | None = None) -> dict:
    hist = hist.sort_values(["code", "date"])
    latest_date = hist["date"].max()
    today = hist[hist["date"] == latest_date].set_index("code")

    if idx_hist is None:
        idx_hist = load_index_history()
    idx_hist = idx_hist.sort_values("date")
    idx_rets = (period_returns(idx_hist["close"].to_numpy())
                if len(idx_hist) else {3: None, 6: None, 9: None, 12: None})

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
            "rs_raw": rs_score(g, idx_rets),
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
    # 歷史不足 130 天的個股算不出 RS, 這裡會留 NaN(不是 None ——
    # float 欄位塞 None 會被轉回 NaN), 交給 pack() 統一處理。

    # SEPA 清單:7 條件全過 + RS 門檻 + 成交值門檻
    def sepa_pass(r):
        return (r["tt"] is not None and all(r["tt"])
                and pd.notna(r["rs"]) and r["rs"] >= MIN_RS
                and pd.notna(r["value"]) and r["value"] >= MIN_VALUE)

    sepa = df[df.apply(sepa_pass, axis=1)].copy()
    sepa = sepa.sort_values("rs", ascending=False)

    # 當日強勢:漲幅、量比、成交值三道門檻
    daily = df[(df["chg_pct"] >= DAILY_CHG)
               & (df["vol_ratio"] >= DAILY_VOL_RATIO)
               & (df["value"] >= MIN_VALUE)].copy()
    daily = daily.sort_values(["chg_pct", "vol_ratio"], ascending=False)

    def pack(sub: pd.DataFrame):
        cols = ["code", "name", "market", "close", "chg_pct", "value",
                "vol_ratio", "off_high", "above_low", "rs"]
        # 必須逐格轉 None: DataFrame.where(..., None) 在 float 欄位上是空操作
        # (None 會被存回 NaN), NaN 再被 json.dumps 寫成裸的 NaN —— 那不是合法
        # JSON, 瀏覽器 JSON.parse 會直接拋錯, 整張表變成「尚無資料」。
        return [{k: (None if pd.isna(v) else v) for k, v in rec.items()}
                for rec in sub[cols].to_dict(orient="records")]

    return {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "trade_date": latest_date,
        "universe": int(len(df)),
        "sepa": pack(sepa),
        "daily": pack(daily.head(150)),
    }


# ---------------------------------------------------------------- main

RECENT_DAYS = 6      # 每次往回檢查幾個日曆日, 補上缺漏的交易日


def fetch_day(d: date):
    """
    抓單一交易日。兩個市場都要有資料才算數 —— 收盤後不久跑時證交所與櫃買的
    發布時間不見得同步, 只擋「兩邊都空」的話會把半個市場寫進歷史, 那天的 RS
    百分位和全市場檔數都會失真, 而且會一直留在滾動視窗裡。

    來源整個掛掉(重試用盡)時回 None 而不是讓例外往上拋。2026-09-14 櫃買換了
    憑證卻沒附中繼憑證, 五次重試全失敗, 例外一路衝出 daily_update 讓整支腳本
    exit 1 —— 那天連「缺哪幾天」都沒印出來, 要翻 Actions 的堆疊追蹤才知道發生
    什麼事。單一天抓不到就跳過那天, 其餘的照補, 最後由 daily_update 一次講清楚。
    """
    try:
        twse, index_close = fetch_twse_date(d)
        tpex = fetch_tpex_date(d)
    except Exception as e:
        print(f"  {d} 抓取失敗({e.__class__.__name__}: {str(e)[:120]}),略過。")
        return None, None
    if twse.empty and tpex.empty:
        return None, None            # 假日或尚未開盤
    if twse.empty or tpex.empty:
        who = "證交所" if twse.empty else "櫃買中心"
        print(f"  {d} {who}資料不完整,略過(避免寫入半個市場)。")
        return None, None
    return pd.concat([twse, tpex], ignore_index=True), index_close


def daily_update():
    """
    補上最近 RECENT_DAYS 天內所有還沒進歷史的交易日, 不是只抓「今天」。

    排程並不可靠:GitHub 會延遲(實測中位數 48 分、最長近三小時)甚至整班跳過。
    只抓今天的話, 任何一次沒跑成的日子就永久缺漏 —— 2026-08-27 就是這樣:
    14:30 那班被跳過, 而前一班延遲五小時後跨過 UTC 午夜, date.today() 已經
    變成隔天, 於是連原本要補的日子也一起錯過。改成掃區間就不受排程時間影響。
    """
    hist = load_history()
    have = set(hist["date"].unique())
    today = date.today()
    targets = [today - timedelta(days=i) for i in range(RECENT_DAYS)]
    targets = [d for d in targets
               if d.weekday() < 5 and d.isoformat() not in have]

    frames, idx_rows = [], []
    for d in sorted(targets):
        df, index_close = fetch_day(d)
        if df is None:
            continue
        frames.append(df)
        if index_close is not None:
            idx_rows.append({"date": d.isoformat(), "close": index_close})
        print(f"  {d} 補上 {len(df)} 檔")

    if not frames:
        # 「沒有缺口」和「有缺口但交易所還沒發布」是完全不同的狀況, 一定要分開
        # 印。2026-08-28 那班在台北 14:25 跑, 兩個交易所都還沒出當日資料, 但
        # 訊息寫「資料已是最新」, 看 log 會以為一切正常。
        if targets:
            print(f"缺 {', '.join(d.isoformat() for d in sorted(targets))},"
                  f"但交易所尚未發布(或為休市日),這次沒補到。")
        else:
            print("沒有需要補的交易日(資料已是最新)。")
        return False

    new_df = pd.concat(frames, ignore_index=True)
    # 名稱/市場別以最新一天為準,回填舊資料
    latest_day = new_df["date"].max()
    name_map = new_df[new_df["date"] == latest_day].set_index("code")[["name", "market"]]
    name_map = name_map[~name_map.index.duplicated()]
    merged = pd.concat([hist, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "code"], keep="last")
    merged["name"] = merged["code"].map(name_map["name"]).fillna(merged["name"])
    merged["market"] = merged["code"].map(name_map["market"]).fillna(merged["market"])
    merged = save_history(merged)

    if idx_rows:
        idx_hist = load_index_history()
        new_idx = pd.DataFrame(idx_rows)
        idx_hist = idx_hist[~idx_hist["date"].isin(new_idx["date"])]
        save_index_history(pd.concat([idx_hist, new_idx], ignore_index=True))

    return enrich_and_write(merged)


def enrich_and_write(merged: pd.DataFrame, idx_hist: pd.DataFrame | None = None):
    """
    在 compute() 的基礎上補上動能篩選、市場廣度、題材、上榜天數, 然後寫檔。

    抽成獨立函式是為了讓 --recompute 走同一條路 —— 改了計算邏輯又還沒到下一個
    交易日時, daily_update 會直接跳過(沒有缺口要補), 網站就看不到新欄位。
    """
    result = compute(merged, idx_hist)

    # 動能篩選 + 市場廣度。發行股數每天更新一次(公司會增減資)。
    try:
        shares = fetch_shares()
        save_shares(shares)
    except Exception as e:
        print(f"發行股數抓取失敗({e.__class__.__name__}),沿用既有的。")
        shares = load_shares()
    idx = idx_hist if idx_hist is not None else load_index_history()
    picks = momentum_screen(merged, shares, idx=idx)
    breadth, members = update_breadth(merged, shares, idx=idx)
    recs = [{k: (None if pd.isna(v) else v) for k, v in r.items()}
            for r in picks.to_dict("records")]

    # 從新聞標題判斷題材。純關鍵字比對, 抓不到就留白 —— 標題會一起輸出,
    # 歸不了類時可以直接翻。失敗不影響前面算好的東西。
    themes = None
    try:
        try:
            import themes            # 直接跑 python scripts/update_data.py
        except ModuleNotFoundError:
            from scripts import themes   # 被當成套件匯入(測試/其他腳本)
        themes.annotate(recs)
        result["themes"] = themes.summarize(recs)
    except Exception as e:
        print(f"題材判斷失敗({e.__class__.__name__}),略過。")
        result["themes"] = []

    # 上榜天數。用有紀錄的交易日序列, 不是日曆天 —— 隔週末不算中斷。
    mem_dates = sorted(members["date"].unique())
    st = streaks(members, mem_dates)
    for r in recs:
        run, since, total = st.get(r["code"], (1, result["trade_date"], 1))
        r["days"] = run            # 連續上榜幾個交易日
        r["since"] = since         # 這一段從哪天開始
        r["days_total"] = total    # 有紀錄以來累計上榜幾天
    result["member_days"] = len(mem_dates)

    # 把今天判斷出來的題材寫回名單, 之後才有「上一個交易日的族群分布」可比。
    # 覆寫檔當後備 —— 昨天在榜、今天掉出去的個股今天不會被標註, 沒有標籤就
    # 全部算進「未歸類」, 差值會失真。
    try:
        ov = themes.load_overrides() if themes else {}
    except Exception:
        ov = {}
    today_theme = {**ov, **{r["code"]: r.get("theme") for r in recs if r.get("theme")}}
    if mem_dates:
        cur = members["date"] == mem_dates[-1]
        members.loc[cur, "theme"] = members.loc[cur, "code"].map(today_theme)
        members = save_members(members)

    # 族群對前一個交易日的增減
    prev_counts = theme_delta(members, mem_dates, today_theme, forced=ov)
    if prev_counts:
        for g in result["themes"]:
            g["prev"] = prev_counts.get(g["theme"], 0)
            g["delta"] = g["count"] - g["prev"]

    result["momentum"] = recs
    result["breadth"] = breadth.tail(120).to_dict("records")

    write_result(result)
    print(f"完成:{result['trade_date']} | 全市場 {result['universe']} 檔 | "
          f"SEPA {len(result['sepa'])} 檔 | 當日強勢 {len(result['daily'])} 檔 | "
          f"動能 {len(result['momentum'])} 檔 | "
          f"上榜天數自 {mem_dates[0] if mem_dates else '—'} 起算")
    return True


def rebuild_breadth():
    """
    用**目前的**篩選條件, 把廣度與上榜名單整段重算。

    改了篩選參數(門檻、天數…)之後要跑這個 —— 否則 breadth.csv 前面是舊定義、
    之後是新定義, 那條線會在改的那天出現一個不存在的斷層, 20MA、上榜天數、族群
    日增減也全部跟著錯。

    每一天只用「那天往前 KEEP_DAYS 個交易日」的資料去篩, 跟線上流水線當天看得
    到的一樣。有 data/history_long.csv.gz(回測用的加長歷史)就用它, 能重算滿
    KEEP_DAYS 天; 沒有的話只能從 280 天的 history.csv.gz 重算最後約 80 天。
    同一天同一檔原本標好的題材會保留。另外把全部可算日的名單存成回測/研究用的
    快取(檔名帶 SCREEN_SIG)。
    """
    long_hist = ROOT / "data" / "history_long.csv.gz"
    long_idx = ROOT / "data" / "index_long.csv.gz"
    hist = load_history()
    idx = load_index_history()
    if long_hist.exists():
        hist = pd.concat([pd.read_csv(long_hist, dtype={"code": str}), hist], ignore_index=True)
        hist = hist.drop_duplicates(subset=["date", "code"], keep="last")
        hist.sort_values(["date", "code"]).to_csv(long_hist, index=False, compression="gzip")
    if long_idx.exists():
        idx = pd.concat([pd.read_csv(long_idx), idx], ignore_index=True)
        idx = idx.dropna(subset=["close"]).drop_duplicates("date", keep="last").sort_values("date")
        idx.to_csv(long_idx, index=False, compression="gzip")

    shares = load_shares()
    dates = sorted(hist["date"].unique())
    usable = list(range(199, len(dates)))
    print(f"重算 {len(usable)} 天 {dates[usable[0]]} ~ {dates[-1]} | 條件 {SCREEN_SIG}", flush=True)

    by_date = {d: g for d, g in hist.groupby("date")}
    members, rows = {}, []
    t0 = time.time()
    for n, i in enumerate(usable, 1):
        d = dates[i]
        window = dates[max(0, i - (KEEP_DAYS - 1)): i + 1]
        picks = momentum_screen(pd.concat([by_date[x] for x in window], ignore_index=True),
                                shares, idx=idx)
        codes = set(picks["code"]) if len(picks) else set()
        members[d] = codes
        uni = len(by_date[d])
        rows.append({"date": d, "count": len(codes), "universe": uni,
                     "pct": round(len(codes) / uni * 100, 2) if uni else 0})
        if n % 50 == 0:
            left = (time.time() - t0) / n * (len(usable) - n) / 60
            print(f"  {n}/{len(usable)} {d}: {len(codes)} 檔 (剩約 {left:.0f} 分)", flush=True)

    cache = ROOT / "data" / f"_bt_members_{SCREEN_SIG}.csv.gz"
    pd.DataFrame([{"date": d, "code": c} for d, cs in members.items() for c in cs]) \
      .to_csv(cache, index=False, compression="gzip")

    # 取代掉整份廣度(不是合併) —— 合併的話重算不到的舊日子會留下舊定義
    BREADTH_PATH.unlink(missing_ok=True)
    save_breadth(pd.DataFrame(rows))

    old = load_members()
    old_theme = {(r.date, r.code): r.theme for r in old.itertuples()
                 if isinstance(r.theme, str) and r.theme}
    keep = set(sorted(members)[-KEEP_DAYS:])
    m = save_members(pd.DataFrame(
        [{"date": d, "code": c, "theme": old_theme.get((d, c))}
         for d, cs in members.items() if d in keep for c in cs],
        columns=["date", "code", "theme"]))
    print(f"完成: breadth {len(load_breadth())} 天, 名單 {m['date'].nunique()} 天"
          f"(保留題材 {m['theme'].notna().sum()} 筆), 回測快取 {len(members)} 天 -> {cache.name}")


def write_result(result: dict):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False),
                           encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true",
                    help="回補約 280 個交易日的歷史資料(證交所+櫃買中心,免金鑰)")
    ap.add_argument("--recompute", action="store_true",
                    help="不抓新資料, 用現有歷史重算一次 latest.json。"
                         "改了計算邏輯又還沒到下一個交易日時用 —— 平常那條路"
                         "在沒有新交易日時會直接跳過, 不會重新產出。")
    ap.add_argument("--rebuild-breadth", action="store_true",
                    help="改了篩選參數之後用: 以目前條件整段重算廣度與上榜名單, "
                         "再重產 latest.json")
    args = ap.parse_args()

    if args.rebuild_breadth:
        rebuild_breadth()
        enrich_and_write(load_history(), load_index_history())
    elif args.backfill:
        backfill()
        enrich_and_write(load_history(), load_index_history())   # 回補完直接算一次
    elif args.recompute:
        enrich_and_write(load_history(), load_index_history())
    else:
        daily_update()
