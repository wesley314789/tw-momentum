#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
themes.py — 從新聞標題判斷個股的族群題材

做法很單純:對每檔股票抓 Google News RSS 的標題, 用關鍵字比對歸類。台股的
題材詞相當標準(CPO、磷化銦、散熱、軍工、記憶體…), 所以光靠標題就抓得到
大部分。

刻意不用 LLM:這支要跑在 GitHub Actions 的排程裡, 保持免費且無外部相依。
代價是**新題材抓不到**(詞典裡沒有的詞就是沒有), 也無法判斷某則新聞是不是
真的在講上漲原因。所以原始標題會一併存下來 —— 歸不了類的時候, 人可以直接
翻標題自己判斷, 也可以之後把這批標題餵給 LLM 做更好的分類。

被列處置股/注意股會另外標記:那代表短期漲太兇被盯上, 是風險訊號而不是題材。
"""
import csv
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

# 人工(或排程的 Claude 任務)研究出來的題材, 優先於關鍵字比對。
# 欄位: code, theme, source, updated —— source 記下判斷依據, 方便事後回頭驗證。
OVERRIDES_PATH = Path(__file__).resolve().parent.parent / "data" / "theme_overrides.csv"

RSS = "https://news.google.com/rss/search?q={}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
HEADERS = {"User-Agent": "Mozilla/5.0"}
WORKERS = 4
MAX_ITEMS = 12         # 每檔取幾則標題。Google News 每次回傳的結果會變, 取多
                       # 一點才穩 —— 只取 6 則時同一檔前後兩次會歸到不同題材
MIN_HITS = 2           # 要有幾「則」不同的新聞提到才算, 不是關鍵字出現幾次:
                       # 用次數計分會被單一則大量堆疊關鍵字的新聞主導

# 題材關鍵字。順序不影響, 分數高者勝。想新增題材直接加一列即可。
THEMES = {
    "CPO・光通訊": ["CPO", "光通訊", "矽光子", "光收發", "磷化銦", "InP",
                   "共同封裝", "光模組", "矽光", "光引擎", "雷射二極體"],
    "PCB・CCL・載板": ["PCB", "銅箔基板", "CCL", "ABF", "載板", "玻纖",
                      "印刷電路板", "軟板", "銅箔", "鑽針"],
    "AI伺服器散熱・機構": ["散熱", "液冷", "水冷", "均熱", "熱管", "MCCP",
                        "冷板", "滑軌", "導軌", "機殼", "軌道", "機構件"],
    "記憶體": ["記憶體", "DRAM", "HBM", "NAND", "SSD", "快閃", "顆粒",
              "利基型記憶體"],
    "AI伺服器": ["AI伺服器", "GB200", "GB300", "輝達", "NVIDIA", "機櫃",
                "資料中心", "AI 伺服器"],
    "軍工・國防": ["軍工", "國防", "無人機", "軍用", "航太", "飛彈"],
    "航運": ["航運", "貨櫃", "SCFI", "運價", "散裝", "BDI", "海運", "船東"],
    "生技・醫材": ["生技", "新藥", "臨床", "FDA", "醫材", "藥證", "授權金",
                  "解盲"],
    "重電・電力": ["重電", "電網", "變壓器", "電力", "儲能", "太陽能"],
    "半導體材料・設備": ["矽晶圓", "磊晶", "晶圓代工", "先進封裝", "光罩",
                       "半導體設備", "CoWoS", "潔淨", "擴廠", "耗材"],
    "被動元件": ["被動元件", "MLCC", "電阻", "電感"],
    "機器人": ["機器人", "人形", "自動化"],
    "IC設計": ["IC設計", "ASIC", "MCU", "矽智財", "IP"],
}

# 風險旗標:漲太兇被交易所盯上, 是警訊不是題材
FLAGS = {
    "處置股": ["處置", "分盤交易", "分盤撮合"],
    "注意股": ["注意股", "注意交易資訊"],
}


def fetch_headlines(code: str, name: str) -> list[str]:
    """抓單檔的新聞標題。失敗回空 list, 不讓單檔拖垮整批。"""
    try:
        q = urllib.parse.quote(f"{name} {code}")
        r = requests.get(RSS.format(q), headers=HEADERS, timeout=20)
        r.raise_for_status()
        items = ET.fromstring(r.content).findall(".//item")[:MAX_ITEMS]
        return [(it.find("title").text or "").strip() for it in items]
    except Exception:
        return []


def fetch_all(stocks: list[tuple]) -> dict:
    """stocks = [(code, name), ...] -> {code: [標題, ...]}"""
    out = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for (code, _), heads in zip(
                stocks, ex.map(lambda s: fetch_headlines(*s), stocks)):
            out[code] = heads
    return out


def classify(headlines: list[str]) -> tuple[str | None, list[str]]:
    """
    回傳 (題材, 旗標)。分數是「有幾則不同的新聞提到這個題材」, 不是關鍵字
    總出現次數 —— 否則一則標題塞滿同義詞就能決定結果。不足 MIN_HITS 就回
    None, 寧可留白讓人翻標題, 也不硬歸類。
    """
    scores = {}
    for theme, words in THEMES.items():
        n = sum(1 for h in headlines if any(w in h for w in words))
        if n:
            scores[theme] = n
    blob = " ".join(headlines)
    flags = [f for f, words in FLAGS.items() if any(w in blob for w in words)]
    if not scores:
        return None, flags
    theme, n = max(scores.items(), key=lambda kv: kv[1])
    return (theme if n >= MIN_HITS else None), flags


def load_overrides() -> dict:
    """讀人工/排程研究出來的題材對照表。檔案不存在就當空的。"""
    if not OVERRIDES_PATH.exists():
        return {}
    out = {}
    with OVERRIDES_PATH.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            code = (row.get("code") or "").strip()
            theme = (row.get("theme") or "").strip()
            if code and theme:
                out[code] = theme
    return out


def annotate(picks: list[dict]) -> list[dict]:
    """
    對篩選結果標上題材與旗標, 並附上前三則標題供人工覆核。
    picks 需含 code / name 欄位, 就地加上 theme / flags / news。

    theme_overrides.csv 裡有的個股直接採用該題材(關鍵字比對抓不到新題材,
    所以留一條人工/LLM 補強的路), 並標 theme_src 讓前端能區分來源。
    """
    if not picks:
        return picks
    ov = load_overrides()
    news = fetch_all([(p["code"], p["name"]) for p in picks])
    for p in picks:
        heads = news.get(p["code"], [])
        theme, flags = classify(heads)
        if p["code"] in ov:
            p["theme"] = ov[p["code"]]
            p["theme_src"] = "override"
        else:
            p["theme"] = theme
            p["theme_src"] = "keyword" if theme else None
        p["flags"] = flags
        p["news"] = heads[:3]
    return picks


def summarize(picks: list[dict]) -> list[dict]:
    """依題材彙總:檔數、成交值合計、平均漲幅。未歸類的單獨一組。"""
    groups = {}
    for p in picks:
        g = groups.setdefault(p.get("theme") or "未歸類",
                              {"theme": p.get("theme") or "未歸類",
                               "count": 0, "value": 0.0, "perf": 0.0,
                               "names": []})
        g["count"] += 1
        g["value"] += p.get("value") or 0
        g["perf"] += p.get("perf_1m") or 0
        g["names"].append(p["name"])
    out = []
    for g in groups.values():
        g["value"] = round(g["value"], 1)
        g["perf"] = round(g["perf"] / g["count"], 1)
        g["names"] = g["names"][:8]
        out.append(g)
    # 未歸類永遠排最後, 其餘依檔數
    return sorted(out, key=lambda g: (g["theme"] == "未歸類", -g["count"]))
