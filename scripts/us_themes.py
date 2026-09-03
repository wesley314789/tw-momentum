#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
us_themes.py — 美股動能清單的族群題材

跟台股 themes.py 共用同一套引擎(抓 Google News、依「有幾則不同新聞提到」計分、
人工覆寫優先), 但**關鍵字比對在美股的效果遠差於台股**, 這點必須講清楚:

  實測 2026-09-01 的 219 檔動能股, 純關鍵字只歸出 42%, 而且**成交值最大的名字
  幾乎全部落在未歸類** —— SNDK、CRM、PLTR、NOW、LITE、MRNA、HOOD、SHOP…
  原因是美股的 RSS 被財經內容農場洗版。把 219 檔的標題做詞頻統計, 出現在最多
  檔的詞是 stocktitan.net、quiver quantitative、gurufocus、insider sells、
  earnings estimates —— 前 70 名裡一個題材詞都沒有。台股的標題是題材密集的
  (CPO、磷化銦、散熱、軍工), 美股不是。

所以美股這邊的分工跟台股相反:

  * **人工/LLM 研究(us_theme_overrides.csv)是主力** —— 大型股靠它
  * **關鍵字是補位** —— 抓那些新聞本來就很直白的中小型股(礦業、生技解盲)

另外多吃 RSS 的 <description> 摘要而不只是標題, 文字量多一倍, 對關鍵字有幫助。
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import themes as T   # 共用引擎: fetch_many / strip_name / compile_lexicon / score / summarize

OVERRIDES_PATH = Path(__file__).resolve().parent.parent / "data" / "us_theme_overrides.csv"
RSS = "https://news.google.com/rss/search?q={}&hl=en-US&gl=US&ceid=US:en"
MAX_ITEMS = 14
MIN_HITS = 2
WORKERS = 4

# Nasdaq 的 name 帶著證券類別樣板, 查新聞前要拿掉(與 us_breadth.clean_name 同義,
# 這裡重寫一份是為了讓本檔可以獨立使用, 不強迫匯入整個 breadth pipeline)
SEC_TYPE = re.compile(
    r"[,\s]*\b(class\s+[a-c]\b|series\s+[a-z]\b|"
    r"common\s+stock|common\s+shares?|ordinary\s+shares?|"
    r"subordinate\s+voting\s+shares?|"
    r"american\s+depositary\s+shares?|depositary\s+shares?|"
    r"perpetual\b|units?\b|warrants?\b|ads\b|adr\b).*$", re.I)


def clean_name(name: str) -> str:
    n = SEC_TYPE.sub("", (name or "").strip())
    return re.sub(r"\s+", " ", n).strip(" ,.") or (name or "")


# 關鍵字。字尾 `*` = 前綴比對(crypto* 也吃 cryptocurrency)。英文一律加詞界,
# 否則 "ai" 會命中 "said"、"ev" 會命中 "seven"。
THEMES = {
    # 2026-08 這波最大的一群。Salesforce 財報帶頭, 「SaaSpocalypse」(AI 會吃掉
    # SaaS)的敘事反轉, 資金從「花錢建 AI」轉向「用 AI 賺錢」的既有軟體商。
    "Enterprise software": [
        "saas", "software", "subscription", "arr", "agentic", "copilot",
        "enterprise ai", "cloud revenue", "seat*", "observability", "devops",
        "database", "crm", "erp", "saaspocalypse", "ai monetization",
    ],
    "Cybersecurity": [
        "cybersecurity", "cyber", "ransomware", "data breach", "zero trust",
        "firewall", "siem", "identity security", "endpoint",
    ],
    # 金 $4,400、銀 $70。美國財政部加倍長債回購 -> 財政恐慌交易; 全產業
    # AISC 低於 $2,000/oz, 營業利益率是史上最寬的一次。
    "Gold & silver": [
        "gold", "silver", "bullion", "precious metal*", "per ounce", "aisc",
        "royalt*", "mine*", "mining", "ore", "assay", "drill result*",
    ],
    "Critical minerals": [
        "rare earth*", "lithium", "copper", "critical mineral*", "cobalt",
        "tungsten", "coal", "metallurgical", "coking",
    ],
    # AI 推論吃儲存, NAND 報價單季 +75~100%。SNDK 2026 上半年 +726%,
    # 是 S&P 500 第一名。
    "Memory/NAND": [
        "nand", "dram", "hbm", "memory chip*", "flash memory", "micron",
        "memory pricing", "storage demand", "ssd",
    ],
    "AI datacenter": [
        "ai data cent*", "data cent*", "hyperscaler*", "nvidia", "blackwell",
        "rubin", "ai chip*", "ai infrastructure", "ai capex", "inference",
        "gpu", "rack*", "liquid cooling",
    ],
    "Optical/networking": [
        "optical", "transceiver*", "silicon photonic*", "co-packaged", "cpo",
        "800g", "1.6t", "coherent optic*", "datacom", "laser*",
    ],
    # BTC 8 月 $60k -> $80k; SEC 把 payment stablecoin 獨立分類, Circle 的
    # 發行模式風險大降。這群包含拿公司資產去買幣的「treasury」股。
    "Crypto/digital assets": [
        "bitcoin", "ethereum", "crypto*", "stablecoin", "blockchain",
        "coinbase", "digital asset*", "btc", "usdc", "eth holdings",
        "treasury company", "mstr",
    ],
    "Fintech/brokerage": [
        "fintech", "brokerage", "neobank", "payment volume", "trading app",
        "prediction market*", "retail trading",
    ],
    # 8/19 Moderna + Merck 的 mRNA 個人化癌症疫苗三期解盲成功(史上第一次),
    # 單日 +177%。整個腫瘤/疫苗族群跟著動。
    "Biotech readout": [
        "phase 3", "phase 2", "phase iii", "topline", "readout", "endpoint",
        "clinical trial", "pdufa", "fda approval", "breakthrough therapy",
        "oncology", "melanoma", "mrna", "cancer vaccine", "orphan drug",
    ],
    "Medtech/diagnostics": [
        "medtech", "medical device", "diagnostic*", "assay", "screening",
        "510(k)", "ce mark", "genomic*", "sequencing",
    ],
    "Energy services": [
        "oilfield", "drilling", "rig count", "frac*", "completion*", "shale",
        "offshore", "wireline", "pressure pumping",
    ],
    "Oil & gas": [
        "crude", "wti", "brent", "opec", "natural gas", "lng", "permian",
        "production guidance", "acreage",
    ],
    "Nuclear/uranium": [
        "nuclear", "uranium", "smr", "modular reactor", "enrichment",
        "reactor",
    ],
    "Power/grid": [
        "grid", "electricity", "transformer", "power demand", "megawatt",
        "turbine", "utility rate*", "interconnection",
    ],
    "Defense/gov AI": [
        "defense", "defence", "pentagon", "drone*", "missile", "munition*",
        "warfighter", "army", "navy", "air force", "government contract",
        "classified", "nato",
    ],
    "Space": ["satellite*", "orbit*", "rocket", "spacecraft", "launch site"],
    "Quantum": ["quantum comput*", "qubit*", "quantum advantage"],
    "Robotics": ["robot*", "humanoid", "warehouse automation"],
    "M&A/corporate action": [
        "acquisition", "acquire*", "merger", "takeover", "buyout", "activist",
        "tender offer", "go private", "spin-off", "strategic alternatives",
        "s&p 500", "index inclusion", "stake in",
    ],
    "Consumer/retail": [
        "same-store", "comparable sales", "apparel", "footwear", "beauty",
        "grocery", "restaurant*", "foot traffic",
    ],
}

# 平手時的優先序。原則跟台股一樣:事件型 > 具體供應鏈 > 籠統分類。
# "Enterprise software" 和 "AI datacenter" 是這份清單裡最容易被誤吸的兩個
# catch-all(幾乎每檔科技股的新聞都會提到 AI 或 software), 所以壓低。
TIE_BREAK = {
    "M&A/corporate action": 2,
    "Biotech readout": 1,
    "Memory/NAND": 1,
    "Optical/networking": 1,
    "Crypto/digital assets": 1,
    "Enterprise software": -1,
    "AI datacenter": -1,
}

# Google News 的標題結尾固定是「標題 - 媒體名」, 而媒體名會製造假命中 ——
# AbCellera(生技)被判成加密貨幣, 只因為其中幾則的來源叫 CryptoRank。
PUBLISHER = re.compile(r"\s+-\s+[^-]{2,40}$")


def scrub(headlines: list[str], name: str, symbol: str) -> list[str]:
    """
    比對前的清理:去掉媒體名、公司名、股票代號。

    代號一定要拿掉, 否則它會在幾乎每一則標題裡出現並撞到題材關鍵字 ——
    Hudbay Minerals 的代號就是 HBM, 12 則標題全部命中「記憶體 HBM」, 一檔
    銅礦公司被歸成記憶體股。同理 P、U、DC、AG、OR、IT、GO 這些短代號。
    """
    out = []
    for h in headlines:
        h = PUBLISHER.sub("", h)
        h = re.sub(r"[($]?\b" + re.escape(symbol) + r"\b\)?", " ", h)
        out.append(h)
    return T.strip_name(out, name) if name else out


_LEX = T.compile_lexicon(THEMES, boundary=True)


def classify(headlines: list[str], name: str = "", symbol: str = "") -> str | None:
    """清掉媒體名/公司名/代號後計分。回傳題材或 None。"""
    theme, _ = T.score(scrub(headlines, name, symbol), _LEX, TIE_BREAK,
                       min_hits=MIN_HITS)
    return theme


def annotate(picks: list[dict]) -> list[dict]:
    """
    就地加上 theme / theme_src / news。picks 需含 symbol / name。
    覆寫檔優先 —— 美股大型股的新聞噪音太大, 關鍵字幾乎抓不到, 主力在覆寫檔。
    """
    if not picks:
        return picks
    ov = T.load_overrides(OVERRIDES_PATH)
    queries = {p["symbol"]: f'{clean_name(p["name"])} ({p["symbol"]}) stock'
               for p in picks}
    # 不開 with_desc:實測 3066 則 Google News 摘要, 平均 13.7 個詞, 其中標題
    # 沒有的只有 1.0 個(7%) —— 摘要基本上就是標題重複一次再接媒體名, 沒有新
    # 資訊, 反而讓媒體名多命中一次。要拿到標題以外的東西, 得去抓文章本文,
    # 那是 us_theme_overrides.csv(人工/LLM 研究)那條路在做的事。
    news = T.fetch_many(queries, rss=RSS, max_items=MAX_ITEMS, workers=WORKERS)
    for p in picks:
        heads = news.get(p["symbol"], [])
        theme = classify(heads, clean_name(p.get("name", "")), p["symbol"])
        if p["symbol"] in ov:
            p["theme"], p["theme_src"] = ov[p["symbol"]], "override"
        else:
            p["theme"], p["theme_src"] = theme, ("keyword" if theme else None)
        p["news"] = heads[:3]
    return picks


def summarize(picks: list[dict]) -> list[dict]:
    return T.summarize(picks, value_key="turnover")
