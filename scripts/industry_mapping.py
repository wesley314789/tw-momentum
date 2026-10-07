"""交易所公告的上市／上櫃產業別，供 Sector Breadth 使用。"""

from pathlib import Path

import pandas as pd


PATH = Path(__file__).resolve().parent.parent / "data" / "sector_mapping.csv"
TWSE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
TPEX_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"

# 證交所與櫃買中心行情規格書的產業別代碼表；32-34 為櫃買類別。
INDUSTRY_NAMES = {
    "01": "水泥工業", "02": "食品工業", "03": "塑膠工業", "04": "紡織纖維",
    "05": "電機機械", "06": "電器電纜", "08": "玻璃陶瓷", "09": "造紙工業",
    "10": "鋼鐵工業", "11": "橡膠工業", "12": "汽車工業", "14": "建材營造",
    "15": "航運業", "16": "觀光餐旅", "17": "金融保險", "18": "貿易百貨",
    "20": "其他", "21": "化學工業", "22": "生技醫療業", "23": "油電燃氣業",
    "24": "半導體業", "25": "電腦及週邊設備業", "26": "光電業",
    "27": "通信網路業", "28": "電子零組件業", "29": "電子通路業",
    "30": "資訊服務業", "31": "其他電子業", "32": "文化創意業",
    "33": "農業科技業", "34": "電子商務", "35": "綠能環保", "36": "數位雲端",
    "37": "運動休閒", "38": "居家生活",
}


def _rows(payload, *, market, code_key, industry_key):
    if not isinstance(payload, list):
        raise ValueError(f"{market} 公司基本資料不是列表")
    rows = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        code = str(item.get(code_key, "")).strip()
        industry_code = str(item.get(industry_key, "")).strip().zfill(2)
        if (len(code) != 4 or not code.isdigit() or code.startswith("00")
                or industry_code not in INDUSTRY_NAMES):
            continue
        rows.append({"code": code, "market": market,
                     "industry_code": industry_code,
                     "sector": INDUSTRY_NAMES[industry_code]})
    return rows


def build(twse_payload, tpex_payload):
    """兩市資料必須同時完整，才可取代舊快照。代碼 91(TDR)無產業別，略過。"""
    listed = _rows(twse_payload, market="上市", code_key="公司代號", industry_key="產業別")
    otc = _rows(tpex_payload, market="上櫃", code_key="SecuritiesCompanyCode",
                industry_key="SecuritiesIndustryCode")
    if len(listed) < 800 or len(otc) < 700:
        raise ValueError(f"官方產業對照不完整：上市 {len(listed)}、上櫃 {len(otc)}")
    df = pd.DataFrame(listed + otc)
    if df["code"].duplicated().any():
        raise ValueError("官方產業對照出現重複股票代碼")
    return df.sort_values("code").reset_index(drop=True)


def load(path=PATH):
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"code": str, "industry_code": str})
    return dict(zip(df["code"], df["sector"]))


def save(df, path=PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)
