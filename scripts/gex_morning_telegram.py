#!/usr/bin/env python3
"""Send one fresh TXO GEX preopen snapshot to Telegram each Taipei morning.

All displayed levels are converted from the option-forward strike coordinate to
TX futures points with the most recently observed settlement basis.
"""

import argparse
import datetime as dt
import json
import math
import os
import time
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent.parent
LATEST = ROOT / "docs" / "data" / "gex_latest.json"
STATE = ROOT / "data" / "gex_telegram_state.json"
TAIPEI = ZoneInfo("Asia/Taipei")


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def check_snapshot(data, state, today):
    """Only a new official night close and matching preopen calculation qualify."""
    preopen_day = data.get("preopen_date")
    if preopen_day != today.isoformat():
        return {"status": "skip", "reason": "今天沒有新的盤前快照",
                "preopen_date": preopen_day}
    try:
        oi_day = dt.date.fromisoformat(data["date"])
    except (KeyError, TypeError, ValueError):
        return {"status": "error", "reason": "OI 日期無效"}
    if oi_day >= today:
        return {"status": "error", "reason": "盤前快照的 OI 日期不合理"}
    if number(data.get("preopen_price")) is None or number(data.get("preopen_gex")) is None:
        return {"status": "error", "reason": "盤前價格或 GEX 缺漏"}
    settlement, forward = number(data.get("txf_settle")), number(data.get("F"))
    if settlement is None or forward is None:
        return {"status": "error", "reason": "缺少換算台指期點位的基差"}
    if data.get("preopen_regime") not in ("positive", "negative", "neutral"):
        return {"status": "error", "reason": "盤前 Gamma 狀態無效"}
    if state.get("last_sent_preopen_date") == preopen_day:
        return {"status": "quiet", "reason": "今天已發送過", "preopen_date": preopen_day}
    return {"status": "ready", "preopen_date": preopen_day,
            "oi_date": data["date"], "price": number(data["preopen_price"])}


def format_message(data, *, demo=False):
    """Separate direction-free concentration from the signed dealer model."""
    basis = float(data["txf_settle"]) - float(data["F"])
    now = dt.datetime.now(TAIPEI)
    delivery = now.strftime("%H:%M")
    timing = "" if now.time() < dt.time(8, 45) else "（延遲送達）"

    def level(key):
        value = number(data.get(key))
        return f"{value + basis:,.0f}" if value is not None else "—"

    regime_key = data["regime"] if demo else data["preopen_regime"]
    regime = {"positive": "正 Gamma", "negative": "負 Gamma",
              "neutral": "中性／符號不可靠"}[regime_key]
    ratio = number(data.get("net_ratio" if demo else "preopen_net_ratio"))
    ratio_text = f"{ratio * 100:.1f}%" if ratio is not None else "—"
    if demo:
        close = number(data.get("txf_close"))
        if close is None:
            raise ValueError("缺少台指期日盤收盤價，無法發送測試訊息")
        title = "🧪 台指期 GEX 測試推播"
        date_line = f"資料日期：{data['date']} 收盤資料"
        price_line = f"台指期收盤 {close:,.0f} 點"
        footer = "這是收盤資料測試訊息，不會占用明早的盤前通知。"
    else:
        title = "🌅 台指期 GEX 盤前快照"
        date_line = f"快照日期：{data['preopen_date']}"
        price_line = f"夜盤收盤 {float(data['preopen_price']):,.0f} 點"
        footer = "盤前狀態依夜盤價格重算；位階仍是上一交易日的 OI 結構。"
    return "\n".join([
        title,
        date_line,
        f"送出時間：{delivery}{'' if demo else timing}",
        price_line,
        f"OI / IV：{data['date']} 收盤",
        "以下位階均換算為台指期點位（沿用前一收盤日基差）。",
        "",
        "📍 Gamma Concentration（不假設造市商持倉方向）",
        f"Call 集中 {level('call_concentration_peak')}",
        f"Put 集中 {level('put_concentration_peak')}",
        f"最強聚集 {level('gamma_cluster_strike')}",
        f"集中峰 {level('concentration_peak')}",
        f"集中谷 {level('concentration_valley')}",
        "",
        "⚖️ Signed GEX（造市商方向假設）",
        f"Call Wall {level('call_wall')}",
        f"Put Wall {level('put_wall')}",
        f"山頂 {level('peak')}",
        f"山谷 {level('valley')}",
        f"Micro Flip {level('micro_flip')}",
        f"Macro Zero {level('macro_zero')}",
        f"{'收盤' if demo else '盤前'}狀態：{regime}",
        f"淨／總 Gamma：{ratio_text}",
        "",
        footer,
        "此模型不是方向預測或交易指令。",
    ])


def send_message(token, chat_id, message):
    import requests

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": message}, timeout=20)
        result = response.json()
    except (requests.RequestException, ValueError):
        raise RuntimeError("Telegram 連線或回應失敗") from None
    if response.status_code != 200 or not result.get("ok"):
        raise RuntimeError(f"Telegram 拒絕訊息（HTTP {response.status_code}）")


def save_state(path, preopen_date, oi_date):
    payload = {"last_sent_preopen_date": preopen_date,
               "last_sent_oi_date": oi_date}
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def seconds_until_eight(now):
    target = now.replace(hour=8, minute=0, second=0, microsecond=0)
    return max(0, (target - now).total_seconds())


def wait_until_eight():
    delay = seconds_until_eight(dt.datetime.now(TAIPEI))
    if delay > 0:
        time.sleep(delay)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--send", action="store_true", help="正式發送，預設只預覽")
    mode.add_argument("--demo", action="store_true", help="立即發送收盤資料測試訊息，不記錄為盤前通知")
    args = parser.parse_args()
    try:
        data = json.loads(LATEST.read_text(encoding="utf-8"))
        if args.demo:
            token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
            if not token or not chat_id:
                result = {"status": "error", "reason": "Telegram Secrets 尚未設定"}
            else:
                message = format_message(data, demo=True)
                send_message(token, chat_id, message)
                result = {"status": "demo_sent", "date": data["date"]}
        else:
            state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
            today = dt.datetime.now(TAIPEI).date()
            result = check_snapshot(data, state, today)
            if args.send and result["status"] == "ready":
                token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
                chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
                if not token or not chat_id:
                    result = {"status": "error", "reason": "Telegram Secrets 尚未設定"}
                else:
                    wait_until_eight()
                    send_message(token, chat_id, format_message(data))
                    save_state(STATE, data["preopen_date"], data["date"])
                    result["status"] = "sent"
    except Exception as exc:
        # send_message only raises credential-free errors.
        result = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 1 if result["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
