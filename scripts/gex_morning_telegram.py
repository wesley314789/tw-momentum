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
LEVEL_ROWS = (
    ("call_concentration_peak", "Call 集中", "📍"),
    ("put_concentration_peak", "Put 集中", "📍"),
    ("gamma_cluster_strike", "最強聚集", "📍"),
    ("concentration_peak", "集中峰", "📍"),
    ("concentration_valley", "集中谷", "📍"),
    ("call_wall", "Call Wall", "⚖️"),
    ("put_wall", "Put Wall", "⚖️"),
    ("peak", "山頂", "⚖️"),
    ("valley", "山谷", "⚖️"),
    ("micro_flip", "Micro Flip", "⚖️"),
    ("macro_zero", "Macro Zero", "⚖️"),
)


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def check_snapshot(data, state, today):
    """Require the latest valid night session assigned to today's day market."""
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
    # 週五夜盤歸屬週一交易日，週一早上仍是最近的盤前價格；其他
    # 跨休市日維持跳過，避免把連假前的舊價誤當新夜盤。
    monday_after_friday = (today.weekday() == 0 and oi_day.weekday() == 4
                           and today - oi_day == dt.timedelta(days=3))
    if today - oi_day != dt.timedelta(days=1) and not monday_after_friday:
        return {"status": "skip", "reason": "夜盤跨休市日，沒有新的盤前行情",
                "oi_date": data["date"], "preopen_date": preopen_day}
    session_end = (oi_day + dt.timedelta(days=1)).isoformat()
    if (data.get("preopen_session_start", data["date"]) != data["date"] or
            data.get("preopen_session_end", session_end) != session_end):
        return {"status": "error", "reason": "夜盤實際時段與 OI 日期不一致"}
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


def ranked_levels(data, *, demo, basis):
    """One descending futures-point ladder with the observed price inserted."""
    anchor = number(data.get("txf_close" if demo else "preopen_price"))
    if anchor is None:
        raise ValueError("缺少台指期價格，無法排列 GEX 位階")
    rows = [(anchor, -1, "☀️ 日盤收盤" if demo else "🌙 夜盤收盤")]
    if not demo:
        previous_close = number(data.get("txf_close"))
        if previous_close is not None:
            rows.append((previous_close, -2, "☀️ 前次日盤收盤"))
    missing = []
    for index, (key, label, model) in enumerate(LEVEL_ROWS):
        raw = number(data.get(key))
        if raw is None:
            missing.append(f"{model} {label} —")
        else:
            rows.append((raw + basis, index, f"{model} {label}"))
    rows.sort(key=lambda row: (-row[0], row[1]))
    return [f"{price:,.0f}  {label}" for price, _, label in rows], missing


def format_message(data, *, demo=False):
    """Rank all levels while tagging the two models separately."""
    basis = float(data["txf_settle"]) - float(data["F"])
    now = dt.datetime.now(TAIPEI)
    delivery = now.strftime("%H:%M")
    timing = "" if now.time() < dt.time(8, 45) else "（延遲送達）"

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
        oi_day = dt.date.fromisoformat(data["date"])
        session_end = data.get("preopen_session_end", (oi_day + dt.timedelta(days=1)).isoformat())
        date_line = f"盤前交易日：{data['preopen_date']}"
        price_line = f"夜盤收盤 {float(data['preopen_price']):,.0f} 點"
        footer = "盤前狀態依夜盤價格重算；位階仍是上一交易日的 OI 結構。"
    ladder, missing = ranked_levels(data, demo=demo, basis=basis)
    return "\n".join([
        title,
        date_line,
        f"送出時間：{delivery}{'' if demo else timing}",
        price_line,
        f"OI / IV：{data['date']} 收盤",
        *([] if demo else [f"夜盤實際時段：{data['date']} 15:00～{session_end} 05:00"]),
        "以下位階均換算為台指期點位（沿用前一收盤日基差）。",
        "",
        "📍 Gamma Concentration：不假設造市商持倉方向",
        "⚖️ Signed GEX：依賴造市商方向假設",
        "",
        "📊 價格階梯（高 → 低）",
        *ladder,
        *(["", "無有效點位：", *missing] if missing else []),
        "",
        f"{'收盤' if demo else '盤前'}狀態：{regime}",
        f"淨／總 Gamma：{ratio_text}",
        "",
        footer,
        "此模型不是方向預測或交易指令。",
    ])


def historical_night_demo():
    """Rebuild the last completed night snapshot without touching alert state."""
    import pandas as pd
    import txo_gex as gex

    history = pd.read_csv(ROOT / "data" / "gex_history.csv").sort_values("date")
    nights = history.index[history["txf_night"].notna()]
    if len(nights) == 0:
        raise ValueError("歷史資料沒有可用夜盤收盤價")
    night_row = history.loc[nights[-1]]
    oi_day = gex.dt.date.fromisoformat(str(night_row["date"]))
    previous = history[history["date"] < night_row["date"]]
    if previous.empty:
        raise ValueError("夜盤前一交易日的 OI 資料不存在")
    levels = previous.iloc[-1]
    levels_day = gex.dt.date.fromisoformat(str(levels["date"]))
    basis = number(levels["txf_settle"]) - number(levels["F"])
    txf = pd.DataFrame([{"date": night_row["date"],
                         "txf_night": night_row["txf_night"]}])
    raw = gex.fetch_range(levels_day, levels_day)
    snapshot = gex.preopen_snapshot(gex.prepare(raw), txf, str(levels_day), basis)
    if snapshot.get("preopen_date") != oi_day.isoformat():
        raise ValueError("無法重建對應夜盤的 GEX 快照")
    data = levels.to_dict()
    data.update(snapshot)
    message = format_message(data)
    message = message.replace("🌅 台指期 GEX 盤前快照", "🧪 GEX 夜盤排版測試（歷史資料）", 1)
    message = message.replace("（延遲送達）", "")
    message += "\n這是歷史排版測試，非今日即時訊號；不占用正式早報紀錄。"
    return message, {"status": "night_demo_sent", "oi_date": str(levels_day),
                     "night_session_end": snapshot["preopen_session_end"],
                     "assigned_date": snapshot["preopen_date"]}


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
    mode.add_argument("--night-demo", action="store_true", help="用最近已完成夜盤資料測試排版，不記錄為盤前通知")
    args = parser.parse_args()
    try:
        data = json.loads(LATEST.read_text(encoding="utf-8"))
        if args.night_demo:
            token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
            if not token or not chat_id:
                result = {"status": "error", "reason": "Telegram Secrets 尚未設定"}
            else:
                message, result = historical_night_demo()
                send_message(token, chat_id, message)
        elif args.demo:
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
