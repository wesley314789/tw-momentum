#!/usr/bin/env python3
"""Send the previous session's newly listed Taiwan momentum stocks each morning.

Uses the same ``days == 1`` definition as the website's 新 badge. No stock
prices are fetched here: the after-close pipeline must have published latest.json.
"""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from gex_morning_telegram import send_message, wait_until_eight


ROOT = Path(__file__).resolve().parent.parent
LATEST = ROOT / "docs" / "data" / "latest.json"
STATE = ROOT / "data" / "momentum_telegram_state.json"
TAIPEI = ZoneInfo("Asia/Taipei")
MAX_AGE_DAYS = 4  # Friday close can still be the latest session on Tuesday after a holiday.
MAX_MESSAGE_CHARS = 3500  # Telegram's text limit is 4096 characters.


def check_snapshot(data: dict, state: dict, today: dt.date) -> dict:
    """Reject partial/stale data and never resend a completed trading date."""
    # The external 08:00 dispatch runs every day. Keep Friday's list for Monday.
    if today.weekday() >= 5:
        return {"status": "skip", "reason": "週末不發送股票早報"}
    try:
        trade_date = dt.date.fromisoformat(data["trade_date"])
    except (KeyError, TypeError, ValueError):
        return {"status": "error", "reason": "動能資料日期無效"}
    age = (today - trade_date).days
    if age < 1 or age > MAX_AGE_DAYS:
        return {"status": "skip", "reason": "不是近期已收盤的交易日資料",
                "trade_date": trade_date.isoformat()}
    picks = data.get("momentum")
    breadth = data.get("breadth")
    if (not isinstance(picks, list) or not isinstance(breadth, list) or not breadth
            or data.get("member_days", 0) < 2
            or breadth[-1].get("date") != trade_date.isoformat()
            or breadth[-1].get("count") != len(picks)
            or any(not isinstance(p, dict) or not isinstance(p.get("days"), int)
                   or p["days"] < 1 for p in picks)):
        return {"status": "error", "reason": "動能名單與市場廣度不一致或缺少上榜天數"}
    if state.get("last_sent_trade_date") == trade_date.isoformat():
        return {"status": "quiet", "reason": "這個交易日已發送過",
                "trade_date": trade_date.isoformat()}
    return {"status": "ready", "trade_date": trade_date.isoformat(),
            "new_count": sum(p["days"] == 1 for p in picks)}


def format_messages(data: dict, *, demo: bool = False) -> list[str]:
    """Include every new name, splitting only when Telegram's size limit requires it."""
    picks = sorted((p for p in data["momentum"] if p["days"] == 1),
                   key=lambda p: (-float(p.get("value") or 0),
                                  -float(p.get("excess_1m") or 0), str(p.get("code", ""))))
    breadth = data["breadth"][-1]
    title = "🧪 動能新上榜測試" if demo else "📈 動能新上榜"
    base = (f"{title}｜{data['trade_date']} 收盤\n"
            f"新增 {len(picks)} 檔｜動能榜 {len(data['momentum'])} 檔"
            f"｜市場廣度 {float(breadth['pct']):.2f}%\n"
            "依成交值由高到低")
    if demo:
        base += "\n這是測試訊息，不影響正式早報。"
    if not picks:
        return [base + "\n今天沒有新進個股。"]

    rows = []
    for i, p in enumerate(picks, 1):
        theme = p.get("theme") or ("個股因素" if p.get("theme_src") == "override" else "未歸類")
        value = float(p.get("value") or 0)
        excess = p.get("excess_1m")
        excess_text = f"{float(excess):+.1f}%" if excess is not None else "—"
        rows.append(f"{i}. {p['code']} {p['name']}｜{theme}｜{value:.2f}億｜超額 {excess_text}")

    # Reserve space for the part number, so every final message remains below the limit.
    limit = MAX_MESSAGE_CHARS - len(base) - 24
    chunks, current = [], []
    for row in rows:
        if current and len("\n".join(current)) + len(row) + 1 > limit:
            chunks.append(current)
            current = []
        current.append(row)
    if current:
        chunks.append(current)
    return [base + (f"（{i}/{len(chunks)}）" if len(chunks) > 1 else "")
            + "\n" + "\n".join(chunk)
            for i, chunk in enumerate(chunks, 1)]


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def deliver(data: dict, state: dict, path: Path, token: str, chat_id: str) -> dict:
    """Checkpoint each successful part so a failed later part can resume."""
    trade_date = data["trade_date"]
    messages = format_messages(data)
    start = state.get("sent_parts", 0) if state.get("pending_trade_date") == trade_date else 0
    if not isinstance(start, int) or start < 0 or start > len(messages):
        start = 0
    for i in range(start, len(messages)):
        send_message(token, chat_id, messages[i])
        save_state(path, {"pending_trade_date": trade_date, "sent_parts": i + 1})
    save_state(path, {"last_sent_trade_date": trade_date})
    return {"status": "sent", "trade_date": trade_date,
            "new_count": sum(p["days"] == 1 for p in data["momentum"]),
            "messages": len(messages)}


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--send", action="store_true", help="正式發送；預設只檢查")
    mode.add_argument("--demo", action="store_true", help="立即送一則測試名單，不更新正式通知紀錄")
    args = parser.parse_args()
    try:
        data = json.loads(LATEST.read_text(encoding="utf-8"))
        state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
        today = dt.datetime.now(TAIPEI).date()
        if args.demo:
            token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
            if not token or not chat_id:
                result = {"status": "error", "reason": "Telegram Secrets 尚未設定"}
            else:
                messages = format_messages(data, demo=True)
                for message in messages:
                    send_message(token, chat_id, message)
                result = {"status": "demo_sent", "trade_date": data["trade_date"],
                          "new_count": sum(p["days"] == 1 for p in data["momentum"]),
                          "messages": len(messages)}
        else:
            result = check_snapshot(data, state, today)
            if args.send and result["status"] == "ready":
                token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
                chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
                if not token or not chat_id:
                    result = {"status": "error", "reason": "Telegram Secrets 尚未設定"}
                else:
                    wait_until_eight()
                    result = deliver(data, state, STATE, token, chat_id)
    except Exception as exc:
        result = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 1 if result["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
