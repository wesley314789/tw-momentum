#!/usr/bin/env python3
"""Set Telegram Actions secrets without echoing the bot token to the shell."""

import getpass
import subprocess

import requests


REPO = "wesley314789/tw-momentum"


def api(token, method, *, data=None):
    try:
        response = requests.request(
            "POST" if data else "GET",
            f"https://api.telegram.org/bot{token}/{method}", data=data, timeout=20)
        result = response.json()
    except (requests.RequestException, ValueError):
        raise RuntimeError("Telegram 連線失敗；請檢查網路與機器人 token。") from None
    if response.status_code != 200 or not result.get("ok"):
        raise RuntimeError(f"Telegram 拒絕請求（HTTP {response.status_code}）。")
    return result["result"]


def save_secret(name, value):
    try:
        result = subprocess.run(
            ["gh", "secret", "set", name, "--app", "actions", "--repo", REPO],
            input=value, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        raise RuntimeError("找不到 gh CLI；請先安裝並登入 GitHub。") from None
    if result.returncode:
        raise RuntimeError(f"無法寫入 GitHub Secret {name}；請確認 gh 已登入且有 repo 權限。")


def main():
    print("先用 @BotFather 建立 bot，並在 Telegram 對新 bot 按 Start。")
    token = getpass.getpass("貼上 BotFather 提供的 token（輸入不顯示）：").strip()
    if not token:
        raise RuntimeError("沒有輸入 token。")
    updates = api(token, "getUpdates")
    chats = {}
    for update in updates:
        msg = update.get("message") or {}
        chat = msg.get("chat") or {}
        if chat.get("type") == "private" and chat.get("id") is not None:
            chats[str(chat["id"])] = chat.get("first_name") or "私訊"
    if not chats:
        raise RuntimeError("還沒有收到私人訊息；請先對 bot 按 Start，再重跑此程式。")
    for chat_id, name in chats.items():
        print(f"可用聊天室：{chat_id} ({name})")
    if len(chats) == 1:
        chat_id = next(iter(chats))
    else:
        chat_id = input("請輸入你自己的聊天室 ID：").strip()
        if chat_id not in chats:
            raise RuntimeError("輸入的聊天室 ID 不在 bot 最近的私人訊息中。")
    save_secret("TELEGRAM_BOT_TOKEN", token)
    save_secret("TELEGRAM_CHAT_ID", chat_id)
    api(token, "sendMessage", data={"chat_id": chat_id,
                                    "text": "✅ 台股風險警戒 Telegram 通道已設定完成。"})
    print("完成：兩個私密設定已存到 GitHub，測試訊息已送到 Telegram。")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"設定未完成：{exc}")
        raise SystemExit(1) from None
