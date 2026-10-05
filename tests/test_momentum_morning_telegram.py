"""Freshness, full-list formatting, and resume after a partial Telegram send."""

import json
import io
import os
import sys
import tempfile
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import momentum_morning_telegram as alert


sample = {
    "trade_date": "2026-10-02", "member_days": 250,
    "breadth": [{"date": "2026-10-02", "count": 2, "pct": 5.0}],
    "momentum": [
        {"code": "1111", "name": "Low", "days": 1, "value": 1.2,
         "theme": "Theme A", "excess_1m": 12.3},
        {"code": "2222", "name": "High", "days": 1, "value": 8.4,
         "theme": None, "excess_1m": 15.1},
    ],
}

assert alert.check_snapshot(sample, {}, date(2026, 10, 3))["status"] == "skip"
assert alert.check_snapshot(sample, {}, date(2026, 10, 4))["status"] == "skip"
assert alert.check_snapshot(sample, {}, date(2026, 10, 5)) == {
    "status": "ready", "trade_date": "2026-10-02", "new_count": 2}
assert alert.check_snapshot(sample, {}, date(2026, 10, 2))["status"] == "skip"
assert alert.check_snapshot(sample, {}, date(2026, 10, 7))["status"] == "skip"
assert alert.check_snapshot(sample, {"last_sent_trade_date": "2026-10-02"},
                            date(2026, 10, 5))["status"] == "quiet"
assert alert.check_snapshot(dict(sample, breadth=[{"date": "2026-10-02", "count": 1}]),
                            {}, date(2026, 10, 5))["status"] == "error"

message = alert.format_messages(sample)[0]
assert message.index("2222 High") < message.index("1111 Low")
assert "新增 2 檔" in message and "未歸類" in message
assert "2026-10-02 收盤" in message
assert len(message) < alert.MAX_MESSAGE_CHARS
demo_message = alert.format_messages(sample, demo=True)[0]
assert "動能新上榜測試" in demo_message and "不影響正式早報" in demo_message

no_new = dict(sample, momentum=[dict(p, days=2) for p in sample["momentum"]])
assert "今天沒有新進個股" in alert.format_messages(no_new)[0]

many = dict(sample,
            momentum=[dict(sample["momentum"][0], code=str(i).zfill(4),
                           name="Long Company Name " * 3)
                      for i in range(220)])
many["breadth"] = [{"date": "2026-10-02", "count": 220, "pct": 10.0}]
parts = alert.format_messages(many)
assert len(parts) > 1 and all(len(part) <= alert.MAX_MESSAGE_CHARS for part in parts)
assert sum(part.count("Long Company Name") for part in parts) == 220 * 3

with tempfile.TemporaryDirectory() as folder:
    state_path = Path(folder) / "state.json"
    sent = []

    def fail_on_second(token, chat_id, text):
        if len(sent) == 1:
            raise RuntimeError("simulated Telegram failure")
        sent.append(text)

    alert.send_message = fail_on_second
    try:
        alert.deliver(many, {}, state_path, "token", "chat")
        raise AssertionError("A partial send must fail")
    except RuntimeError:
        pass
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state == {"pending_trade_date": "2026-10-02", "sent_parts": 1}

    alert.send_message = lambda token, chat_id, text: sent.append(text)
    result = alert.deliver(many, state, state_path, "token", "chat")
    assert result["status"] == "sent" and sent == parts
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state == {"last_sent_trade_date": "2026-10-02"}
    assert alert.check_snapshot(many, state, date(2026, 10, 5))["status"] == "quiet"

    latest_path = Path(folder) / "latest.json"
    latest_path.write_text(json.dumps(sample), encoding="utf-8")
    alert.LATEST, alert.STATE = latest_path, state_path
    alert.send_message = lambda token, chat_id, text: sent.append(text)
    os.environ["TELEGRAM_BOT_TOKEN"] = "test-token"
    os.environ["TELEGRAM_CHAT_ID"] = "test-chat"
    sys.argv = ["momentum_morning_telegram.py", "--demo"]
    before = state_path.read_text(encoding="utf-8")
    with redirect_stdout(io.StringIO()) as output:
        assert alert.main() == 0
    assert json.loads(output.getvalue())["status"] == "demo_sent"
    assert sent[-1] == demo_message
    assert state_path.read_text(encoding="utf-8") == before

print("Momentum Telegram checks passed")
