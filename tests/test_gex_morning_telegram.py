"""Fresh-date gate, futures basis conversion, and deduplication."""

import importlib.util
import json
import sys
import tempfile
import types
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


path = Path(__file__).resolve().parents[1] / "scripts" / "gex_morning_telegram.py"
spec = importlib.util.spec_from_file_location("gex_morning_telegram", path)
alert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alert)

today = date(2026, 10, 2)
sample = {"date": "2026-10-01", "F": 100., "txf_settle": 102.,
          "preopen_date": today.isoformat(), "preopen_price": 105.,
          "preopen_gex": 1000., "preopen_net_ratio": .2,
          "preopen_regime": "positive", "call_wall": 110., "put_wall": 90.,
          "peak": 101., "valley": None, "micro_flip": 99.5,
          "macro_zero": None, "call_concentration_peak": 108.,
          "put_concentration_peak": 91., "gamma_cluster_strike": 99.,
          "concentration_peak": 100., "concentration_valley": None}

assert alert.check_snapshot(sample, {}, today)["status"] == "ready"
assert alert.check_snapshot(sample, {}, date(2026, 10, 3))["status"] == "skip"
assert alert.check_snapshot(sample, {"last_sent_preopen_date": today.isoformat()},
                            today)["status"] == "quiet"
bad = dict(sample, txf_settle=None)
assert alert.check_snapshot(bad, {}, today)["status"] == "error"
assert alert.seconds_until_eight(datetime(2026, 10, 2, 7, 55,
                                          tzinfo=ZoneInfo("Asia/Taipei"))) == 300
assert alert.seconds_until_eight(datetime(2026, 10, 2, 8, 1,
                                          tzinfo=ZoneInfo("Asia/Taipei"))) == 0
bad = dict(sample, date=today.isoformat())
assert alert.check_snapshot(bad, {}, today)["status"] == "error"

message = alert.format_message(sample)
assert "Call Wall 112" in message  # 110 option strike + 2 futures basis
assert "Put Wall 92" in message
assert "Gamma Concentration" in message and "Signed GEX" in message
assert "山谷 —" in message
assert "｜" not in message
assert "Call 集中 110\nPut 集中 93" in message
assert "Call Wall 112\nPut Wall 92" in message
assert "Micro Flip 102\nMacro Zero —" in message
assert "token" not in message.lower()

demo_sample = dict(sample, txf_close=103., regime="negative", net_ratio=-.3)
demo_message = alert.format_message(demo_sample, demo=True)
assert "測試推播" in demo_message and "2026-10-01 收盤資料" in demo_message
assert "台指期收盤 103 點" in demo_message
assert "收盤狀態：負 Gamma" in demo_message
assert "盤前快照" not in demo_message and "夜盤收盤" not in demo_message
assert "延遲送達" not in demo_message
assert "不會占用明早" in demo_message

with tempfile.TemporaryDirectory() as folder:
    state_path = Path(folder) / "state.json"
    delivered = []

    def post(url, data, timeout):
        delivered.append(data["text"])
        return types.SimpleNamespace(status_code=200, json=lambda: {"ok": True})

    sys.modules["requests"] = types.SimpleNamespace(post=post, RequestException=Exception)
    alert.send_message("private-test-token", "12345", message)
    alert.save_state(state_path, sample["preopen_date"], sample["date"])
    state = json.loads(state_path.read_text())
    assert state["last_sent_preopen_date"] == today.isoformat()
    assert alert.check_snapshot(sample, state, today)["status"] == "quiet"
    assert delivered == [message]

    sys.modules["requests"].post = lambda *a, **k: types.SimpleNamespace(
        status_code=401, json=lambda: {"ok": False})
    try:
        alert.send_message("private-test-token", "12345", message)
        raise AssertionError("A rejected send must fail")
    except RuntimeError as exc:
        assert "private-test-token" not in str(exc)
    assert json.loads(state_path.read_text()) == state  # failed send cannot advance state

print("GEX Telegram checks passed")
