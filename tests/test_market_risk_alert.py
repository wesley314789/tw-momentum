"""Alert transitions and Telegram delivery never silently count a failed send."""

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path


script = Path(__file__).resolve().parents[1] / "scripts" / "market_risk_alert.py"
spec = importlib.util.spec_from_file_location("market_risk_alert", script)
alert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alert)


def sample(day, active, series):
    return {"status": "ok", "values": {"date": day, "active_confirmed": active},
            "confirmed_series": [{"date": d, "active": a} for d, a in series]}


with tempfile.TemporaryDirectory() as folder:
    state = Path(folder) / "state.json"
    assert alert.process(sample("2026-10-01", False,
                                [("2026-09-30", True), ("2026-10-01", False)]),
                         True, state)["status"] == "quiet"
    assert alert.process(sample("2026-10-02", True,
                                [("2026-10-01", False), ("2026-10-02", True)]),
                         False, state)["status"] == "triggered"
    assert json.loads(state.read_text())["last_alert_date"] is None
    assert alert.process(sample("2026-10-02", True,
                                [("2026-10-02", True)]), True, state)["status"] == "triggered"
    assert alert.process(sample("2026-10-02", True,
                                [("2026-10-02", True)]), True, state)["status"] == "quiet"
    assert alert.process(sample("2026-10-05", False,
                                [("2026-10-02", True), ("2026-10-05", False)]),
                         True, state)["status"] == "quiet"
    assert alert.process(sample("2026-10-06", True,
                                [("2026-10-05", False), ("2026-10-06", True)]),
                         True, state)["status"] == "triggered"


values = {"date": "2026-10-06", "taiex_close": 100., "taiex_ma20": 110.,
          "broad_pct60": 30., "broad_ma20": 40., "broad_10ago": 45.,
          "leader_pct": 3., "leader_ma20": 5., "leader_10ago": 6.,
          "official_actions_through": "2026-10-06"}
sent = []


def post(url, data, timeout):
    sent.append((url, data, timeout))
    return types.SimpleNamespace(status_code=200, json=lambda: {"ok": True})


sys.modules["requests"] = types.SimpleNamespace(post=post, RequestException=Exception)
alert.send_telegram(values, "private-test-token", "12345")
assert sent[0][1]["chat_id"] == "12345"
assert "2026-10-06" in sent[0][1]["text"]
assert "private-test-token" not in sent[0][1]["text"]
sys.modules["requests"].post = lambda *args, **kwargs: types.SimpleNamespace(
    status_code=401, json=lambda: {"ok": False})
try:
    alert.send_telegram(values, "private-test-token", "12345")
    raise AssertionError("A rejected send must fail")
except RuntimeError as exc:
    assert "private-test-token" not in str(exc)
print("market risk alert checks passed")
