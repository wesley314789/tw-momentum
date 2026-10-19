"""Fresh-date gate, futures basis conversion, and deduplication."""

import importlib.util
import json
import sys
import tempfile
import types
import datetime as dt
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


path = Path(__file__).resolve().parents[1] / "scripts" / "gex_morning_telegram.py"
spec = importlib.util.spec_from_file_location("gex_morning_telegram", path)
alert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alert)

today = date(2026, 10, 2)
sample = {"date": "2026-10-01", "F": 100., "txf_settle": 102.,
          "txf_close": 103.,
          "preopen_date": today.isoformat(), "preopen_price": 105.,
          "preopen_gex": 1000., "preopen_net_ratio": .2,
          "preopen_regime": "positive", "call_wall": 110., "put_wall": 90.,
          "peak": 101., "valley": None, "micro_flip": 99.5,
          "macro_zero": None, "call_concentration_peak": 108.,
          "put_concentration_peak": 91., "gamma_cluster_strike": 99.,
          "concentration_peak": 100., "concentration_valley": None}

assert alert.check_snapshot(sample, {}, today)["status"] == "ready"
friday_night = dict(sample, date="2026-10-16", preopen_date="2026-10-19",
                    preopen_session_start="2026-10-16",
                    preopen_session_end="2026-10-17")
assert alert.check_snapshot(friday_night, {}, date(2026, 10, 19))["status"] == "ready"
assert alert.check_snapshot(friday_night, {}, date(2026, 10, 18))["status"] == "skip"
assert alert.check_snapshot(friday_night,
                            {"last_sent_preopen_date": "2026-10-19"},
                            date(2026, 10, 19))["status"] == "quiet"
assert "夜盤實際時段：2026-10-16 15:00～2026-10-17 05:00" in alert.format_message(friday_night)
assert "盤前交易日：2026-10-19" in alert.format_message(friday_night)
bad_session = dict(friday_night, preopen_session_end="2026-10-19")
assert alert.check_snapshot(bad_session, {}, date(2026, 10, 19))["status"] == "error"
holiday_snapshot = dict(sample, date="2026-10-08", preopen_date="2026-10-12")
assert alert.check_snapshot(holiday_snapshot, {}, date(2026, 10, 12))["status"] == "skip"
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
assert "112  ⚖️ Call Wall" in message  # 110 option strike + 2 futures basis
assert "夜盤實際時段：2026-10-01 15:00～2026-10-02 05:00" in message
assert "92  ⚖️ Put Wall" in message
assert "Gamma Concentration" in message and "Signed GEX" in message
assert "⚖️ 山谷 —" in message
assert "｜" not in message
assert "110  📍 Call 集中" in message
assert "105  🌙 夜盤收盤" in message
assert "103  ☀️ 前次日盤收盤" in message
assert "102  ⚖️ Micro Flip" in message
assert "⚖️ Macro Zero —" in message
ladder, missing = alert.ranked_levels(sample, demo=False, basis=2.)
assert len([line for line in ladder if "夜盤收盤" in line]) == 1
assert [int(line.split("  ")[0].replace(",", "")) for line in ladder] == sorted(
    [int(line.split("  ")[0].replace(",", "")) for line in ladder], reverse=True)
assert len(missing) == 3
assert "\n".join(ladder) in message
assert "token" not in message.lower()

demo_sample = dict(sample, txf_close=103., regime="negative", net_ratio=-.3)
demo_message = alert.format_message(demo_sample, demo=True)
assert "測試推播" in demo_message and "2026-10-01 收盤資料" in demo_message
assert "台指期收盤 103 點" in demo_message
assert "103  ☀️ 日盤收盤" in demo_message
assert "收盤狀態：負 Gamma" in demo_message
assert "盤前快照" not in demo_message and "夜盤收盤" not in demo_message
assert "延遲送達" not in demo_message
assert "不會占用明早" in demo_message

with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "data").mkdir()
    pd.DataFrame([
        dict(sample, date="2026-10-16", txf_settle=102., F=100.,
             txf_close=103., txf_night=104.),
        dict(sample, date="2026-10-19", txf_night=105.),
    ]).to_csv(root / "data" / "gex_history.csv", index=False)
    called = []
    fake_gex = types.SimpleNamespace(
        dt=dt,
        fetch_range=lambda start, end: called.append((start, end)) or pd.DataFrame(),
        prepare=lambda raw: raw,
        preopen_snapshot=lambda df, txf, levels_date, basis: {
            "preopen_date": "2026-10-19", "preopen_session_start": levels_date,
            "preopen_session_end": "2026-10-17", "preopen_price": float(txf.iloc[0]["txf_night"]),
            "preopen_gex": 1000., "preopen_net_ratio": .2,
            "preopen_regime": "positive"} if basis == 2. else {},
    )
    old_root = alert.ROOT
    alert.ROOT = root
    sys.modules["txo_gex"] = fake_gex
    try:
        night_message, report = alert.historical_night_demo()
    finally:
        alert.ROOT = old_root
        del sys.modules["txo_gex"]
    assert report["status"] == "night_demo_sent"
    assert called == [(date(2026, 10, 16), date(2026, 10, 16))]
    assert "歷史資料" in night_message and "非今日即時訊號" in night_message
    assert "105  🌙 夜盤收盤" in night_message
    assert "103  ☀️ 前次日盤收盤" in night_message
    assert "夜盤實際時段：2026-10-16 15:00～2026-10-17 05:00" in night_message
    assert "延遲送達" not in night_message

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
