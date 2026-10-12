"""A holiday-assigned futures night must not become a fresh preopen snapshot."""

import datetime as dt
import sys
from unittest.mock import patch
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import txo_gex as g  # noqa: E402


day_df = pd.DataFrame({"_trade_date": [dt.date(2026, 10, 8)]})
holiday_night = pd.DataFrame([{"date": "2026-10-12", "txf_night": 48705.0}])
with patch.object(g, "build_chain") as build:
    stale = g.preopen_snapshot(day_df, holiday_night, "2026-10-08", 202.0)
    build.assert_not_called()
assert stale == {
    "stale_night_date": "2026-10-12",
    "stale_night_session_start": "2026-10-08",
    "stale_night_price": 48705.0,
}
assert "preopen_date" not in stale

normal_night = pd.DataFrame([{"date": "2026-10-13", "txf_night": 49100.0}])
chain = [{"K": 49000.0, "T": 2 / 365, "iv": 0.25, "oi_c": 100.0,
          "oi_p": 50.0}]
with patch.object(g, "build_chain", return_value=(chain, 49000.0)):
    fresh = g.preopen_snapshot(
        pd.DataFrame({"_trade_date": [dt.date(2026, 10, 12)]}),
        normal_night, "2026-10-12", 200.0)
assert fresh["preopen_date"] == "2026-10-13"
assert fresh["preopen_price"] == 49100.0
assert fresh["preopen_regime"] in {"positive", "negative", "neutral"}
assert "stale_night_date" not in fresh

print("GEX preopen freshness checks passed")
