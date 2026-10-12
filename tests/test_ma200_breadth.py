"""200MA participation uses complete histories and adjusted closes."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ma200_breadth
import market_risk_alert


class Ma200BreadthTest(unittest.TestCase):
    def test_window_and_denominator(self):
        days = pd.bdate_range("2025-01-01", periods=202).strftime("%Y-%m-%d")
        rows = []
        for i, day in enumerate(days):
            rows.extend([
                {"date": day, "code": "1001", "market": "上市", "close": 100 + i / 10},
                {"date": day, "code": "1002", "market": "上市", "close": 200 - i / 10},
            ])
            if i >= 181:
                rows.append({"date": day, "code": "1003", "market": "上市", "close": 300})
        hist = pd.DataFrame(rows)
        actions = pd.DataFrame(columns=["date", "code", "market", "prev_close", "ratio"])
        got = ma200_breadth.calculate(hist, actions)
        self.assertEqual(got.date.iloc[0], days[200])
        self.assertEqual(len(got), 2)
        self.assertEqual(got.iloc[-1][["count", "eligible", "pct"]].tolist(), [1, 2, 50.0])
        json.dumps(got.to_dict("records"), allow_nan=False)

    def test_ex_dividend_adjustment(self):
        days = pd.bdate_range("2025-01-01", periods=202).strftime("%Y-%m-%d")
        hist = pd.DataFrame({"date": days, "code": "1001", "market": "上市",
                             "close": np.where(np.arange(202) < 100, 100.0, 50.0)})
        actions = pd.DataFrame([{"date": days[100], "code": "1001", "market": "上市",
                                 "prev_close": 100.0, "ratio": 0.5}])
        got = ma200_breadth.calculate(hist, actions)
        self.assertEqual(got.iloc[-1][["count", "eligible", "pct"]].tolist(), [0, 1, 0.0])
        raw = market_risk_alert.participation(hist, actions, False, window=200)
        self.assertEqual(int(raw.eligible.iloc[-1]), 0)
        self.assertIn("pct60", market_risk_alert.participation(hist, actions, True).columns)

    def test_incremental_run_keeps_prior_dates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "breadth.csv"
            old = pd.DataFrame([{"date": "2026-10-09", "count": 5,
                                 "eligible": 10, "pct": 50.0}])
            old.to_csv(path, index=False)
            fresh = pd.DataFrame([
                {"date": "2026-10-09", "count": 4, "eligible": 10, "pct": 40.0},
                {"date": "2026-10-12", "count": 6, "eligible": 10, "pct": 60.0},
            ])
            history = pd.DataFrame({"date": ["2026-10-12"]})
            with patch.object(ma200_breadth, "PATH", path), \
                 patch.object(ma200_breadth, "calculate", return_value=fresh), \
                 patch.object(ma200_breadth.market_risk_alert, "load_actions",
                              return_value=(pd.DataFrame(), {"end": "2026-10-12"})):
                got, _ = ma200_breadth.update(history)
            self.assertEqual(got.pct.tolist(), [50.0, 60.0])


if __name__ == "__main__":
    unittest.main()
