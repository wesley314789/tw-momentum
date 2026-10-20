"""52-week records use completed closes and a fully eligible universe."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import high_low_breadth as breadth


class HighLowBreadthTest(unittest.TestCase):
    def test_records_exclude_today_from_reference_and_ties(self):
        dates = pd.bdate_range("2025-01-01", periods=254).strftime("%Y-%m-%d")
        rows = []
        for i, day in enumerate(dates):
            rise = 100 + min(i, 252) / 10
            fall = 200 - min(i, 252) / 10
            rows.extend([
                {"date": day, "code": "1001", "market": "上市", "close": rise},
                {"date": day, "code": "1002", "market": "上市", "close": fall},
                {"date": day, "code": "1003", "market": "上市", "close": 150.0},
            ])
            if i >= 150:
                rows.append({"date": day, "code": "1004", "market": "上市", "close": 300 + i})
        empty_actions = pd.DataFrame(columns=["date", "code", "market", "prev_close", "ratio"])
        result = breadth.calculate(pd.DataFrame(rows), empty_actions)
        self.assertEqual(result.date.iloc[0], dates[252])
        self.assertEqual(result.iloc[0][["new_high", "new_low", "eligible"]].tolist(), [1, 1, 3])
        self.assertEqual(result.iloc[-1][["new_high", "new_low", "eligible"]].tolist(), [0, 0, 3])
        self.assertAlmostEqual(result.high_pct.iloc[0], 33.33)
        json.dumps(result.to_dict("records"), allow_nan=False)

    def test_adjustment_avoids_false_low(self):
        dates = pd.bdate_range("2025-01-01", periods=254).strftime("%Y-%m-%d")
        hist = pd.DataFrame({"date": dates, "code": "1001", "market": "上市",
                             "close": [100.] * 130 + [50.] * 124})
        actions = pd.DataFrame([{"date": dates[130], "code": "1001",
                                 "market": "上市", "prev_close": 100., "ratio": .5}])
        result = breadth.calculate(hist, actions)
        self.assertEqual(result.iloc[-1][["new_high", "new_low", "eligible"]].tolist(),
                         [0, 0, 1])
        # An unexplained split is excluded instead of counted as a new low.
        raw = breadth.calculate(hist, actions.iloc[0:0])
        self.assertTrue(raw.empty)

    def test_incremental_update_keeps_backfilled_dates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "high_low.csv"
            old = pd.DataFrame([{"date": "2026-10-19", "new_high": 2, "new_low": 3,
                                 "eligible": 10, "high_pct": 20., "low_pct": 30.}])
            old.to_csv(path, index=False)
            fresh = pd.DataFrame([
                {"date": "2026-10-19", "new_high": 1, "new_low": 1,
                 "eligible": 10, "high_pct": 10., "low_pct": 10.},
                {"date": "2026-10-20", "new_high": 4, "new_low": 0,
                 "eligible": 10, "high_pct": 40., "low_pct": 0.},
            ])
            with patch.object(breadth, "PATH", path), \
                 patch.object(breadth, "calculate", return_value=fresh), \
                 patch.object(breadth.market_risk_alert, "load_actions",
                              return_value=(pd.DataFrame(), {"end": "2026-10-20"})):
                got, _ = breadth.update(pd.DataFrame({"date": ["2026-10-20"]}))
            self.assertEqual(got.new_high.tolist(), [2, 4])


if __name__ == "__main__":
    unittest.main()
