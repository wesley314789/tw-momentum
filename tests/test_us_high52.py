import datetime as dt
import unittest

import pandas as pd

from scripts.us_breadth import add_high52_distance


class USHigh52Tests(unittest.TestCase):
    def test_adjusted_history_ignores_a_split(self):
        dates = pd.bdate_range("2025-01-01", periods=252).date
        bars = {"TEST": pd.DataFrame({
            "date": dates,
            "close": [100.0] * 251 + [50.0],
            "adjclose": [50.0] * 252,
        })}
        recs = [{"symbol": "TEST", "close": 50.0}]
        add_high52_distance(recs, bars, dates[-1])
        self.assertEqual(recs[0]["high52"], 50.0)
        self.assertEqual(recs[0]["high52_distance_pct"], 0.0)

    def test_distance_and_full_window_requirement(self):
        dates = pd.bdate_range("2025-01-01", periods=252).date
        bars = {"FULL": pd.DataFrame({
            "date": dates,
            "close": [100.0] * 251 + [80.0],
            "adjclose": [100.0] * 251 + [80.0],
        }), "SHORT": pd.DataFrame({
            "date": dates[1:],
            "close": [100.0] * 250 + [80.0],
            "adjclose": [100.0] * 250 + [80.0],
        })}
        recs = [{"symbol": "FULL", "close": 80.0},
                {"symbol": "SHORT", "close": 80.0}]
        add_high52_distance(recs, bars, dates[-1])
        self.assertEqual(recs[0]["high52"], 100.0)
        self.assertEqual(recs[0]["high52_distance_pct"], 20.0)
        self.assertIsNone(recs[1]["high52"])
        self.assertIsNone(recs[1]["high52_distance_pct"])

    def test_no_stale_quote_or_missing_adjusted_price(self):
        dates = pd.bdate_range("2025-01-01", periods=252).date
        bars = {"STALE": pd.DataFrame({
            "date": dates,
            "close": [100.0] * 252,
            "adjclose": [100.0] * 252,
        }), "BAD": pd.DataFrame({
            "date": dates,
            "close": [100.0] * 252,
            "adjclose": [100.0] * 251 + [None],
        })}
        recs = [{"symbol": "STALE", "close": 100.0},
                {"symbol": "BAD", "close": 100.0}]
        add_high52_distance(recs, bars, dates[-1] + dt.timedelta(days=1))
        self.assertIsNone(recs[0]["high52_distance_pct"])
        add_high52_distance(recs, bars, dates[-1])
        self.assertIsNone(recs[1]["high52_distance_pct"])


if __name__ == "__main__":
    unittest.main()
