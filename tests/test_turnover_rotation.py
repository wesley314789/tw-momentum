"""Turnover share uses the full covered universe, not momentum picks."""

import datetime as dt
import unittest

import numpy as np
import pandas as pd

from scripts import turnover_rotation as rotation


class TaiwanRotationTests(unittest.TestCase):
    def test_current_map_and_excluded_sectors(self):
        history = pd.DataFrame([
            {"date": "2026-10-15", "code": "2330", "market": "上市", "value": 50},
            {"date": "2026-10-15", "code": "1101", "market": "上市", "value": 50},
            {"date": "2026-10-16", "code": "2330", "market": "上市", "value": 80},
            {"date": "2026-10-16", "code": "1101", "market": "上市", "value": 20},
            {"date": "2026-10-16", "code": "2881", "market": "上市", "value": 30},
        ])
        mapping = pd.DataFrame([
            {"code": "2330", "market": "上市", "industry_code": "24"},
            {"code": "1101", "market": "上市", "industry_code": "01"},
            {"code": "2881", "market": "上市", "industry_code": "17"},
        ])
        rows = rotation.taiwan_series(history, mapping)
        self.assertEqual([r["date"] for r in rows], ["2026-10-15", "2026-10-16"])
        self.assertEqual(rows[-1]["tech_share_pct"], 80)
        self.assertEqual(rows[-1]["other_value"], 30)
        self.assertEqual((rows[-1]["tech_count"], rows[-1]["traditional_count"]), (1, 1))


class USRotationTests(unittest.TestCase):
    def test_same_day_prices_and_excluded_sectors(self):
        d1 = dt.date(2026, 10, 14).toordinal()
        d2 = dt.date(2026, 10, 15).toordinal()
        universe = pd.DataFrame([
            {"yahoo": "MSFT", "sector": "Technology"},
            {"yahoo": "A", "sector": "Industrials"},
            {"yahoo": "JPM", "sector": "Finance"},
        ])
        arrays = {
            "MSFT": (np.array([d1, d2]), np.array([10., 20.]), np.array([10., 10.])),
            "A": (np.array([d1, d2]), np.array([10., 10.]), np.array([10., 10.])),
            "JPM": (np.array([d1, d2]), np.array([10., 10.]), np.array([10., 10.])),
        }
        # The production guard requires 500 covered stocks. Use real-sized
        # independent symbols so the test covers that guard too.
        for i in range(500):
            sym = f"T{i}"
            universe.loc[len(universe)] = {"yahoo": sym, "sector": "Technology"}
            arrays[sym] = (np.array([d1, d2]), np.array([1., 1.]),
                           np.array([1., 1.]))
        rows = rotation.us_series(arrays, universe, [d1, d2])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["tech_value"], 700)
        self.assertEqual(rows[-1]["traditional_value"], 100)
        self.assertEqual(rows[-1]["other_value"], 100)
        self.assertEqual(rows[-1]["tech_share_pct"], 87.5)
        self.assertEqual(rows[-1]["coverage_pct"], 100)

    def test_sparse_date_is_not_published(self):
        d = dt.date(2026, 10, 15).toordinal()
        universe = pd.DataFrame({"yahoo": [str(i) for i in range(700)],
                                 "sector": ["Technology"] * 700})
        arrays = {str(i): (np.array([d]), np.array([1.]), np.array([1.]))
                  for i in range(500)}
        self.assertEqual(rotation.us_series(arrays, universe, [d]), [])


if __name__ == "__main__":
    unittest.main()
