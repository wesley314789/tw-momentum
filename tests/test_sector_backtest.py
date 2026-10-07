"""歷史 Strong Score 與當日訊號的時間界線。"""

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import backtest_sector_breadth as study
from scripts import backtest_momentum as bt
from scripts import sector_breadth


class SectorBacktestTest(unittest.TestCase):
    def test_historical_score_matches_live_calculation_and_excludes_today_high(self):
        days = pd.bdate_range("2026-04-01", periods=62).strftime("%Y-%m-%d").tolist()
        close = [90.] * 60 + [96., 101.]
        high = [max(100., price) for price in close]
        hist = pd.DataFrame({"date": days, "code": "A001", "name": "A",
                             "market": "上市", "close": close,
                             "high": high, "adj_close": close, "adj_high": high,
                             "volume": [100.] * 61 + [150.], "_corp": False})
        idx = pd.DataFrame({"date": days, "close": 100.})
        live = sector_breadth.stock_at(hist, 61, lambda _: 100.)
        scores = study.historical_scores(hist, {"A001": "半導體業"}, idx, days[-1], days[-1])
        self.assertEqual(len(scores), 1)
        self.assertEqual(bool(scores.iloc[0]["strong"]), live["strong"])
        self.assertEqual(live["strong_score"], 6)
        hist.loc[61, "adj_high"] = 999.
        again = study.historical_scores(hist, {"A001": "半導體業"}, idx, days[-1], days[-1])
        self.assertEqual(bool(again.iloc[0]["strong"]), live["strong"])

    def test_vectorized_historical_scores_match_each_scalar_trading_day(self):
        rng = np.random.default_rng(7)
        days = pd.bdate_range("2026-01-02", periods=90).strftime("%Y-%m-%d").tolist()
        idx = pd.DataFrame({"date": days, "close": np.linspace(100, 105, len(days))})
        lookup = dict(zip(idx["date"], idx["close"]))
        frames = []
        for code in ("A001", "A002", "A003"):
            close = 100 + np.cumsum(rng.normal(0.1, 1.7, len(days)))
            frames.append(pd.DataFrame({
                "date": days, "code": code, "name": code, "market": "上市",
                "close": close, "adj_close": close, "high": close * 1.02,
                "adj_high": close * 1.02,
                "volume": rng.integers(80, 200, len(days)), "_corp": False,
            }))
        hist = pd.concat(frames, ignore_index=True)
        scores = study.historical_scores(hist, {c: "半導體業" for c in ("A001", "A002", "A003")},
                                         idx, days[60], days[-1])
        observed = {(r.date, r.code): bool(r.strong) for r in scores.itertuples()}
        for group in frames:
            for i in range(60, len(days)):
                scalar = sector_breadth.stock_at(group, i, lookup.get)
                self.assertEqual(observed[(days[i], group["code"].iloc[0])], scalar["strong"])

    def test_ranked_sectors_use_only_same_day_and_at_least_three_stocks(self):
        rows = []
        for sector, strong in (("A", 3), ("B", 2), ("C", 1),
                               ("D", 0), ("E", 3), ("F", 2)):
            for i in range(3):
                rows.append({"date": "2026-10-06", "sector": sector,
                             "code": sector+str(i), "strong": i < strong})
        rows.append({"date": "2026-10-06", "sector": "Z", "code": "Z1", "strong": True})
        top, leaders = study.top_five(pd.DataFrame(rows))
        self.assertEqual(top["2026-10-06"], {"A", "E", "B", "F", "C"})
        self.assertEqual(leaders["sector"].tolist(), ["A", "E", "B", "F", "C"])

    def test_sector_rotation_does_not_turn_old_member_into_new_listing(self):
        days = ["2026-10-05", "2026-10-06", "2026-10-07"]
        members = {days[0]: {"A"}, days[1]: {"A"}, days[2]: {"A", "B"}}
        top = {days[0]: set(), days[1]: {"X"}, days[2]: {"X"}}
        signals, counts = study.filtered_new_signals(
            members, days, set(), {"A": "X", "B": "X"}, top)
        self.assertEqual(signals, {days[0]: set(), days[1]: set(), days[2]: {"B"}})
        self.assertEqual((counts["new"], counts["selected"]), (2, 1))

    def test_shorter_backtest_does_not_truncate_membership_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "members.csv.gz"
            pd.DataFrame({"date": ["2024-01-02", "2026-01-02"],
                          "code": ["1101", "2330"]}).to_csv(path, index=False)
            old_path = bt.MEM_CACHE
            try:
                bt.MEM_CACHE = path
                selected = bt.membership(pd.DataFrame(), pd.DataFrame(), ["2026-01-02"])
                self.assertEqual(selected["2026-01-02"], {"2330"})
                cached = pd.read_csv(path, dtype={"date": str, "code": str})
                self.assertEqual(set(cached["date"]), {"2024-01-02", "2026-01-02"})
            finally:
                bt.MEM_CACHE = old_path


if __name__ == "__main__":
    unittest.main()
