import unittest
import tempfile
from pathlib import Path

import pandas as pd

from scripts import study_us_index_drawdown as study
from scripts.study_us_index_drawdown import first_crossings, stock_breadth


class USIndexDrawdownStudyTests(unittest.TestCase):
    def test_rolling_high_uses_history_available_that_day(self):
        dates = pd.date_range("2025-01-01", periods=253, freq="B").strftime("%Y-%m-%d").tolist()
        close = [100.0] * 252 + [80.0]
        bars = {"TEST": pd.DataFrame({"date": dates, "adjclose": close})}
        breadth = stock_breadth(bars, ["TEST"], dates)
        at_20 = breadth[breadth.depth.eq(20)].set_index("date")
        self.assertEqual(int(at_20.loc[dates[250], "eligible"]), 0)
        self.assertEqual(int(at_20.loc[dates[251], "count"]), 0)
        self.assertEqual(int(at_20.loc[dates[252], "count"]), 1)

    def test_missing_stock_bar_is_excluded_from_numerator_and_denominator(self):
        dates = pd.date_range("2025-01-01", periods=253, freq="B").strftime("%Y-%m-%d").tolist()
        bars = {"TEST": pd.DataFrame({"date": dates[:-1], "adjclose": [100.0]*252})}
        breadth = stock_breadth(bars, ["TEST"], dates)
        last = breadth[(breadth.date.eq(dates[-1])) & breadth.depth.eq(20)].iloc[0]
        self.assertEqual(int(last.eligible), 0)
        self.assertEqual(int(last["count"]), 0)

    def test_first_crossings_skips_left_censored_and_applies_cooldown(self):
        a = pd.Series([True, False, True, True, False, True, False, True])
        self.assertEqual(first_crossings(a, cooldown=4), [2, 7])

    def test_daily_snapshot_is_observed_and_idempotent(self):
        dates = pd.date_range("2025-01-01", periods=252, freq="B").strftime("%Y-%m-%d").tolist()
        index_frame = pd.DataFrame({"date": dates, "close": [100.0]*252})
        stock_frame = pd.DataFrame({"date": dates, "adjclose": [100.0]*251+[80.0]})
        bars = {"^GSPC": index_frame, "^NDX": index_frame, "^SOX": index_frame,
                "TEST": stock_frame}
        members = {name: ["TEST"] for name in study.INDICES}
        holdings_dates = {name: dates[-1] for name in study.INDICES}
        old_csv, old_json = study.LIVE_CSV, study.LIVE_JSON
        with tempfile.TemporaryDirectory() as directory:
            try:
                study.LIVE_CSV = Path(directory) / "live.csv"
                study.LIVE_JSON = Path(directory) / "live.json"
                study.write_daily(bars, members, holdings_dates)
                study.write_daily(bars, members, holdings_dates)
                saved = pd.read_csv(study.LIVE_CSV)
                self.assertEqual(len(saved), 3)
                self.assertTrue(saved["count"].eq(1).all())
                self.assertTrue(saved["basis"].eq("observed_etf_holdings").all())
            finally:
                study.LIVE_CSV, study.LIVE_JSON = old_csv, old_json


if __name__ == "__main__":
    unittest.main()
