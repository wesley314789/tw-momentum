"""Boundary and timing checks for the drawdown-depth event study."""

import unittest

import pandas as pd

from scripts import study_drawdown_depth as study


class DrawdownDepthStudyTest(unittest.TestCase):
    def test_exact_20_percent_matches_published_threshold(self):
        h = pd.DataFrame([
            ("2026-01-01", "上市", 80., 100.),
            ("2026-01-01", "上櫃", 81., 100.),
        ], columns=["date", "market", "price", "high"])
        result = study.breadth_for_depth(h, 20)
        self.assertEqual(int(result.listed_count.iloc[0]), 1)
        self.assertEqual(int(result.otc_count.iloc[0]), 0)

    def test_outcome_uses_only_future_sessions_and_onset_is_observed(self):
        index = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=41,
                                                      freq="B").strftime("%Y-%m-%d"),
                              "close": [100.] * 41, "low": [100.] * 41})
        index.loc[0, "low"] = 1.  # Today's low is not a future outcome.
        index.loc[20, "low"] = 95.
        index.loc[21, "low"] = 80.  # Outside the 20-session window.
        result = study.add_outcomes(index)
        self.assertAlmostEqual(float(result.worst_next20_pct.iloc[0]), -5.)
        self.assertAlmostEqual(float(result.worst_next40_pct.iloc[0]), -20.)
        self.assertTrue(bool(result.down5.iloc[0]))

        active = pd.Series([True, True, False, True] + [False] * 18 + [True])
        self.assertEqual(study.first_crossings(active), [3])


if __name__ == "__main__":
    unittest.main()
