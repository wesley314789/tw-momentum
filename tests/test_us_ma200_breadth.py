"""US 200MA breadth: same-day membership and no future price leakage."""

import datetime as dt
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import us_breadth


class UsMa200BreadthTest(unittest.TestCase):
    def test_window_denominator_and_no_future_bar(self):
        days = pd.bdate_range("2025-01-01", periods=202)
        dates = np.array([d.date().toordinal() for d in days])
        rising = np.linspace(100, 120, len(days))
        falling = np.linspace(120, 100, len(days))
        short = np.linspace(100, 120, 199)
        arrs = {
            "UP": (dates, rising, np.ones(len(days))),
            "DOWN": (dates, falling, np.ones(len(days))),
            "IPO": (dates[-199:], short, np.ones(199)),
            "HALT": (dates[:-1], rising[:-1], np.ones(len(days) - 1)),
        }
        count, eligible = us_breadth.ma200_participation(arrs, int(dates[-1]), arrs)
        self.assertEqual((count, eligible), (1, 2))

        # The last bar is a large future rally; the penultimate result must
        # still use only prices through the penultimate date.
        rising[-1] = 10000
        count, eligible = us_breadth.ma200_participation(arrs, int(dates[-2]), arrs)
        self.assertEqual((count, eligible), (2, 3))

    def test_backfill_labels_snapshot_vs_reconstructed(self):
        days = pd.bdate_range("2025-01-01", periods=202)
        dates = np.array([d.date().toordinal() for d in days])
        closes = np.linspace(100, 120, len(days))
        symbols = [f"S{i:03d}" for i in range(500)]
        arrs = {s: (dates, closes, np.ones(len(days))) for s in symbols}
        universe = pd.DataFrame({"yahoo": symbols})
        observed = dt.date.fromordinal(int(dates[-1])).isoformat()
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(us_breadth, "MA200_PATH", Path(folder) / "us_ma200.csv"):
            got = us_breadth.update_ma200(
                arrs, dates.tolist(), universe,
                {observed: universe}, backfill=True)
            self.assertEqual(len(got), 2)
            self.assertEqual(got.universe_basis.tolist(),
                             ["reconstructed_current", "observed_snapshot"])
            self.assertEqual(got.iloc[-1][["count", "eligible", "pct"]].tolist(),
                             [500, 500, 100.0])
            json.dumps(us_breadth.pack_breadth(got), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
