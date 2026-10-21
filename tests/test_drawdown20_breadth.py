"""上市／上櫃 52 週跌深比例的市場分組與校正驗證。"""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import drawdown20_breadth as study


class Drawdown20BreadthTest(unittest.TestCase):
    @staticmethod
    def history():
        dates = pd.date_range("2024-01-01", periods=253, freq="B").strftime("%Y-%m-%d")
        rows = []
        paths = (
            ("1101", "上市", [100.] * 250 + [90., 81., 80.]),
            ("3001", "上櫃", [100.] * 250 + [90., 81., 81.]),
            ("3002", "上櫃", [100.] * 252 + [50.]),  # 當日股票分割
            ("3003", "上櫃", [100.] * 99 + [80.]),   # 歷史不足
        )
        for code, market, prices in paths:
            for date, close in zip(dates[-len(prices):], prices):
                rows.append((date, code, market, close))
        return pd.DataFrame(rows, columns=["date", "code", "market", "close"]), dates[-1]

    def test_market_counts_and_split_do_not_create_false_drawdown(self):
        history, last_date = self.history()
        actions = pd.DataFrame([{
            "date": last_date, "code": "3002", "market": "上櫃",
            "prev_close": 100., "ratio": .5,
        }])
        result = study.calculate(history, actions)
        today = result[result.date.eq(last_date)].set_index("market")
        self.assertEqual(int(today.loc["上市", "count"]), 1)
        self.assertEqual(int(today.loc["上市", "eligible"]), 1)
        self.assertEqual(float(today.loc["上市", "pct"]), 100.)
        self.assertEqual(int(today.loc["上櫃", "count"]), 0)
        self.assertEqual(int(today.loc["上櫃", "eligible"]), 2)
        self.assertEqual(float(today.loc["上櫃", "pct"]), 0.)

    def test_daily_update_preserves_older_curve(self):
        previous = pd.DataFrame([
            ("2026-10-19", "上市", 20, 100, 20.),
            ("2026-10-19", "上櫃", 30, 100, 30.),
        ], columns=study.COLS)
        fresh = pd.DataFrame([
            ("2026-10-20", "上市", 25, 100, 25.),
            ("2026-10-20", "上櫃", 35, 100, 35.),
        ], columns=study.COLS)
        with tempfile.TemporaryDirectory() as directory:
            saved = study.save_series(fresh, previous, Path(directory) / "breadth.csv", False)
        self.assertEqual(saved.date.nunique(), 2)
        self.assertEqual(len(saved), 4)


if __name__ == "__main__":
    unittest.main()
