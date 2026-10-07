"""產業動能公式與時間邊界；不依賴網路或現有真實行情。"""

import unittest

import numpy as np
import pandas as pd

from scripts import sector_breadth


class SectorBreadthTest(unittest.TestCase):
    def setUp(self):
        self.days = pd.bdate_range("2026-06-01", periods=62).strftime("%Y-%m-%d").tolist()

    def stock(self, code, yesterday, today, today_volume=100, bars=62):
        prices = [90.0] * (bars - 2) + [yesterday, today]
        dates = self.days[-bars:]
        return pd.DataFrame({
            "date": dates, "code": code, "name": code,
            "market": "上市", "close": prices,
            "high": [max(100, price) for price in prices],
            "volume": [100.0] * (bars - 1) + [today_volume],
        })

    def test_two_sectors_hand_calculated_percentages_and_pp_delta(self):
        # A: 昨天 1/3 強，今天 2/3；今天 2/3 靠近 Pivot、1/3 確認突破。
        # B: 昨天 2/3 強，今天 1/3；今天 1/3 靠近 Pivot、0 突破。
        hist = pd.concat([
            self.stock("A1", 90, 96), self.stock("A2", 96, 101, 150),
            self.stock("A3", 90, 90), self.stock("B1", 96, 90),
            self.stock("B2", 96, 96), self.stock("B3", 90, 90),
        ], ignore_index=True)
        mapping = {code: code[0] for code in ("A1", "A2", "A3", "B1", "B2", "B3")}
        output = sector_breadth.calculate(hist, mapping, lambda day: 100.0)
        a, b = (next(s for s in output["sectors"] if s["sector"] == x) for x in ("A", "B"))
        self.assertEqual((a["strong_count"], a["pivot_count"], a["breakout_count"]), (2, 2, 1))
        self.assertEqual((a["strong_pct"], a["pivot_pct"], a["breakout_pct"]),
                         (66.67, 66.67, 33.33))
        self.assertEqual((a["previous_strong_pct"], a["delta_1d_pp"]), (33.33, 33.33))
        self.assertEqual((b["strong_count"], b["pivot_count"], b["breakout_count"]), (1, 1, 0))
        self.assertEqual((b["strong_pct"], b["pivot_pct"], b["breakout_pct"]),
                         (33.33, 33.33, 0.0))
        self.assertEqual((b["previous_strong_pct"], b["delta_1d_pp"]), (66.67, -33.33))
        self.assertEqual(output["sectors"][0]["sector"], "A")
        self.assertEqual(a["stocks"][0]["code"], "A2")
        self.assertEqual(a["stocks"][0]["pivot_status"], "FRESH_BREAKOUT")
        self.assertEqual(a["stocks"][0]["strong_score"], 6)
        self.assertEqual(next(x for x in a["stocks"] if x["code"] == "A1")["strong_score"], 5)
        self.assertEqual(next(x for x in a["stocks"] if x["code"] == "A1")["strong_change"], 1)
        self.assertEqual(next(x for x in b["stocks"] if x["code"] == "B1")["strong_change"], -1)

    def test_pivot_and_breakout_volume_exclude_today(self):
        stock = self.stock("A1", 96, 101, 150)
        row = sector_breadth.stock_at(stock, len(stock) - 1, lambda day: 100.0)
        self.assertTrue(row["fresh_breakout"])  # 150 / 昨日前 20 日均量 100 = 1.5
        stock.loc[stock.index[-1], "high"] = 999
        self.assertTrue(sector_breadth.stock_at(stock, len(stock) - 1,
                                                lambda day: 100.0)["fresh_breakout"])
        stock.loc[stock.index[-1], "volume"] = 120
        self.assertFalse(sector_breadth.stock_at(stock, len(stock) - 1,
                                                 lambda day: 100.0)["fresh_breakout"])
        # 成交量暴增或縮小不能改變分母。
        self.assertEqual(sector_breadth.breakout.previous_volume_average(
            stock["volume"].to_numpy(dtype=float)), 100)

    def test_missing_history_and_non_finite_rows_do_not_enter_denominator(self):
        valid = self.stock("A1", 96, 101)
        short = self.stock("A2", 90, 101, bars=30)
        broken = self.stock("A3", 90, 101)
        broken.loc[broken.index[-1], "close"] = np.nan
        hist = pd.concat([valid, short, broken], ignore_index=True)
        out = sector_breadth.calculate(hist, {x: "A" for x in ("A1", "A2", "A3")},
                                       lambda day: 100.0)
        self.assertEqual(out["valid_stocks"], 1)
        self.assertEqual(out["sectors"][0]["total_stocks"], 1)
        self.assertTrue(out["sectors"][0]["small_sample"])
        self.assertTrue(np.isfinite(out["sectors"][0]["strong_pct"]))


if __name__ == "__main__":
    unittest.main()
