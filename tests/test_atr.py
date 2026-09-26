import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import backtest_momentum as b


atr = b.wilder_atr([11, 13, 14, 13], [9, 10, 12, 11], [10, 12, 13, 12], [False]*4, 3)
assert np.isnan(atr[:2]).all()
np.testing.assert_allclose(atr[2:], [7/3, 20/9])
prefix = b.wilder_atr([11, 13, 14], [9, 10, 12], [10, 12, 13], [False]*3, 3)
np.testing.assert_allclose(prefix, atr[:3], equal_nan=True)
reset = b.wilder_atr([11]*5, [9]*5, [10]*5, [False, False, True, False, False], 2)
np.testing.assert_allclose(reset, [np.nan, 2, np.nan, np.nan, 2], equal_nan=True)

dates = ["2026-01-02", "2026-01-05", "2026-01-06"]
p = {"i": dict(zip(dates, range(3))), "name": "Test",
     "o": np.array([100., 100., 113.]), "h": np.array([101., 114., 115.]),
     "l": np.array([99., 99., 111.]), "c": np.array([100., 113., 114.]),
     "x": np.array([False]*3), "atr": np.array([2., 50., 60.])}
members = {d: {"1111"} for d in dates}
b.SMOOTH = False
b.R = .07
t = b.run({"1111": p}, members, dates, atr_multiple=2., with_fees=False)
assert len(t) == 1 and t.iloc[0].reason == "移動停利"
assert t.iloc[0].entry_date == dates[1] and t.iloc[0].exit_date == dates[2]
np.testing.assert_allclose(t.iloc[0][["risk_pct", "atr_signal", "exit_px", "r_mult"]].astype(float), [.04, 2., 112., 3.])

# Entry-day low is checked against the initial stop before its high is used.
p_low = {**p, "l": np.array([99., 95., 111.])}
t = b.run({"1111": p_low}, members, dates, atr_multiple=2., with_fees=False)
assert t.iloc[0].exit_date == dates[1]
np.testing.assert_allclose(t.iloc[0].exit_px, 96.)

# A gap through the stop fills at the opening price, not the requested stop.
p_gap = {**p, "o": np.array([100., 100., 90.]), "l": np.array([99., 99., 89.])}
t = b.run({"1111": p_gap}, members, dates, atr_multiple=2., with_fees=False)
np.testing.assert_allclose(t.iloc[0].exit_px, 90.)

# Fixed-percent mode retains the original calculation; prior membership suppresses false entries.
t = b.run({"1111": p}, members, dates, with_fees=False)
np.testing.assert_allclose(t.iloc[0][["risk_pct", "exit_px", "r_mult"]].astype(float), [.07, 113., .13/.07])
assert b.run({"1111": p}, members, dates, initial_prev={"1111"}, with_fees=False).empty
p_bad = {**p, "atr": np.array([np.nan, 50., 60.])}
t = b.run({"1111": p_bad}, members, dates, atr_multiple=2.)
assert t.empty and t.attrs["skipped_atr"] == 1
print("ATR tests OK")
