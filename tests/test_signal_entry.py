import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import signal_entry_study as s

dates = pd.bdate_range("2026-01-01", periods=8).strftime("%Y-%m-%d").tolist()
p = {"i": dict(zip(dates, range(8))), "o": np.full(8, 100.), "c": np.full(8, 110.),
     "x": np.zeros(8, dtype=bool), "ex": np.zeros(8, dtype=bool)}
bench = pd.DataFrame({"date": dates, "open": [200.]*8, "close": [220.]*8}).set_index("date")
members = {d: {"A"} for d in dates}
t, audit = s.evaluate({"A": p}, members, dates, bench, horizon=2, max_horizon=3)
assert len(t) == 1 and t.entry_date.iloc[0] == dates[1] and t.exit_date.iloc[0] == dates[2]
np.testing.assert_allclose(t.iloc[0][["gross", "market", "net", "excess"]].astype(float), [.1, .1, .0927, -.0073])
assert not t.dividend.iloc[0]
assert s.evaluate({"A": p}, members, dates, bench, 2, 3, initial_prev={"A"})[0].empty

# Reentries overlap if they buy at the open on the original exit day.
reentry = {dates[0]: {"A"}, dates[2]: {"A"}, dates[-1]: {"A"}}
t, audit = s.evaluate({"A": p}, reentry, dates, bench, 3, 3)
assert len(t) == 1 and audit["overlap"] == 1 and audit["right_censored"] == 1
assert sum(v for k,v in audit.items() if k != "new_events") == audit["new_events"]
assert len(s.evaluate({"A": p}, reentry, dates, bench, 3, 3, no_overlap=False)[0]) == 2

# A missing future outcome must not retrospectively free the first position.
bad = {**p, "c": p["c"].copy()}
bad["c"][3] = np.nan
t, audit = s.evaluate({"A": bad}, reentry, dates, bench, 3, 3)
assert t.empty and audit["missing_exit"] == 1 and audit["overlap"] == 1

# Count an ex-date only after entry; no entitlement when buying on the ex-date.
div = {**p, "ex": p["ex"].copy()}
div["ex"][1] = True
assert not s.evaluate({"A": div}, members, dates, bench, 2, 3)[0].dividend.iloc[0]
div["ex"][2] = True
assert s.evaluate({"A": div}, members, dates, bench, 2, 3)[0].dividend.iloc[0]

# Date-weighted inference gives each date one vote, not each stock.
events = pd.DataFrame({"signal": [dates[0]]*100 + [dates[1]], "excess": [.1]*100 + [-.1]})
n, mean, lo, hi = s.cohort_interval(events, dates, block=2, repeats=100)
assert n == 2 and abs(mean) < 1e-10 and lo <= mean <= hi
print("Signal entry tests OK")
