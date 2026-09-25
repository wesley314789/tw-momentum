import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import corporate_actions as ca
import us_breadth as us
import update_data as tw

hist = pd.DataFrame([
    {"date": "2026-01-02", "code": "1111", "market": "上市", "open": 100., "high": 101., "low": 99., "close": 100.},
    {"date": "2026-01-05", "code": "1111", "market": "上市", "open": 96., "high": 97., "low": 95., "close": 96.},
    {"date": "2026-01-06", "code": "1111", "market": "上市", "open": 98., "high": 99., "low": 97., "close": 98.},
])
actions = pd.DataFrame([{"date": "2026-01-05", "code": "1111", "market": "上市",
                         "prev_close": 100., "ref_price": 95., "ratio": .95}])
adjusted, audit = ca.adjust_history(hist, actions)
assert adjusted.adj_close.tolist() == [95., 96., 98.]
assert adjusted.adj_open.tolist() == [95., 96., 98.]
assert audit == {"events_seen": 1, "events_applied": 1,
                 "events_mismatched": 0, "residual_jumps": 0}
assert not adjusted._corp.any()

bad = actions.copy()
bad["prev_close"] = 80.
unadjusted, audit = ca.adjust_history(hist, bad)
assert audit["events_mismatched"] == 1
assert unadjusted.adj_close.tolist() == [100., 96., 98.]
assert unadjusted._corp.iloc[1]

dates = np.array([(dt.date(2025, 1, 1) + dt.timedelta(days=i)).toordinal()
                  for i in range(201)])
close = np.geomspace(10., 100., 201)
volume = np.full(201, 1e7)
arrs = {"AAA": (dates, close, volume), "BBB": (dates, close, volume)}
universe = pd.DataFrame({"yahoo": ["AAA"], "symbol": ["AAA"],
                         "name": ["Alpha"], "mcap": [1e9]})
bench = (dates, np.full(201, 100.))
selected = us.screen(arrs, universe, dates[-1], bench)
assert selected.symbol.tolist() == ["AAA"]
assert us.count_universe(arrs, dates[-1], {"AAA"}) == 1
tw_dates = [(dt.date(2025, 1, 1) + dt.timedelta(days=i)).isoformat()
            for i in range(201)]
tw_hist = pd.DataFrame({"date": tw_dates, "code": ["1111"] * 201,
                        "name": ["Test"] * 201, "market": ["上市"] * 201,
                        "close": close, "signal_close": close * .5,
                        "value": [1e8] * 201, "_corp": [False] * 201})
shares = pd.DataFrame({"code": ["1111"], "shares": [2.5e7]})
idx = pd.DataFrame({"date": tw_dates, "close": [100.] * 201})
assert tw.momentum_screen(tw_hist, shares, idx=idx).code.tolist() == ["1111"]
with tempfile.TemporaryDirectory() as tmp:
    us.SNAPSHOT_DIR = Path(tmp) / "snapshots"
    us.BREADTH_PATH = Path(tmp) / "breadth.csv"
    pd.DataFrame([{"date": "2026-01-01", "count": 10,
                   "universe": 500, "pct": 2.}]).to_csv(us.BREADTH_PATH, index=False)
    old = us.load_breadth()
    assert old.universe_basis.iloc[0] == "unknown_legacy"
    mixed = us.save_breadth(pd.concat([old, pd.DataFrame([
        {"date": "2026-01-02", "count": 20, "universe": 500,
         "pct": 4., "universe_basis": "observed_snapshot"}])], ignore_index=True))
    json.dumps(us.pack_breadth(mixed), allow_nan=False)
    snap_uni = pd.DataFrame({"yahoo": [f"T{i}" for i in range(500)],
                             "symbol": [f"T{i}" for i in range(500)],
                             "name": ["Test"] * 500, "mcap": [1e9] * 500})
    late = pd.Timestamp("2026-01-03 16:00", tz="America/New_York")
    assert us.save_snapshot("2026-01-02", snap_uni, late).empty
    timely = pd.Timestamp("2026-01-02 18:00", tz="America/New_York")
    snap = us.save_snapshot("2026-01-02", snap_uni, timely)
    assert len(snap) == 500 and snap.symbol.iloc[0] == "T0"
    assert len(us.save_snapshot("2026-01-02", snap_uni.iloc[0:0], timely)) == 500
    assert len(us.load_snapshots()) == 500
print("data-quality tests OK")
