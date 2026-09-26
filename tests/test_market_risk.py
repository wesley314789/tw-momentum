"""Execution timing, costs, causal indicators, and event-window tests."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"scripts"))
import market_risk_study as s

# A close warning cannot avoid the next opening gap. An already-owned unit
# drops from 100 to 80 before we can sell, but avoids the later decline to 70.
f=pd.DataFrame({"risk":[False,False,True,True]})
target=s.execution_targets(f,"risk").iloc[1:].to_numpy()
np.testing.assert_array_equal(target,[1,1,0])
nav,_,_=s.simulate([100,100,80],[100,100,70],target,0)
np.testing.assert_allclose(nav,[1,1,.8])

# Full ownership is identical to last close / first open despite opening gaps.
nav,_,_=s.simulate([100,90,120],[110,130,125],[1,1,1],0)
np.testing.assert_allclose(nav,[1.1,1.3,1.25])
nav,_,_=s.simulate([100,90],[110,80],[0,0],.001)
np.testing.assert_allclose(nav,[1,1])
nav,_,trades=s.simulate([100,100],[100,100],[1,1],.001)
np.testing.assert_allclose(nav[-1],.999/1.001)
assert trades==2
# Fractional exposure uses post-fee equity, and the final sale is also charged.
nav,_,_=s.simulate([100],[110],[.5],.001)
assets=.5/1.0005
np.testing.assert_allclose(nav[-1],1-assets-assets*.001+assets*1.1*.999)
try:
    s.simulate([100,101],[100],[1,1])
    raise AssertionError("Mismatched vectors must not silently truncate")
except ValueError:pass

days=pd.bdate_range("2024-01-01",periods=200).strftime("%Y-%m-%d")
close=100+np.arange(200)*.02+5*np.sin(np.arange(200)/7)
idx=pd.DataFrame({"date":days,"open":close+.2,"close":close,"high":close+1,"low":close-1})
market=pd.DataFrame({"date":days,"pct60":50+10*np.sin(np.arange(200)/9),"eligible":1000})
full=s.indicators(idx,market)
prefix=s.indicators(idx.iloc[:140],market.iloc[:140])
pd.testing.assert_frame_equal(full.iloc[:140].reset_index(drop=True),prefix)
assert full.prior_low10.iloc[140]==idx.low.iloc[130:140].min()
np.testing.assert_allclose(full.atr_base.iloc[140],full.atr_pct.iloc[80:140].median())
assert full.price.equals((full.close<full.ma20).rolling(3,min_periods=3).sum()==3)

shock=idx.copy()
shock[["open","close"]]=100.
shock["high"]=101.;shock["low"]=99.
shock.loc[100,["close","low"]]=[80.,79.]
v=s.indicators(shock,market)
assert v.vol_trigger.iloc[100] and v.volatility.iloc[100]
assert not v.volatility.iloc[101]  # recovers above SMA20, clears warning

# Unexplained corporate action invalidates that stock's full comparison window.
parts=[]
for code in ["A","B"]:
    h=pd.DataFrame({"code":code,"date":days[:100],"adj_close":np.arange(100)+100.,"_corp":False})
    if code=="B":h.loc[65,"_corp"]=True
    parts.append(h)
hist=pd.concat(parts,ignore_index=True)
p=s.participation(hist)
assert p.eligible.iloc[58]==0 and p.eligible.iloc[59]==2
assert p.eligible.iloc[65]==1 and p.pct60.iloc[65]==100
pd.testing.assert_frame_equal(s.participation(hist[hist.date<=days[80]]),p.iloc[:81].reset_index(drop=True))

# Outcome uses next open, excludes signal-day low, includes exactly 20 days.
events=pd.DataFrame({"date":days[:45],"open":100.,"close":100.,"low":99.,"risk":False})
events.loc[0,"risk"]=True
events.loc[0,"low"]=1.
events.loc[21,"low"]=1.
rows,censored=s.warning_events(events,"risk",days[0],days[44])
assert len(rows)==1 and censored==0 and not rows[0]["hit_5pct"]
events.loc[20,"low"]=94.
rows,_=s.warning_events(events,"risk",days[0],days[44])
assert rows[0]["hit_5pct"]
events.loc[44,"risk"]=True
_,censored=s.warning_events(events,"risk",days[0],days[44])
assert censored==1
print("Market risk tests OK")
