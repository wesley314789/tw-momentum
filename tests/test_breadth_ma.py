import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"scripts"))
import breadth_ma_study as s

ma,cross=s.cross_below([1,2,3,1,2,2,1],3)
np.testing.assert_array_equal(cross,[False,False,False,True,False,False,True])
np.testing.assert_allclose(ma,[np.nan,np.nan,2.,2.,2.,5/3,5/3],equal_nan=True)
_,future=s.cross_below([1,2,3,1,2,2,1,999],3)
np.testing.assert_array_equal(cross,future[:-1])
raw=np.zeros(60,dtype=bool);raw[[0,5,19,20,39,40]]=True
np.testing.assert_array_equal(np.flatnonzero(s.separate_events(raw)),[0,20,40])
r=np.array([.1,.2,-.1,-.2]);years=np.array(["a","a","b","b"])
x=s.comparison(r,np.array([True,False,False,False]),years)
assert x["n"]==1
np.testing.assert_allclose([x["mean"],x["base"],x["delta"]],[10,15,-5])
rr=np.full(120,.02);ev=np.zeros(120,dtype=bool);ev[::20]=True
lo,hi=s.block_interval(rr,ev,np.repeat(["a","b"],60),repeats=100)
np.testing.assert_allclose([lo,hi],[0,0],atol=1e-12)
# End-to-end alignment: open=100 and every later close=110 must yield +10%,
# whereas incorrectly using the signal close as entry would yield zero.
days=pd.bdate_range("2025-01-01",periods=120).strftime("%Y-%m-%d")
series=pd.DataFrame({"date":days,"count":100+40*np.sin(np.arange(120)*2*np.pi/30)})
bench=pd.DataFrame({"date":days,"open":100.,"close":110.})
rows,*_=s.analyze(series,bench,days[0],days[-1],"test")
assert len(rows)==9 and all(x["n"]>0 for x in rows)
np.testing.assert_allclose([x["mean"] for x in rows],10.)
np.testing.assert_allclose([x["delta"] for x in rows],0.,atol=1e-10)
print("Breadth MA tests OK")
