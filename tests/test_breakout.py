"""Breakout 的時間邊界、分類、趨勢及資料不足檢查。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import breakout as b


def market(close_today=150.4, volume_today=40, n=252):
    dates = pd.bdate_range("2025-01-01", periods=n).strftime("%Y-%m-%d")
    close = 100 + np.arange(n)*.2
    close[-21:-6] -= 4  # 讓最近20日相對大盤真的超過5%
    close[-1] = close_today
    high = close + 2
    low = close - 2
    high[-6:-1] = close[-6:-1]+.5
    low[-6:-1] = close[-6:-1]-.5
    high[-1] = 158  # 今日創 intraday 高；不得因此抬高 Pivot
    low[-1] = close_today-.5
    volume = np.full(n,100.)
    volume[-6:-1] = 40
    volume[-1] = volume_today
    hist = pd.DataFrame({"date":dates,"code":"1234","name":"測試股","market":"上市",
                         "close":close,"high":high,"low":low,"volume":volume})
    idx = pd.DataFrame({"date":dates,"close":100+np.arange(n)*.01})
    return hist,idx


hist,idx=market()
pivot=b.prior_pivot(hist.high.to_numpy(),60)
assert 150.4 < pivot < 151.5
np.testing.assert_allclose(pivot,b.prior_pivot(hist.assign(high=hist.high.where(hist.index!=251,999)).high.to_numpy(),60))
np.testing.assert_allclose(b.previous_volume_average(hist.volume.to_numpy()),85.)
spike=hist.volume.to_numpy().copy();spike[-1]=999999
np.testing.assert_allclose(b.previous_volume_average(spike),85.)

watch,fresh=b.scan(hist,idx)
assert len(watch)==1 and not fresh
assert watch[0]["pivot"]==round(pivot,2)
assert isinstance(watch[0]["contraction"],bool)  # flag 只是標記，不參與排除
assert watch[0]["vol_ratio"]<.8 and watch[0]["atr_ratio"]<.8
irregular=hist.copy();irregular.loc[249,"low"]=130
loose_atr=b.BreakoutConfig(max_atr5_atr20=10)
irregular_watch,_=b.scan(irregular,idx,loose_atr)
assert len(irregular_watch)==1 and irregular_watch[0]["contraction"] is False

hist,idx=market(close_today=pivot*1.01,volume_today=500)
watch,fresh=b.scan(hist,idx)
assert not watch and len(fresh)==1
assert fresh[0]["breakout_pct"]==round(1.,2)
np.testing.assert_allclose(fresh[0]["breakout_volume_ratio"],round(500/85,2))
assert fresh[0]["vol_ratio"]<.8  # 突破日整理期量比排除今日500的大量
normal_volume=hist.copy();normal_volume.loc[normal_volume.index[-21:-1],"volume"]=100
_,unconstrained=b.scan(normal_volume,idx)
assert len(unconstrained)==1 and unconstrained[0]["vol_ratio"]==1.0
not_enough_volume=hist.copy();not_enough_volume.loc[not_enough_volume.index[-1],"volume"]=100
assert b.scan(not_enough_volume,idx)==([],[])
too_extended=hist.copy();too_extended.loc[too_extended.index[-1],"close"]=pivot*1.04
assert b.scan(too_extended,idx)==([],[])

# Prefix invariance: adding later bars must not change the earlier signal.
old,idx=market()
next_day=old.iloc[[-1]].copy();next_day["date"]="2026-01-01"
next_day["close"]=999;next_day["high"]=1000;next_day["volume"]=100000
out1=b.scan(old,idx)
out2=b.scan(pd.concat([old,next_day],ignore_index=True).query("date <= @old.date.iloc[-1]"),idx)
assert out1==out2

# Bars不足252，即使趨勢強也不入選；RS門檻確實可調。
hist,idx=market(n=251)
assert b.scan(hist,idx)==([],[])
hist,idx=market()
assert b.scan(hist,idx,b.BreakoutConfig(min_rs20=100))==([],[])
print("Breakout tests OK")
