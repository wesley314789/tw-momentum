#!/usr/bin/env python3
"""Does Taiwan breadth crossing below its 5/10/20-day MA precede weaker TAIEX?"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_momentum as b
from signal_entry_study import ensure_benchmark


def rebuild_breadth(hist, shares, idx):
    """Vectorized equivalent of the live raw-price momentum_screen, not adjusted backtest prices."""
    h = hist.sort_values(["code", "date"]).reset_index(drop=True).copy()
    ix = idx.sort_values("date").drop_duplicates("date", keep="last")
    loc = np.searchsorted(ix.date.to_numpy(), h.date.to_numpy(), side="right")-1
    h["_bench"] = np.where(loc >= 0, ix.close.to_numpy()[loc.clip(min=0)], np.nan)
    g = h.groupby("code", sort=False)
    means = {n: g.close.transform(lambda s: s.rolling(n).mean()) for n in [10,20,200]}
    excess = (h.close/g.close.shift(b.u.BR_PERF_DAYS) - h._bench/g._bench.shift(b.u.BR_PERF_DAYS))*100
    share_map = dict(zip(shares.code, shares.shares))
    passed = ((h.close > means[200]) & (means[10] > means[20])
              & (h.close*h.code.map(share_map) > b.u.BR_MCAP)
              & (h.value > b.u.BR_TURNOVER) & (excess > b.u.BR_EXCESS))
    h["_passed"] = passed
    series = h.groupby("date").agg(count=("_passed", "sum"), universe=("code", "size")).reset_index()
    series["pct"] = series["count"]/series.universe*100
    dates = sorted(hist.date.unique())
    series = series[series.date >= dates[199]].reset_index(drop=True)
    return series, h.loc[passed, ["date","code"]]


def cross_below(values, period):
    counts = pd.Series(values, dtype=float)
    ma = counts.rolling(period, min_periods=period).mean()
    signal = (counts.shift(1) >= ma.shift(1)) & (counts < ma)
    return ma.to_numpy(), signal.to_numpy()


def separate_events(raw, gap=20):
    out = np.zeros(len(raw), dtype=bool)
    last = -gap
    for i in np.flatnonzero(raw):
        if i-last >= gap:
            out[i] = True
            last = i
    return out


def comparison(returns, event, years):
    r, event = np.asarray(returns), np.asarray(event, dtype=bool)
    n = int(event.sum())
    if not n:
        return dict(n=0, mean=np.nan, median=np.nan, base=np.nan, delta=np.nan, down=np.nan, base_down=np.nan)
    base, base_down = 0., 0.
    for year in np.unique(years):
        mask = years == year
        weight = (event & mask).sum()/n
        base += weight*r[mask].mean()
        base_down += weight*(r[mask]<0).mean()
    return dict(n=n, mean=float(r[event].mean()*100), median=float(np.median(r[event])*100),
                base=float(base*100), delta=float((r[event].mean()-base)*100),
                down=float((r[event]<0).mean()*100), base_down=float(base_down*100))


def block_interval(returns, event, years, block=40, repeats=2000):
    """Calendar-block resampling; recompute event-year-matched baseline in every draw."""
    rng = np.random.default_rng(20260926)
    n = len(returns)
    if event.sum() < 5:
        return np.nan, np.nan
    starts = rng.integers(0,n,size=(repeats,int(np.ceil(n/block))))
    take = ((starts[:,:,None]+np.arange(block)) % n).reshape(repeats,-1)[:,:n]
    rr, ee, yy = returns[take], event[take], years[take]
    ne = ee.sum(axis=1)
    baseline_sum = np.zeros(repeats)
    for year in np.unique(years):
        mask = yy==year
        ny = mask.sum(axis=1)
        mean_y = np.divide((rr*mask).sum(axis=1),ny,out=np.zeros(repeats),where=ny>0)
        baseline_sum += mean_y*(ee & mask).sum(axis=1)
    valid = ne>0
    delta = ((rr*ee).sum(axis=1)[valid]-baseline_sum[valid])/ne[valid]*100
    return tuple(float(x) for x in np.percentile(delta,[2.5,97.5]))


def analyze(series, benchmark, start, end, label):
    series = series[series.date<=end].sort_values("date")
    calendar = benchmark.loc[(benchmark.date>=series.date.min()) & (benchmark.date<=series.date.max()),"date"].sort_values()
    if set(series.date)-set(calendar):
        raise ValueError("Breadth contains dates absent from index calendar")
    s = series.set_index("date").reindex(calendar).rename_axis("date").reset_index()
    ix = benchmark.set_index("date").reindex(s.date)
    if ix[["open","close"]].isna().any().any():
        raise ValueError("Missing exact-date index price")
    op, cl = ix.open.to_numpy(), ix.close.to_numpy()
    ma20, _ = cross_below(s["count"],20)
    # All nine comparisons share the same full-MA and full-20-day window.
    eligible = (s.date.to_numpy()>=start) & np.isfinite(ma20) & np.isfinite(np.roll(ma20,1)) & (np.arange(len(s))+20<len(s))
    eligible[0] = False
    positions = np.flatnonzero(eligible)
    years = s.date.str[:4].to_numpy()[positions]
    results, sensitivity, yearly, events, snapshots = [], [], [], [], []
    for period in [5,10,20]:
        ma, raw = cross_below(s["count"],period)
        raw = raw & eligible
        event = separate_events(raw,20)
        masks = {"全部跌破（未去重）":raw, "每日低於均線":(s["count"].to_numpy()<ma)&eligible}
        for horizon in [5,10,20]:
            ret = cl[positions+horizon]/op[positions+1]-1
            em = event[positions]
            stats = comparison(ret,em,years)
            lo, hi = block_interval(ret,em,years)
            results.append(dict(sample=label,ma=period,h=horizon,raw=int(raw.sum()),lo=lo,hi=hi,**stats))
            for year in np.unique(years):
                selected = years==year
                yearly.append(dict(sample=label,ma=period,h=horizon,year=year,**comparison(ret[selected],em[selected],years[selected])))
            for name, mask in masks.items():
                sensitivity.append(dict(sample=label,ma=period,h=horizon,kind=name,**comparison(ret,mask[positions],years)))
            close_ret = cl[positions+horizon]/cl[positions]-1
            sensitivity.append(dict(sample=label,ma=period,h=horizon,kind="跌破日收盤起算",**comparison(close_ret,em,years)))
            for i, value in zip(positions[em],ret[em]):
                events.append(dict(sample=label,ma=period,h=horizon,date=s.date.iloc[i],count=int(s["count"].iloc[i]),
                                   average=ma[i],entry=s.date.iloc[i+1],exit=s.date.iloc[i+horizon],return_pct=value*100))
        snapshots.append(dict(sample=label,ma=period,date=s.date.iloc[-1],count=int(s["count"].iloc[-1]),average=ma[-1],
                              below=bool(s["count"].iloc[-1]<ma[-1])))
    return results,sensitivity,yearly,events,snapshots, [s.date.iloc[positions[0]],s.date.iloc[positions[-1]],len(positions)]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start",default="2023-11-06")
    ap.add_argument("--end",default="2026-09-24")
    ap.add_argument("--out",default="_breadth_ma_study.md")
    args=ap.parse_args()
    hist=pd.concat([pd.read_csv(b.LONG_HIST,dtype={"code":str}),b.u.load_history()],ignore_index=True) if b.LONG_HIST.exists() else b.u.load_history()
    hist=hist[hist.date<=args.end].drop_duplicates(["date","code"],keep="last")
    shares=b.u.load_shares()
    rebuilt,members=rebuild_breadth(hist,shares,b.IDX)
    # Verify vectorization against the actual site screener before using the results.
    checkpoints=sorted(set([rebuilt.date.iloc[0],rebuilt.date.iloc[len(rebuilt)//3],rebuilt.date.iloc[len(rebuilt)*2//3],rebuilt.date.iloc[-1]]))
    for day in checkpoints:
        ref=b.u.momentum_screen(hist,shares,as_of=day,idx=b.IDX)
        codes=set(ref.code) if len(ref) else set()
        got=set(members.loc[members.date==day,"code"])
        if codes != got:
            raise ValueError(f"Vectorized screener mismatch {day}: {codes ^ got}")
        print(f"Validated {day}: {len(codes)} names",flush=True)
    saved=pd.read_csv(b.u.BREADTH_PATH)
    saved=saved[saved.date<=args.end].copy()
    overlap=rebuilt.merge(saved,on="date",suffixes=("_rebuilt","_saved"))
    mismatch=int((overlap.count_rebuilt!=overlap.count_saved).sum())
    maxdiff=int((overlap.count_rebuilt-overlap.count_saved).abs().max())
    benchmark=ensure_benchmark(min(rebuilt.date.min(),saved.date.min()),args.end)
    rows,sens,years,events,states,spans=[],[],[],[],[],[]
    for label,series in [("長期重建",rebuilt),("網站已存",saved)]:
        result=analyze(series,benchmark,args.start,args.end,label)
        for target,new in zip([rows,sens,years,events,states],result[:5]):target.extend(new)
        spans.append((label,*result[5]))
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    for suffix,values in [("",rows),("_sensitivity",sens),("_yearly",years),("_events",events),("_latest",states)]:
        pd.DataFrame(values).to_csv(out.with_name(out.stem+suffix+".csv"),index=False,encoding="utf-8-sig")
    rebuilt.to_csv(out.with_name(out.stem+"_series.csv"),index=False)
    lines=["# 台股市場廣度跌破均線後，大盤如何？","",
           "## 定義與限制","",
           "- 廣度為現有動能篩選通過檔數；5、10、20日簡單均線皆包含當天。跌破＝前日廣度≥前日均線、當日廣度<當日均線。",
           "- 沿用網站原始行情的篩選口徑，沒有拿除權息校正後的回測名單冒充網站廣度。長期依當前規則及現有股數重建，並非當時實際發布紀錄。",
           f"- 長期重建與網站已存資料重疊{len(overlap)}日，其中{mismatch}日檔數不同，最大相差{maxdiff}檔。分開報告兩套資料，不混接。重建程式已與正式篩選器核對4個日期。",
           "- 訊號在收盤後確認。主結果從次日開盤至第5／10／20個交易日收盤，進場當天算第1天。只看大盤指數，沒有交易成本、配息或部位策略。另列從訊號日收盤起算作敏感度檢查。",
           "- 每條均線的跌破事件保留第一筆，之後至少相隔20交易日才再計入，避免同一段反覆穿越、後續期間重疊。不同均線之間仍可能是同一事件。",
           "- 所有比較皆要求前日已有完整20日均線、未來20日行情完整。一般日基準依事件發生年度加權，避免把多空年度占比差異當成訊號效果。",
           "- 依官方大盤交易日曆對齊，網站缺漏的廣度不補成零，均線窗口含缺值時不納入。",
           "- 95%區間使用40交易日連續區塊重抽2,000次，每次重算同年度基準。九個均線／期間組合未作多重比較校正，樣本少、區間寬時不能下確定結論。這是歷史關聯，不能當成保證下跌的預測。","",
           "## 可評估訊號區間","", "|資料|起日|末日|一般日數|","|---|---|---|---:|"]
    for label,start,end,n in spans:lines.append(f"|{label}|{start}|{end}|{n}|")
    lines += ["","## 跌破事件（間隔20交易日）","",
              "差距＝事件後平均報酬−同年度一般日平均；單位為百分點。下跌率指持有期末指數低於起算開盤，並非期間內曾經下跌。","",
              "|資料|均線|持有日數|原始跌破|去重次數|事件平均|一般日平均|差距|差距95%區間|下跌率|一般日下跌率|",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for x in rows:
        lines.append(f"|{x['sample']}|{x['ma']}|{x['h']}|{x['raw']}|{x['n']}|{x['mean']:+.2f}%|{x['base']:+.2f}%|{x['delta']:+.2f}|[{x['lo']:+.2f}, {x['hi']:+.2f}]|{x['down']:.1f}%|{x['base_down']:.1f}%|")
    lines += ["","## 逐年檢查（長期重建）","","|均線|持有日數|年份|事件數|事件平均|一般日平均|差距|","|---:|---:|---|---:|---:|---:|---:|"]
    for x in years:
        if x['sample']=="長期重建":lines.append(f"|{x['ma']}|{x['h']}|{x['year']}|{x['n']}|{x['mean']:+.2f}%|{x['base']:+.2f}%|{x['delta']:+.2f}|")
    lines += ["","## 定義敏感度（長期重建）","","每日低於均線是狀態，全部跌破未去重會有重疊，都不能當作更多獨立證據。","",
              "|定義|均線|日數|次數|事件平均|一般日平均|差距|","|---|---:|---:|---:|---:|---:|---:|"]
    for x in sens:
        if x['sample']=="長期重建":lines.append(f"|{x['kind']}|{x['ma']}|{x['h']}|{x['n']}|{x['mean']:+.2f}%|{x['base']:+.2f}%|{x['delta']:+.2f}|")
    lines += ["","## 網站最近狀態（不代表後續結果已知）","","|日期|均線|廣度|均線值|低於均線|","|---|---:|---:|---:|---|"]
    for x in states:
        if x['sample']=="網站已存":lines.append(f"|{x['date']}|{x['ma']}|{x['count']}|{x['average']:.2f}|{'是' if x['below'] else '否'}|")
    lines += ["","大盤資料：[證交所發行量加權股價指數歷史資料](https://www.twse.com.tw/indicesReport/MI_5MINS_HIST?response=html)。",
              "",f"重跑：`python scripts/breadth_ma_study.py --start {args.start} --end {args.end} --out _breadth_ma_study.md`。需本機加長歷史；事件日期與個別報酬輸出在同名 `_events.csv`。",""]
    out.write_text("\n".join(lines),encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(f"Saved {out}")


if __name__=="__main__":main()
