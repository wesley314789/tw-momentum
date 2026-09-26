#!/usr/bin/env python3
"""Fixed-rule TAIEX risk controls: drawdown reduction versus missed upside."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_momentum as b
from signal_entry_study import ensure_benchmark

RULES = {"price":"價格連弱3日", "breadth":"全市場多數弱勢", "volatility":"放大波動破低", "combined":"三項至少兩項"}


def participation(hist):
    """Current-day eligible TWSE/TPEx stocks above adjusted 60-bar SMA.

    No market-cap filter; unexplained-action windows cannot provide a valid
    comparison and are excluded from both numerator and denominator.
    """
    h=hist.sort_values(["code","date"]).copy()
    g=h.groupby("code",sort=False)
    ma=g.adj_close.transform(lambda s:s.rolling(60,min_periods=60).mean())
    valid=ma.notna() & h.adj_close.notna() & (g._corp.transform(lambda s:s.rolling(60,min_periods=60).sum())==0)
    h["eligible"]=valid
    h["above"]=valid & (h.adj_close>ma)
    out=h.groupby("date").agg(eligible=("eligible","sum"),above=("above","sum")).reset_index()
    out["pct60"]=out.above/out.eligible.replace(0,np.nan)*100
    return out


def indicators(index, market):
    f=index.sort_values("date").merge(market,on="date",how="left",validate="one_to_one")
    f["ma20"]=f.close.rolling(20).mean()
    f["atr14"]=b.wilder_atr(f.high.to_numpy(),f.low.to_numpy(),f.close.to_numpy(),np.zeros(len(f),dtype=bool),14)
    f["atr_pct"]=f.atr14/f.close
    f["atr_base"]=f.atr_pct.rolling(60,min_periods=60).median().shift(1)
    f["prior_low10"]=f.low.shift(1).rolling(10).min()
    f["price"]=(f.close<f.ma20).rolling(3,min_periods=3).sum()==3
    f["breadth"]=(f.pct60<50).rolling(3,min_periods=3).sum()==3
    f["vol_trigger"]=(f.close<f.prior_low10) & (f.atr_pct>1.5*f.atr_base)
    active=False; state=[]
    for trigger,close,ma in zip(f.vol_trigger,f.close,f.ma20):
        if trigger: active=True
        elif close>=ma: active=False
        state.append(active)
    f["volatility"]=state
    f["combined"]=f[["price","breadth","volatility"]].sum(axis=1)>=2
    return f


def simulate(op,cl,targets,fee=.001):
    """Hold overnight; rebalance only at next open, fees charged on traded value.

    Cash earns zero. Constant fractional controls also pay for daily rebalancing.
    End-of-study holdings are liquidated at the final close, with the same fee.
    """
    if not len(op) or len(op)!=len(cl) or len(op)!=len(targets) or not 0<=fee<1:
        raise ValueError("Invalid simulation inputs")
    cash,units=1.,0.
    nav=[]; turnover=0.; trades=0
    for opening,closing,target in zip(op,cl,targets):
        if not np.isfinite([opening,closing,target]).all() or min(opening,closing)<=0 or not 0<=target<=1:
            raise ValueError("Invalid price or exposure")
        assets=units*opening; equity=cash+assets
        wanted=target*equity-assets
        delta=wanted/(1+target*fee if wanted>=0 else 1-target*fee)
        cost=abs(delta)*fee
        cash-=delta+cost; units+=delta/opening
        if abs(delta)>equity*1e-10:
            turnover+=abs(delta)/equity; trades+=1
        nav.append(cash+units*closing)
    if units>0:
        liquidation=units*cl[-1]
        turnover+=liquidation/nav[-1];trades+=1
        nav[-1]-=liquidation*fee
    return np.asarray(nav),turnover,trades


def execution_targets(f, rule):
    """Close signal t only changes holdings at open t+1."""
    return (~f[rule]).astype(float).shift(1)


def metrics(nav,targets):
    path=np.r_[1.,nav]
    return {"total":float((nav[-1]-1)*100),"mdd":float((path/np.maximum.accumulate(path)-1).min()*100),
            "exposure":float(np.mean(targets)*100)}


def warning_events(f,rule,start,end):
    state=f[rule].to_numpy(dtype=bool)
    rises=state & ~np.r_[False,state[:-1]]
    rows=[];censored=0
    for i in np.flatnonzero(rises):
        if not start<=f.date.iloc[i]<=end:continue
        if i+20>=len(f) or f.date.iloc[i+20]>end:
            censored+=1;continue
        entry=f.open.iloc[i+1]
        downside=float(f.low.iloc[i+1:i+21].min()/entry-1)
        rows.append({"rule":rule,"signal":f.date.iloc[i],"entry":f.date.iloc[i+1],
                     "min_20d_pct":downside*100,"hit_5pct":bool(downside<=-.05),
                     "end_20d_pct":float((f.close.iloc[i+20]/entry-1)*100)})
    return rows,censored


def cash_episodes(f,targets):
    rows=[];i=0
    while i<len(targets):
        if targets[i]>0:i+=1;continue
        j=i+1
        while j<len(targets) and targets[j]==0:j+=1
        exit_price=f.open.iloc[j] if j<len(targets) else f.close.iloc[-1]
        rows.append({"exit":f.date.iloc[i],"reenter":f.date.iloc[j] if j<len(targets) else "尚未恢復",
                     "market_return_pct":float((exit_price/f.open.iloc[i]-1)*100)})
        i=j
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start",default="2023-11-06")
    ap.add_argument("--end",default="2026-09-24")
    ap.add_argument("--out",default="_market_risk_study.md")
    args=ap.parse_args()
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    hist=b.load_prices()
    hist=hist[hist.date<=args.end]
    market=participation(hist)
    index=ensure_benchmark(hist.date.min(),args.end)
    f=indicators(index[index.date<=args.end],market)
    mask=(f.date>=args.start)&(f.date<=args.end)
    evalf=f.loc[mask].reset_index(drop=True)
    required=f[["ma20","atr_base","pct60"]].notna().all(axis=1) & (f.eligible>=500)
    if not required.loc[mask].all() or not required.shift(1,fill_value=False).loc[mask].all():
        raise ValueError("Incomplete indicator warmup or insufficient stock coverage")
    op,cl=evalf.open.to_numpy(),evalf.close.to_numpy()
    targets={"buyhold":np.ones(len(evalf))}
    for rule in RULES:
        # Signal at close t can change exposure only at open t+1.
        targets[rule]=execution_targets(f,rule).loc[mask].to_numpy()
    summaries,annual,costs,warnings,warning_audit,cash_rows,paths=[],[],[],[],[],[],[]
    names={"buyhold":"全程持有",**RULES}
    for rule,target in targets.items():
        nav,turnover,trades=simulate(op,cl,target)
        summary={"rule":rule,"name":names[rule],**metrics(nav,target),"turnover":turnover,"trades":trades}
        if rule!="buyhold":
            fixed=np.full(len(target),target.mean())
            matched,_,_=simulate(op,cl,fixed)
            mm=metrics(matched,fixed)
            summary.update(matched_total=mm["total"],matched_mdd=mm["mdd"])
            ev,censored=warning_events(f,rule,args.start,args.end)
            warnings.extend(ev)
            warning_audit.append({"rule":rule,"n":len(ev),"no_5pct":sum(not x["hit_5pct"] for x in ev),"censored":censored})
            for row in cash_episodes(evalf,target):cash_rows.append({"rule":rule,**row})
        else:summary.update(matched_total=np.nan,matched_mdd=np.nan)
        summaries.append(summary)
        daily=nav/np.r_[1.,nav[:-1]]-1
        for year in sorted(evalf.date.str[:4].unique()):
            ym=(evalf.date.str[:4]==year).to_numpy()
            annual.append({"rule":rule,"year":year,"return_pct":float((np.prod(1+daily[ym])-1)*100),
                           "mdd":metrics(np.cumprod(1+daily[ym]),target[ym])["mdd"]})
        july=evalf.date.str.startswith("2026-07").to_numpy()
        summary["july_return"]=float((np.prod(1+daily[july])-1)*100)
        summary["july_mdd"]=metrics(np.cumprod(1+daily[july]),target[july])["mdd"] if july.any() else np.nan
        paths.extend({"date":date,"rule":rule,"exposure":weight,"nav":value} for date,weight,value in zip(evalf.date,target,nav))
        for fee in [0.,.002]:
            alternate,_,_=simulate(op,cl,target,fee=fee)
            costs.append({"rule":rule,"fee":fee,**metrics(alternate,target)})
        print(summary,flush=True)
    for suffix,rows in [("",summaries),("_annual",annual),("_warnings",warnings),("_cash",cash_rows),("_costs",costs),("_nav",paths)]:
        pd.DataFrame(rows).to_csv(out.with_name(out.stem+suffix+".csv"),index=False,encoding="utf-8-sig")
    f.to_csv(out.with_name(out.stem+"_indicators.csv"),index=False)
    lines=["# 市場風險訊號：能否減少回撤，代價為何？","",
           f"研究區間：{args.start}～{args.end}，{len(evalf)}交易日。以下規則在本次運算前固定，未依結果調參數。", "",
           "## 四組規則與執行方式","",
           "1. **價格連弱3日**：大盤連續3日收在20日均線下；收盤站回均線即解除。",
           "2. **全市場多數弱勢**：上市櫃合計，站上自身60日均線的股票比例連續3日低於50%；比例回到50%即解除。只用當日有行情、至少60筆有效校正價格且窗口無未明公司行為的股票，不套動能或市值條件。",
           "3. **放大波動破低**：收盤跌破前10日最低價，且ATR14／收盤價超過此前60日該波動率中位數的1.5倍；觸發後持續警戒到收盤站回20日線。",
           "4. **三項至少兩項**：上述三個警戒狀態至少兩項同時成立。",
           "- 收盤訊號，隔日開盤才切換全倉／現金；持有部位必須承受隔夜跳空，沒有用當日訊號躲掉當天跌幅。現金利息為0，不放空。",
           "- 以加權價格指數模擬單一帳戶，單邊成交名目金額收0.1%假設成本，期末出清亦計成本。這是研究用成本，不是特定ETF或期貨的實際費率；未含配息、融資、追蹤誤差。另列0／0.2%敏感度。",
           "- 同曝險基準用該規則全期平均持股比例，固定比例每日開盤再平衡並按相同成本扣費。比例由事後樣本算出，僅作分析對照，不是事前可用策略。",
           "- 最大回撤依每日收盤淨值計算，不含盤中回撤。年度與七月報酬延續全期持倉，不在年初或月初重新進場。",
           "- 每個警報是否有後續跌勢，以次日開盤起20交易日內最低價是否跌5%作固定描述；未跌5%不代表警報完全無用，事件也可能重疊，沒有把它們當作獨立樣本算顯著性。",
           "- 參數是在已知七月案例後選定，所有結果仍屬回顧性。檢查其他年份不能稱為真正未見樣本；本段歷史偏多頭，尚不足以證明長期有效。股價校正為近似，歷史停牌／下市缺漏仍有限制。","",
           "## 全期結果（單邊成本0.1%）","",
           "|規則|總報酬|最大回撤|平均持股|同曝險總報酬|同曝險最大回撤|成交次數|", "|---|---:|---:|---:|---:|---:|---:|"]
    for x in summaries:
        matched="—|—" if x["rule"]=="buyhold" else f"{x['matched_total']:+.2f}%|{x['matched_mdd']:.2f}%"
        lines.append(f"|{x['name']}|{x['total']:+.2f}%|{x['mdd']:.2f}%|{x['exposure']:.1f}%|{matched}|{x['trades']}|")
    lines += ["","## 逐年結果","","2023及2026為部分年度。","","|規則|2023|2024|2025|2026截至9/24|2026七月|","|---|---:|---:|---:|---:|---:|"]
    for x in summaries:
        vals={a['year']:a['return_pct'] for a in annual if a['rule']==x['rule']}
        lines.append(f"|{x['name']}|"+"|".join(f"{vals.get(y,np.nan):+.2f}%" for y in ["2023","2024","2025","2026"])+f"|{x['july_return']:+.2f}%|")
    lines += ["","### 各年內最大回撤","","以各年開始前淨值為起點，跨年度高點不計在這張表內。","",
              "|規則|2024|2025|2026截至9/24|2026七月內|","|---|---:|---:|---:|---:|"]
    for x in summaries:
        vals={a['year']:a['mdd'] for a in annual if a['rule']==x['rule']}
        lines.append(f"|{x['name']}|"+"|".join(f"{vals.get(y,np.nan):.2f}%" for y in ["2024","2025","2026"])+f"|{x['july_mdd']:.2f}%|")
    lines += ["","## 警報代價","","|規則|可評價警報|未出現5%續跌|近期未滿20日|避險期間反而上漲|完整避險段數|","|---|---:|---:|---:|---:|---:|"]
    for a in warning_audit:
        episodes=[c for c in cash_rows if c['rule']==a['rule'] and c['reenter']!="尚未恢復"]
        lines.append(f"|{names[a['rule']]}|{a['n']}|{a['no_5pct']}|{a['censored']}|{sum(x['market_return_pct']>0 for x in episodes)}|{len(episodes)}|")
    lines += ["","## 七月當時可見狀態","","各欄為當日收盤才知道的狀態，實際調整晚一天。全市場比例分母每日變動。","",
              "|日期|指數收盤|站上60MA比例|價格警戒|全市場警戒|波動警戒|組合警戒|","|---|---:|---:|---|---|---|---|"]
    for _,r in f[f.date.isin(["2026-06-26","2026-07-07","2026-07-09","2026-07-16","2026-07-17","2026-07-24","2026-07-28","2026-07-31"])].iterrows():
        lines.append(f"|{r.date}|{r.close:.2f}|{r.pct60:.1f}%|"+"|".join("是" if r[k] else "否" for k in RULES)+"|")
    lines += ["","## 成本敏感度","","|規則|單邊成本|總報酬|最大回撤|","|---|---:|---:|---:|"]
    for x in costs:lines.append(f"|{names[x['rule']]}|{x['fee']*100:.1f}%|{x['total']:+.2f}%|{x['mdd']:.2f}%|")
    lines += ["","行情來源：[證交所指數歷史](https://www.twse.com.tw/indicesReport/MI_5MINS_HIST?response=html)，個股沿用專案TWSE／TPEx與官方除權息資料。", "",
              f"重跑：`python scripts/market_risk_study.py --start {args.start} --end {args.end} --out _market_risk_study.md`。完整警報、避險段、淨值與指標輸出在同名CSV。",""]
    out.write_text("\n".join(lines),encoding="utf-8")
    print(f"Saved {out}")


if __name__=="__main__":main()
