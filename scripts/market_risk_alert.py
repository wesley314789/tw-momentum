"""Telegram alert for the price + two-participation risk rule.

Runs after the daily stock pipeline. Never records an alert as sent until
Telegram confirms receipt. With no --send it only previews the current signal.
"""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "market_risk_alert_state.json"


def participation(history, actions, adjusted):
    h = history.sort_values(["code", "date"]).drop_duplicates(
        ["code", "date"], keep="last"
    ).copy()
    if adjusted:
        a = actions.drop_duplicates(["date", "code", "market"], keep="last")
        h = h.merge(a[["date", "code", "market", "prev_close", "ratio"]],
                    on=["date", "code", "market"], how="left", validate="one_to_one")
        prev = h.groupby("code", sort=False).close.shift(1)
        event = h.ratio.notna()
        matched = event & prev.notna() & ((prev / h.prev_close - 1).abs() <= .03)
        h["_ratio"] = h.ratio.where(matched, 1.0)
        h["_factor"] = h.groupby("code", sort=False)._ratio.transform(
            lambda s: pd.Series(np.cumprod(s.to_numpy()[::-1])[::-1] / s.to_numpy(),
                                index=s.index))
        h["price"] = h.close * h._factor
        unmatched = event & ~matched
    else:
        h["price"] = h.close
        unmatched = pd.Series(False, index=h.index)
    g = h.groupby("code", sort=False)
    ma60 = g.price.transform(lambda s: s.rolling(60, min_periods=60).mean())
    jumps = g.price.pct_change().abs().gt(.105).fillna(False) | unmatched
    clean = g.apply(lambda x: jumps.loc[x.index].rolling(60, min_periods=60).sum(),
                    include_groups=False).reset_index(level=0, drop=True).sort_index().eq(0)
    eligible = ma60.notna() & h.price.notna() & clean
    h["eligible"] = eligible
    h["above"] = eligible & h.price.gt(ma60)
    out = h.groupby("date").agg(eligible=("eligible", "sum"),
                                above=("above", "sum")).reset_index()
    out["pct60"] = out.above / out.eligible.replace(0, np.nan) * 100
    return out[["date", "eligible", "pct60"]]


def signals(index, leaders, adjusted, raw):
    f = index.merge(leaders[["date", "pct"]], on="date", validate="one_to_one")
    f = f.merge(adjusted.rename(columns={"pct60": "broad_pct", "eligible": "eligible"}),
                on="date", validate="one_to_one")
    f = f.merge(raw[["date", "pct60"]].rename(columns={"pct60": "raw_broad_pct"}),
                on="date", validate="one_to_one").sort_values("date").reset_index(drop=True)
    f["price_ma20"] = f.close.rolling(20, min_periods=20).mean()
    f["price_weak"] = f.close.lt(f.price_ma20).rolling(3, min_periods=3).sum().eq(3)
    f["leader_ma20"] = f.pct.rolling(20, min_periods=20).mean()
    f["leader_10ago"] = f.pct.shift(10)
    f["leader_weak"] = f.pct.lt(f.leader_ma20) & f.pct.lt(f.leader_10ago)
    for name in ("broad", "raw_broad"):
        col = f"{name}_pct"
        f[f"{name}_ma20"] = f[col].rolling(20, min_periods=20).mean()
        f[f"{name}_10ago"] = f[col].shift(10)
        f[f"{name}_weak"] = f[col].lt(f[f"{name}_ma20"]) & f[col].lt(f[f"{name}_10ago"])
    f["active_adjusted"] = f.price_weak & f.leader_weak & f.broad_weak
    f["active_raw"] = f.price_weak & f.leader_weak & f.raw_broad_weak
    f["active_confirmed"] = f.active_adjusted & f.active_raw
    return f


def load_actions(p, latest, refresh):
    actions = pd.read_csv(p / "corporate_actions.csv", dtype={"code": str}, encoding="utf-8")
    meta_path = p / "corporate_actions_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not refresh or meta.get("end", "") >= latest:
        return actions, meta
    try:
        import corporate_actions as ca

        start = dt.date.fromisoformat(meta["end"]) + dt.timedelta(days=1)
        if start <= dt.date.fromisoformat(latest):
            new = ca.fetch_range(start, dt.date.fromisoformat(latest))
            actions = pd.concat([actions, new], ignore_index=True).drop_duplicates(
                ["date", "code", "market"], keep="last")
            actions = actions.sort_values(["date", "code", "market"])
            meta["end"] = latest
            meta["events"] = len(actions)
            meta["fetched_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            actions.to_csv(p / "corporate_actions.csv", index=False)
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # Cached factors plus the independent raw-price check remain usable.
        # Do not log transport exceptions: their URLs may contain credentials.
        pass
    return actions, meta


def evaluate(refresh_actions=False):
    p = ROOT / "data"
    history = pd.read_csv(p / "history.csv.gz", dtype={"code": str}, encoding="utf-8")
    index = pd.read_csv(p / "index.csv.gz", encoding="utf-8")
    leaders = pd.read_csv(p / "breadth.csv", encoding="utf-8")
    latest = min(history.date.max(), index.date.max(), leaders.date.max())
    if latest != max(history.date.max(), index.date.max(), leaders.date.max()):
        return {"status": "stale", "reason": "資料日期不一致", "data_date": latest}
    taipei_now = dt.datetime.now(ZoneInfo("Asia/Taipei"))
    if (taipei_now.date() - dt.date.fromisoformat(latest)).days > 4:
        return {"status": "stale", "reason": "收盤資料超過四天未更新", "data_date": latest}
    if taipei_now.hour >= 14 and latest < taipei_now.date().isoformat():
        return {"status": "stale", "reason": "今日台股收盤資料尚未齊全", "data_date": latest}
    actions, meta = load_actions(p, latest, refresh_actions)
    adj = participation(history, actions, True)
    raw = participation(history, actions, False)
    f = signals(index, leaders, adj, raw)
    if meta.get("end", "") >= latest:
        # With complete official action coverage, use the exact backtest rule.
        f["active_confirmed"] = f.active_adjusted
    if f.empty or f.date.iloc[-1] != latest:
        return {"status": "stale", "reason": "無法對齊風險指標", "data_date": latest}
    row = f.iloc[-1]
    values = {"date": latest, "taiex_close": round(float(row.close), 2),
              "taiex_ma20": round(float(row.price_ma20), 2),
              "price_below_ma20_3d": bool(row.price_weak),
              "broad_pct60": round(float(row.broad_pct), 2),
              "broad_ma20": round(float(row.broad_ma20), 2),
              "broad_10ago": round(float(row.broad_10ago), 2),
              "raw_broad_pct60": round(float(row.raw_broad_pct), 2),
              "raw_broad_ma20": round(float(row.raw_broad_ma20), 2),
              "raw_broad_10ago": round(float(row.raw_broad_10ago), 2),
              "leader_pct": round(float(row.pct), 2),
              "leader_ma20": round(float(row.leader_ma20), 2),
              "leader_10ago": round(float(row.leader_10ago), 2),
              "eligible_stocks": int(row.eligible),
              "official_actions_through": meta.get("end"),
              "active_adjusted": bool(row.active_adjusted),
              "active_raw": bool(row.active_raw),
              "active_confirmed": bool(row.active_confirmed)}
    if not np.isfinite([values[k] for k in ("taiex_ma20", "broad_ma20", "leader_ma20")]).all():
        return {"status": "stale", "reason": "指標歷史不足", "data_date": latest}
    return {"status": "ok", "values": values,
            "confirmed_series": [{"date": str(x.date), "active": bool(x.active_confirmed)}
                                 for x in f.itertuples(index=False)]}


def process(result, commit=False, state_path=STATE):
    if result["status"] != "ok":
        return result
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    values = result["values"]
    latest = values["date"]
    prior_date = state.get("last_evaluated_date", "")
    prior_active = bool(state.get("last_active", False))
    unseen = [r for r in result["confirmed_series"] if r["date"] > prior_date]
    if not prior_date:
        newly_active = bool(values["active_confirmed"])
    elif prior_date == latest:
        newly_active = bool(values["active_confirmed"] and not prior_active)
    else:
        crossed = False
        for row in unseen:
            if row["active"] and not prior_active:
                crossed = True
            prior_active = row["active"]
        newly_active = bool(values["active_confirmed"] and crossed)
    if state.get("last_alert_date") == latest:
        newly_active = False
    output = {"status": "triggered" if newly_active else "quiet", "values": values}
    if commit:
        next_state = {"last_evaluated_date": latest,
                      "last_active": bool(values["active_confirmed"]),
                      "last_alert_date": latest if newly_active else state.get("last_alert_date")}
        temp = state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(next_state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, state_path)
    return output


def message(values):
    note = ""
    if values["official_actions_through"] < values["date"]:
        note = ("\n注意：近期除權息資料尚未完全校正，已用原始與已知校正價格"
                "交叉確認警戒方向。")
    return (
        f"⚠️ 台股價格＋市場廣度轉弱警戒｜{values['date']} 收盤\n"
        f"加權指數 {values['taiex_close']:,.2f}；20MA {values['taiex_ma20']:,.2f}，"
        "已連續 3 日收低於 20MA。\n"
        f"全市場站上 60MA：{values['broad_pct60']:.2f}%"
        f"（20 日均值 {values['broad_ma20']:.2f}%；10 日前 {values['broad_10ago']:.2f}%）。\n"
        f"動能上榜比例：{values['leader_pct']:.2f}%"
        f"（20 日均值 {values['leader_ma20']:.2f}%；10 日前 {values['leader_10ago']:.2f}%）。\n"
        "三項條件首次同時成立。這是研究警戒，不是交易指令。" + note
    )


def send_telegram(values, token, chat_id):
    import requests

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": message(values)}, timeout=20)
        payload = response.json()
    except (requests.RequestException, ValueError):
        raise RuntimeError("Telegram transport failed") from None
    if response.status_code != 200 or not payload.get("ok"):
        raise RuntimeError(f"Telegram rejected message (HTTP {response.status_code})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true", help="用環境變數中的機器人資料發送警戒")
    args = parser.parse_args()
    try:
        evaluated = evaluate(refresh_actions=args.send)
        result = process(evaluated)
        if args.send and result["status"] in ("triggered", "quiet"):
            token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
            if not token or not chat_id:
                result = {"status": "unconfigured", "reason": "Telegram Secrets 尚未設定"}
            else:
                if result["status"] == "triggered":
                    send_telegram(result["values"], token, chat_id)
                # Record the date only after Telegram confirms the alert.
                process(evaluated, commit=True)
    except Exception as exc:
        result = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
