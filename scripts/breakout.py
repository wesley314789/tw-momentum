"""台股 Pivot 整理與當日突破；只用訊號日及以前的日線資料。"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BreakoutConfig:
    pivot_days: int = 60
    max_distance_from_pivot: float = .05
    max_breakout_extension: float = .03
    max_distance_from_52w_high: float = .15
    min_above_52w_low: float = .30
    min_rs20: float = 5.0             # 百分點，非倍數
    max_vol5_vol20: float = .80
    max_atr5_atr20: float = .80
    min_breakout_volume_ratio: float = 1.30


CONFIG = BreakoutConfig()
REQUIRED_BARS = 252
RS_DAYS = 20  # 與現有台股動能篩選的 BR_PERF_DAYS 相同


def prior_pivot(high: np.ndarray, days: int) -> float:
    """過去 days 根最高價，今日 high 一律排除。"""
    return float(np.max(high[-days-1:-1]))


def previous_volume_average(volume: np.ndarray, days: int = 20) -> float:
    """突破前 days 根平均量，今日量一律排除。"""
    return float(np.mean(volume[-days-1:-1]))


def scan(hist: pd.DataFrame, idx: pd.DataFrame,
         config: BreakoutConfig = CONFIG) -> tuple[list[dict], list[dict]]:
    """回傳 (watchlist, fresh_breakout)，不改原資料或原 scanner。

    突破日的 vol5/vol20、ATR 比與 range 為「突破前」整理狀態；
    今日量另與突破前20日均量相比。RS20 沿用 index_lookup 的同日期 as-of 口徑。
    """
    try:
        from update_data import index_lookup
    except ModuleNotFoundError:
        from scripts.update_data import index_lookup

    if hist.empty or idx.empty:
        return [], []
    today = hist["date"].max()
    ix_at = index_lookup(idx)
    ix_today = ix_at(today)
    if ix_today is None or ix_today <= 0:
        raise ValueError(f"{today} 沒有加權指數收盤，無法計算 breakout RS20")

    watch, fresh = [], []
    for code, g in hist.sort_values("date").groupby("code", sort=False):
        if g["date"].iloc[-1] != today or len(g) < max(REQUIRED_BARS, config.pivot_days + 1):
            continue
        tail = g.tail(max(REQUIRED_BARS, config.pivot_days + 1))
        c = tail["close"].to_numpy(dtype=float)
        h = tail["high"].to_numpy(dtype=float)
        l = tail["low"].to_numpy(dtype=float)
        v = tail["volume"].to_numpy(dtype=float)
        if (not np.isfinite([c, h, l, v]).all() or np.min(c) <= 0
                or np.min(l) <= 0 or np.min(v) < 0):
            continue
        # 52週、均線與 Pivot 是彼此獨立的視窗；Pivot 明確不含今日 high。
        ma50, ma150, ma200 = (c[-n:].mean() for n in (50, 150, 200))
        ma200_old = c[-220:-20].mean()  # 正好20根 K 棒前的200MA
        high52, low52 = h[-REQUIRED_BARS:].max(), l[-REQUIRED_BARS:].min()
        pivot = prior_pivot(h, config.pivot_days)
        close = c[-1]
        if not (close > ma50 > ma150 > ma200 > ma200_old
                and close >= high52 * (1-config.max_distance_from_52w_high)
                and close >= low52 * (1+config.min_above_52w_low)
                and pivot > 0):
            continue

        ix_base = ix_at(tail["date"].iloc[-RS_DAYS-1])
        if ix_base is None or ix_base <= 0:
            continue
        rs20 = (close/c[-RS_DAYS-1] - ix_today/ix_base) * 100
        if rs20 < config.min_rs20:
            continue

        # True Range 採 high/low/前收；fresh 的整理期指標全部截至昨天。
        tr = np.maximum.reduce([h[1:]-l[1:],
                                abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])])
        is_watch = pivot * (1-config.max_distance_from_pivot) <= close <= pivot
        is_fresh = (c[-2] <= pivot < close
                    and close <= pivot * (1+config.max_breakout_extension))
        if not (is_watch or is_fresh):
            continue

        vol20 = v[-20:].mean() if is_watch else previous_volume_average(v)
        vol5 = v[-5:].mean() if is_watch else v[-6:-1].mean()
        atr20 = tr[-20:].mean() if is_watch else tr[-21:-1].mean()
        atr5 = tr[-5:].mean() if is_watch else tr[-6:-1].mean()
        if vol20 <= 0 or atr20 <= 0:
            continue
        vol_ratio, atr_ratio = vol5/vol20, atr5/atr20
        ranges = []
        for n in (5, 10, 20):
            highs = h[-n:] if is_watch else h[-n-1:-1]
            lows = l[-n:] if is_watch else l[-n-1:-1]
            ranges.append((highs.max()/lows.min()-1)*100)
        row = {
            "code": str(code), "name": str(g["name"].iloc[-1]),
            "market": str(g["market"].iloc[-1]), "close": round(close, 2),
            "pivot": round(float(pivot), 2),
            "distance_pct": round((close/pivot-1)*100, 2),
            "off_high_pct": round((close/high52-1)*100, 2),
            "rs20": round(float(rs20), 2),
            "vol_ratio": round(float(vol_ratio), 2),
            "atr_ratio": round(float(atr_ratio), 2),
            "range5": round(ranges[0], 2), "range10": round(ranges[1], 2),
            "range20": round(ranges[2], 2),
            "contraction": bool(ranges[0] < ranges[1] < ranges[2]),
            "ma50": round(float(ma50), 2), "ma150": round(float(ma150), 2),
            "ma200": round(float(ma200), 2),
        }
        if is_watch:
            if vol_ratio <= config.max_vol5_vol20 and atr_ratio <= config.max_atr5_atr20:
                watch.append((1-close/pivot, -rs20, vol_ratio, row))
        else:
            breakout_volume_ratio = v[-1]/vol20
            if breakout_volume_ratio >= config.min_breakout_volume_ratio:
                row["breakout_pct"] = round((close/pivot-1)*100, 2)
                row["breakout_volume_ratio"] = round(float(breakout_volume_ratio), 2)
                fresh.append((close/pivot-1, -rs20, -breakout_volume_ratio, row))

    watch.sort(key=lambda x: x[:3])
    fresh.sort(key=lambda x: x[:3])
    return [x[3] for x in watch], [x[3] for x in fresh]
