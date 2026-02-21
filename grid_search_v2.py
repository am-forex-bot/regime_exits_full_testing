#!/usr/bin/env python3
"""
Grid Search Backtester v2 — Standalone, no dependencies on Streamlit apps.
Searches TP/SL × day-of-week × 30-min time window with walk-forward validation.

Uses bid/ask prices throughout:
  - BUY enters at ask_close, exits tracked on bid (high/low/close)
  - SELL enters at bid_close, exits tracked on ask (high/low/close)

Usage:
  python grid_search_v2.py --data-dir "C:\\path\\to\\parquets"
  python grid_search_v2.py --mode relaxed --data-dir "C:\\path"
"""

import argparse
import glob
import logging
import os
import sys
import time as time_mod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

try:
    import pytz
    _TZ_TOKYO = pytz.timezone('Asia/Tokyo')
    _TZ_LONDON = pytz.timezone('Europe/London')
    _TZ_NY = pytz.timezone('America/New_York')
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False

# ── Logging ──
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ======================================================================
# CONFIG
# ======================================================================

BOT_CONFIG = {
    'CONFIDENCE_THRESHOLD': 0.60,
    'MIN_ATR_PIPS': 5,
    'MAX_ATR_PIPS': 100,
    'MTF_WEIGHTS': {'M1': 0.05, 'M5': 0.20, 'M15': 0.30, 'H1': 0.25, 'H4': 0.20},
    'HURST_THRESHOLD': 0.52,
    'HURST_WINDOW': 100,
    'HURST_MIN_WINDOW': 10,
    'HURST_MAX_WINDOW': 50,
    'HURST_NUM_WINDOWS': 15,
}

# Grid search parameters
LEVELS = np.arange(10, 201, 5, dtype=np.float64)   # 10, 15, 20 ... 200 pips
N_LEVELS = len(LEVELS)
N_WINDOWS = 48                 # 30-min UTC slots per day
MAX_HOLD_HOURS = 168           # 7 days
MIN_TRADES = 20                # minimum trades for a slot to count
SENTINEL = np.iinfo(np.int64).max
DOW_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']


# ======================================================================
# DATA LOADING
# ======================================================================

def load_pair_data(filepath: str) -> pd.DataFrame:
    """Load parquet, set time index, keep ALL columns including bid/ask."""
    df = pd.read_parquet(filepath)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.set_index('time')
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Keep float64 for bid/ask precision — don't downcast
    return df


def build_timeframes(df_5s: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Resample 5s data to M1, M5, M15, H1, H4. Carries bid/ask OHLC through."""
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}

    if 'volume' in df_5s.columns:
        agg['volume'] = 'sum'

    # Only add bid/ask columns that actually exist in the data
    ba_cols = {
        'bid_open': 'first', 'bid_high': 'max', 'bid_low': 'min', 'bid_close': 'last',
        'ask_open': 'first', 'ask_high': 'max', 'ask_low': 'min', 'ask_close': 'last',
    }
    has_bidask = all(c in df_5s.columns for c in ['bid_high', 'bid_low', 'bid_close',
                                                    'ask_high', 'ask_low', 'ask_close'])
    if has_bidask:
        for col, func in ba_cols.items():
            if col in df_5s.columns:
                agg[col] = func

    tfs = {}
    for label, rule in [('M1', '1min'), ('M5', '5min'), ('M15', '15min'),
                         ('H1', '1h'), ('H4', '4h')]:
        df = df_5s.resample(rule).agg(agg).dropna(subset=['close'])
        if 'volume' not in df.columns:
            df['volume'] = 0.0
        if has_bidask:
            df['spread'] = (df_5s['ask_close'] - df_5s['bid_close']).resample(rule).mean()
        tfs[label] = df

    return tfs


# ======================================================================
# INDICATORS (pure numpy — no TA-Lib required, uses it if available)
# ======================================================================

def _ema_np(arr, span):
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out

def _rsi_np(close, period=14):
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    return 100 - 100 / (1 + rs)

def _macd_np(close, fast=12, slow=26, sig=9):
    macd = _ema_np(close, fast) - _ema_np(close, slow)
    signal = _ema_np(macd, sig)
    return macd, signal, macd - signal

def _bbands_np(close, period=20, nstd=2):
    s = pd.Series(close)
    mid = s.rolling(period).mean().values
    sd = s.rolling(period).std().values
    return mid + nstd * sd, mid, mid - nstd * sd

def _atr_np(high, low, close, period=14):
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    atr = np.full(len(high), np.nan)
    atr[period] = np.mean(tr[:period])
    for i in range(period + 1, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i-1]) / period
    return atr


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add full indicator suite to a timeframe dataframe."""
    if len(df) < 50:
        return df

    c = df['close'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    lo = df['low'].values.astype(np.float64)
    v = df['volume'].values.astype(np.float64) if 'volume' in df.columns else np.zeros(len(df))

    if HAS_TALIB:
        df['ema_9'] = talib.EMA(c, timeperiod=9)
        df['ema_21'] = talib.EMA(c, timeperiod=21)
        df['sma_20'] = talib.SMA(c, timeperiod=20)
        df['sma_50'] = talib.SMA(c, timeperiod=50)
        df['rsi'] = talib.RSI(c, timeperiod=14)
        macd, macd_sig, macd_hist = talib.MACD(c, 12, 26, 9)
        df['macd'] = macd; df['macd_signal'] = macd_sig; df['macd_hist'] = macd_hist
        upper, middle, lower = talib.BBANDS(c, 20, 2, 2)
        df['bb_upper'] = upper; df['bb_middle'] = middle; df['bb_lower'] = lower
        df['atr'] = talib.ATR(h, lo, c, timeperiod=14)
        df['momentum'] = c - np.roll(c, 10)
        df.iloc[:10, df.columns.get_loc('momentum')] = np.nan
        if v.sum() > 0:
            vol_sma = talib.SMA(v, timeperiod=20)
            df['volume_ratio'] = np.where(vol_sma > 0, v / vol_sma, 1.0)
        else:
            df['volume_ratio'] = 1.0
    else:
        df['ema_9'] = _ema_np(c, 9)
        df['ema_21'] = _ema_np(c, 21)
        df['sma_20'] = pd.Series(c).rolling(20).mean().values
        df['sma_50'] = pd.Series(c).rolling(50).mean().values
        df['rsi'] = _rsi_np(c, 14)
        df['macd'], df['macd_signal'], df['macd_hist'] = _macd_np(c, 12, 26, 9)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = _bbands_np(c, 20, 2)
        df['atr'] = _atr_np(h, lo, c, 14)
        df['momentum'] = c - np.roll(c, 10)
        df.iloc[:10, df.columns.get_loc('momentum')] = np.nan
        df['volume_ratio'] = 1.0

    return df


# ======================================================================
# SIGNAL COMPONENTS
# ======================================================================

def get_session_quality(ts: pd.Timestamp) -> Tuple[str, float]:
    """Session detection from UTC timestamp with DST handling."""
    if ts.weekday() >= 5:
        return 'weekend', 0.0

    if HAS_PYTZ:
        utc_ts = ts.astimezone(pytz.utc) if ts.tzinfo else pytz.utc.localize(ts)
        tokyo_hour = utc_ts.astimezone(_TZ_TOKYO).hour
        london_hour = utc_ts.astimezone(_TZ_LONDON).hour
        ny_hour = utc_ts.astimezone(_TZ_NY).hour
    else:
        tokyo_hour = (ts.hour + 9) % 24
        london_hour = ts.hour
        ny_hour = (ts.hour - 5) % 24

    sessions = []
    if 9 <= tokyo_hour < 18: sessions.append(('tokyo', 0.7))
    if 8 <= london_hour < 17: sessions.append(('london', 0.9))
    if 8 <= ny_hour < 17: sessions.append(('ny', 0.8))

    if sessions:
        names = [s[0] for s in sessions]
        if 'london' in names and 'ny' in names: return 'london_ny_overlap', 1.0
        if 'tokyo' in names and 'london' in names: return 'tokyo_london_overlap', 0.85
        return max(sessions, key=lambda x: x[1])

    return 'dead', 0.3


def compute_mtf_bias(tfs: Dict[str, pd.DataFrame], m5_index: pd.DatetimeIndex) -> np.ndarray:
    """Weighted multi-timeframe bias aligned to M5 bars."""
    weights = BOT_CONFIG['MTF_WEIGHTS']
    total_w = sum(weights.values())
    n = len(m5_index)
    bias = np.zeros(n, dtype=np.float64)

    tf_data = {}
    for tf_name in weights:
        df = tfs.get(tf_name)
        if df is None or len(df) < 22:
            continue
        ema9 = df['ema_9'].values; ema21 = df['ema_21'].values; close = df['close'].values
        sig = np.where((ema9 > ema21) & (close > ema9), 1.0,
                       np.where((ema9 < ema21) & (close < ema9), -1.0, 0.0))
        tf_data[tf_name] = (df.index.asi8, sig)

    m5_ns = m5_index.asi8
    for tf_name, w in weights.items():
        if tf_name not in tf_data:
            continue
        tf_times, tf_sigs = tf_data[tf_name]
        idxs = np.clip(np.searchsorted(tf_times, m5_ns, side='right') - 1, 0, len(tf_sigs) - 1)
        bias += tf_sigs[idxs] * w

    if total_w > 0 and total_w != 1.0:
        bias /= total_w
    return np.clip(bias, -1.0, 1.0)


def compute_order_flow(m1_df: pd.DataFrame, m5_index: pd.DatetimeIndex) -> np.ndarray:
    """Order flow bias from M1 data aligned to M5 bars."""
    n = len(m5_index)
    flow = np.zeros(n, dtype=np.float64)
    m1c = m1_df['close'].values.astype(np.float64)
    m1o = m1_df['open'].values.astype(np.float64)
    m1h = m1_df['high'].values.astype(np.float64)
    m1l = m1_df['low'].values.astype(np.float64)
    m1t = m1_df.index.asi8
    m5ns = m5_index.asi8

    for i in range(n):
        idx = np.searchsorted(m1t, m5ns[i], side='right') - 1
        if idx < 60:
            continue
        s, e = idx - 59, idx + 1
        c, o, h, lo = m1c[s:e], m1o[s:e], m1h[s:e], m1l[s:e]
        vwap = ((h + lo + c) / 3).mean()
        bp = np.sum(c > o) / len(c)

        rej_up = rej_down = 0
        for j in range(-10, -1):
            body = abs(c[j] - o[j])
            if body > 0:
                if h[j] - max(c[j], o[j]) > body * 2: rej_up += 1
                if min(c[j], o[j]) - lo[j] > body * 2: rej_down += 1

        bias = 0.0
        if c[-1] > vwap and bp > 0.6: bias = bp
        elif c[-1] < vwap and bp < 0.4: bias = -(1 - bp)
        if rej_up > 3: bias -= 0.3
        if rej_down > 3: bias += 0.3
        flow[i] = np.clip(bias, -1.0, 1.0)

    return flow


def compute_trend_strength(m5_df: pd.DataFrame) -> np.ndarray:
    """Trend strength score from M5 indicators."""
    n = len(m5_df)
    ts = np.zeros(n, dtype=np.float64)
    ema9 = m5_df['ema_9'].values; ema21 = m5_df['ema_21'].values
    sma20 = m5_df['sma_20'].values; close = m5_df['close'].values
    macd = m5_df['macd'].values; macd_sig = m5_df['macd_signal'].values
    mom = m5_df['momentum'].values

    for i in range(n):
        score = 0.0
        if not np.isnan(ema9[i]) and not np.isnan(ema21[i]):
            score += 0.3 if ema9[i] > ema21[i] else -0.3
        if not np.isnan(sma20[i]):
            score += 0.2 if close[i] > sma20[i] else -0.2
        if not np.isnan(macd[i]) and not np.isnan(macd_sig[i]):
            score += 0.25 if macd[i] > macd_sig[i] else -0.25
        if not np.isnan(mom[i]):
            score += 0.25 if mom[i] > 0 else -0.25
        ts[i] = np.clip(score, -1.0, 1.0)
    return ts


def _compute_single_hurst(returns: np.ndarray) -> float:
    """R/S Hurst exponent for a window of log returns."""
    cfg = BOT_CONFIG
    n_ret = len(returns)
    if n_ret < cfg['HURST_MIN_WINDOW'] * 2:
        return 0.5
    max_w = min(cfg['HURST_MAX_WINDOW'], n_ret // 2)
    if max_w < cfg['HURST_MIN_WINDOW']:
        return 0.5

    wsizes = np.unique(np.logspace(
        np.log10(cfg['HURST_MIN_WINDOW']), np.log10(max_w), cfg['HURST_NUM_WINDOWS']).astype(int))
    if len(wsizes) < 3:
        return 0.5

    rs_vals = []
    for w in wsizes:
        rs_list = []
        for j in range(n_ret // w):
            chunk = returns[j * w:(j + 1) * w]
            dev = np.cumsum(chunk - np.mean(chunk))
            R = np.max(dev) - np.min(dev)
            S = np.std(chunk, ddof=1)
            if S > 1e-10:
                rs_list.append(R / S)
        if rs_list:
            rs_vals.append((w, np.mean(rs_list)))

    if len(rs_vals) < 3:
        return 0.5

    lx = np.log([x[0] for x in rs_vals])
    ly = np.log([x[1] for x in rs_vals])
    n = len(lx)
    denom = n * np.dot(lx, lx) - np.sum(lx) ** 2
    if abs(denom) < 1e-10:
        return 0.5
    return float(np.clip((n * np.dot(lx, ly) - np.sum(lx) * np.sum(ly)) / denom, 0.0, 1.0))


def compute_rolling_hurst(h1_df: pd.DataFrame, m5_index: pd.DatetimeIndex) -> np.ndarray:
    """Rolling Hurst on H1 data, mapped to M5 bars."""
    h1c = h1_df['close'].values.astype(np.float64)
    h1t = h1_df.index.asi8
    window = BOT_CONFIG['HURST_WINDOW']
    n_h1 = len(h1c)
    h1_hurst = np.full(n_h1, 0.5)
    for i in range(window, n_h1):
        rets = np.diff(np.log(h1c[i - window:i]))
        h1_hurst[i] = _compute_single_hurst(rets)

    m5ns = m5_index.asi8
    idxs = np.clip(np.searchsorted(h1t, m5ns, side='right') - 1, 0, n_h1 - 1)
    return h1_hurst[idxs]


# ======================================================================
# STRATEGIES (exact bot replicas)
# ======================================================================

def _momentum_pullback(i, close, high, low, rsi, mtf_bias, order_flow,
                        trend_strength, session_quality, pip_mult):
    if i < 12:
        return None, 0
    mom_pips = (close[i] / close[i-12] - 1) * pip_mult
    if abs(mom_pips) < 10:
        return None, 0

    signal = None; conf = 0.5
    if mom_pips > 0 and mtf_bias[i] > 0:
        pullback = min(low[i], low[i-1], low[i-2]) < close[i-4] if i >= 4 else False
        if pullback and rsi[i] < 70 and order_flow[i] > -0.3:
            signal = 'buy'; conf += 0.2
            if session_quality > 0.8: conf += 0.1
            if trend_strength[i] > 0.2: conf += 0.1
            conf += mtf_bias[i] * 0.1
    elif mom_pips < 0 and mtf_bias[i] < 0:
        pullback = max(high[i], high[i-1], high[i-2]) > close[i-4] if i >= 4 else False
        if pullback and rsi[i] > 30 and order_flow[i] < 0.3:
            signal = 'sell'; conf += 0.2
            if session_quality > 0.8: conf += 0.1
            if trend_strength[i] < -0.2: conf += 0.1
            conf += abs(mtf_bias[i]) * 0.1
    return signal, conf


def _trend_following(i, close, macd, macd_sig, mom, rsi, mtf_bias, order_flow, trend_strength):
    if abs(trend_strength[i]) < 0.3 or abs(mtf_bias[i]) < 0.5 or i < 2:
        return None, 0

    signal = None; conf = 0.5
    cross_up = macd[i] > macd_sig[i] and macd[i-1] <= macd_sig[i-1]
    cross_dn = macd[i] < macd_sig[i] and macd[i-1] >= macd_sig[i-1]
    mom_up = mom[i] > 0 and rsi[i] > 45 if not np.isnan(mom[i]) else False
    mom_dn = mom[i] < 0 and rsi[i] < 55 if not np.isnan(mom[i]) else False

    if trend_strength[i] > 0.2 and mtf_bias[i] > 0 and (cross_up or mom_up):
        signal = 'buy'; conf += 0.3
        if order_flow[i] > 0: conf += 0.1
        conf += mtf_bias[i] * 0.1
    elif trend_strength[i] < -0.2 and mtf_bias[i] < 0 and (cross_dn or mom_dn):
        signal = 'sell'; conf += 0.3
        if order_flow[i] < 0: conf += 0.1
        conf += abs(mtf_bias[i]) * 0.1
    return signal, conf


def _volatility_breakout(i, close, atr, bb_upper, bb_lower, rsi, mtf_bias, order_flow, vol_ratio):
    if i < 20:
        return None, 0
    recent = np.mean(atr[i-4:i+1]); longer = np.mean(atr[i-19:i+1])
    if np.isnan(recent) or np.isnan(longer) or longer == 0 or recent <= longer * 1.15:
        return None, 0

    signal = None; conf = 0.5
    if close[i] > bb_upper[i] and not np.isnan(bb_upper[i]):
        signal = 'buy'; conf += 0.2
        if 60 < rsi[i] < 80: conf += 0.1
        if mtf_bias[i] > 0: conf += 0.1
    elif close[i] < bb_lower[i] and not np.isnan(bb_lower[i]):
        signal = 'sell'; conf += 0.2
        if 20 < rsi[i] < 40: conf += 0.1
        if mtf_bias[i] < 0: conf += 0.1

    if signal is None:
        return None, 0
    if vol_ratio[i] > 1.5: conf += 0.1
    if (signal == 'buy' and order_flow[i] > 0) or (signal == 'sell' and order_flow[i] < 0):
        conf += 0.1
    return signal, conf


# ======================================================================
# PHASE 1 — ENTRY GENERATION
# ======================================================================

def generate_entries(m5_df, mtf_bias, order_flow, trend_strength, hurst, pair,
                     mode='strict'):
    """
    Generate entries from M5 data.

    strict:  Full bot filters + 3 strategies.
    relaxed: MTF bias + trend direction agreement only.

    Returns arrays: (entry_ns, entry_prices, directions, windows, years, dows)
    Uses datetime index directly for year/dow/window extraction (not nanosecond math).
    """
    pip_mult = 100.0 if 'JPY' in pair else 10000.0
    conf_thresh = BOT_CONFIG['CONFIDENCE_THRESHOLD']
    hurst_thresh = BOT_CONFIG['HURST_THRESHOLD']

    close = m5_df['close'].values.astype(np.float64)
    high = m5_df['high'].values.astype(np.float64)
    low = m5_df['low'].values.astype(np.float64)
    rsi = m5_df['rsi'].values.astype(np.float64)
    macd_v = m5_df['macd'].values.astype(np.float64)
    macd_sig = m5_df['macd_signal'].values.astype(np.float64)
    mom = m5_df['momentum'].values.astype(np.float64)
    atr = m5_df['atr'].values.astype(np.float64)
    bb_upper = m5_df['bb_upper'].values.astype(np.float64)
    bb_lower = m5_df['bb_lower'].values.astype(np.float64)
    vol_ratio = (m5_df['volume_ratio'].values.astype(np.float64)
                 if 'volume_ratio' in m5_df.columns else np.ones(len(m5_df)))
    has_spread = 'spread' in m5_df.columns
    spread = m5_df['spread'].values.astype(np.float64) if has_spread else np.zeros(len(m5_df))

    has_bidask = 'bid_close' in m5_df.columns and 'ask_close' in m5_df.columns
    if has_bidask:
        bid_c = m5_df['bid_close'].values.astype(np.float64)
        ask_c = m5_df['ask_close'].values.astype(np.float64)

    m5_index = m5_df.index
    m5_ns = m5_index.asi8  # nanoseconds — for hit level lookup ONLY
    n = len(m5_df)
    is_relaxed = mode == 'relaxed'

    # Collect as lists, convert at end
    ens_list = []; price_list = []; dir_list = []
    win_list = []; year_list = []; dow_list = []

    for i in range(25, n):
        if is_relaxed:
            if np.isnan(mtf_bias[i]) or np.isnan(trend_strength[i]):
                continue
            if abs(mtf_bias[i]) < 0.5:
                continue
            if mtf_bias[i] > 0 and trend_strength[i] <= 0:
                continue
            if mtf_bias[i] < 0 and trend_strength[i] >= 0:
                continue
            ts = m5_index[i]
            if ts.weekday() >= 5:
                continue
            direction = 1 if mtf_bias[i] > 0 else -1
        else:
            if np.isnan(atr[i]):
                continue
            atr_pips = atr[i] * pip_mult
            if atr_pips < BOT_CONFIG['MIN_ATR_PIPS'] or atr_pips > BOT_CONFIG['MAX_ATR_PIPS']:
                continue

            ts = m5_index[i]
            session_name, session_quality = get_session_quality(ts)
            if session_quality <= 0.5:
                continue

            if has_spread:
                sv = spread[i]
                if not np.isnan(sv) and not np.isnan(atr[i]) and atr[i] > 0:
                    max_sr = 0.25 * (2 - session_quality)
                    abs_max = 0.03 if 'JPY' in pair else 0.0003
                    if sv > min(atr[i] * max_sr, abs_max):
                        continue

            if abs(mtf_bias[i]) < 0.5:
                continue
            if hurst[i] < hurst_thresh:
                continue

            best_signal = None; best_score = 0.0

            sig, conf = _momentum_pullback(i, close, high, low, rsi, mtf_bias,
                                            order_flow, trend_strength, session_quality, pip_mult)
            if sig and conf >= conf_thresh:
                score = min(conf, 0.95) * 2.17
                if score > best_score: best_signal = sig; best_score = score

            sig, conf = _trend_following(i, close, macd_v, macd_sig, mom, rsi,
                                          mtf_bias, order_flow, trend_strength)
            if sig and conf >= conf_thresh:
                score = min(conf, 0.95) * 2.17
                if score > best_score: best_signal = sig; best_score = score

            sig, conf = _volatility_breakout(i, close, atr, bb_upper, bb_lower, rsi,
                                              mtf_bias, order_flow, vol_ratio)
            if sig and conf >= conf_thresh:
                score = min(conf, 0.95) * 2.17
                if score > best_score: best_signal = sig; best_score = score

            if not best_signal:
                continue
            direction = 1 if best_signal == 'buy' else -1

        # ── Extract metadata from datetime directly (NOT from nanoseconds) ──
        ts = m5_index[i]
        year = ts.year
        dow = ts.dayofweek           # 0=Mon ... 4=Fri
        window = ts.hour * 2 + (1 if ts.minute >= 30 else 0)

        # Entry price: BUY at ask, SELL at bid
        if has_bidask:
            entry_price = float(ask_c[i]) if direction == 1 else float(bid_c[i])
        else:
            entry_price = float(close[i])

        ens_list.append(int(m5_ns[i]))
        price_list.append(entry_price)
        dir_list.append(direction)
        win_list.append(window)
        year_list.append(year)
        dow_list.append(dow)

    return (np.array(ens_list, dtype=np.int64),
            np.array(price_list, dtype=np.float64),
            np.array(dir_list, dtype=np.int64),
            np.array(win_list, dtype=np.int32),
            np.array(year_list, dtype=np.int32),
            np.array(dow_list, dtype=np.int32))


# ======================================================================
# PHASE 2 — HIT LEVEL COMPUTATION (bid/ask aware)
# ======================================================================

def _compute_hit_levels_numpy(bid_h, bid_l, bid_c, ask_h, ask_l, ask_c, t5s,
                               ens_arr, prices, dirs, pip_mult, levels_price, max_hold_ns):
    """
    For each entry, walk 5s bars and record first timestamp where MFE/MAE
    hits each of 39 pip levels.
      BUY:  exit on bid side (MFE from bid_high, MAE from bid_low)
      SELL: exit on ask side (MFE from ask_low, MAE from ask_high)
    """
    n_entries = len(ens_arr)
    n_levels = len(levels_price)
    mfe_ns = np.full((n_entries, n_levels), SENTINEL, dtype=np.int64)
    mae_ns = np.full((n_entries, n_levels), SENTINEL, dtype=np.int64)
    mh_pnl = np.zeros(n_entries, dtype=np.float64)
    mh_exit = np.zeros(n_entries, dtype=np.int64)

    for e in range(n_entries):
        ens = ens_arr[e]; ep = prices[e]; is_buy = dirs[e] == 1
        start = np.searchsorted(t5s, ens)
        end = min(np.searchsorted(t5s, ens + max_hold_ns, side='right'), len(bid_h))
        if start >= end - 1:
            continue

        sl = slice(start + 1, end)
        if is_buy:
            fav = np.maximum.accumulate(bid_h[sl] - ep)
            adv = np.maximum.accumulate(ep - bid_l[sl])
        else:
            fav = np.maximum.accumulate(ep - ask_l[sl])
            adv = np.maximum.accumulate(ask_h[sl] - ep)

        ts_sl = t5s[sl]
        for k in range(n_levels):
            hits = np.where(fav >= levels_price[k])[0]
            if len(hits) > 0: mfe_ns[e, k] = ts_sl[hits[0]]
            hits = np.where(adv >= levels_price[k])[0]
            if len(hits) > 0: mae_ns[e, k] = ts_sl[hits[0]]

        fi = end - 1
        mh_exit[e] = t5s[fi]
        mh_pnl[e] = (bid_c[fi] - ep) * pip_mult if is_buy else (ep - ask_c[fi]) * pip_mult

        if (e + 1) % 5000 == 0:
            log.info(f"  Hit levels: {e+1}/{n_entries}")

    return mfe_ns, mae_ns, mh_pnl, mh_exit


if HAS_NUMBA:
    @njit(cache=True)
    def _hit_single_numba(fav_h, fav_l, exit_c, t5s, start, end,
                          ep, is_buy, levels_price, pip_mult):
        n_lev = len(levels_price)
        mfe = np.full(n_lev, np.int64(9223372036854775807), dtype=np.int64)
        mae = np.full(n_lev, np.int64(9223372036854775807), dtype=np.int64)
        r_mfe = 0.0; r_mae = 0.0; mfe_next = 0; mae_next = 0

        for j in range(start + 1, end):
            f = fav_h[j] - ep if is_buy else ep - fav_l[j]
            a = ep - fav_l[j] if is_buy else fav_h[j] - ep
            if f > r_mfe: r_mfe = f
            if a > r_mae: r_mae = a
            while mfe_next < n_lev and r_mfe >= levels_price[mfe_next]:
                mfe[mfe_next] = t5s[j]; mfe_next += 1
            while mae_next < n_lev and r_mae >= levels_price[mae_next]:
                mae[mae_next] = t5s[j]; mae_next += 1
            if mfe_next >= n_lev and mae_next >= n_lev:
                break

        fi = min(end - 1, len(exit_c) - 1)
        mhp = (exit_c[fi] - ep) * pip_mult if is_buy else (ep - exit_c[fi]) * pip_mult
        return mfe, mae, mhp, t5s[fi]

    @njit(cache=True)
    def _compute_hit_levels_numba(bid_h, bid_l, bid_c, ask_h, ask_l, ask_c, t5s,
                                   ens_arr, prices, dirs, pip_mult, levels_price, max_hold_ns):
        n_ent = len(ens_arr); n_lev = len(levels_price)
        SENT = np.int64(9223372036854775807)
        all_mfe = np.full((n_ent, n_lev), SENT, dtype=np.int64)
        all_mae = np.full((n_ent, n_lev), SENT, dtype=np.int64)
        all_mhp = np.zeros(n_ent, dtype=np.float64)
        all_mhe = np.zeros(n_ent, dtype=np.int64)

        for e in range(n_ent):
            ens = ens_arr[e]; ep = prices[e]; is_buy = dirs[e] == 1
            start = np.searchsorted(t5s, ens)
            # searchsorted default is side='left'. For end boundary we need side='right'
            # (include bars AT the max-hold time). Numba doesn't support side param,
            # so we search left then scan forward past equal values.
            end_target = ens + max_hold_ns
            end = np.searchsorted(t5s, end_target)
            while end < len(t5s) and t5s[end] <= end_target:
                end += 1
            if end > len(bid_h): end = len(bid_h)
            if start >= end - 1: continue

            if is_buy:
                fh, fl, ec = bid_h, bid_l, bid_c
            else:
                fh, fl, ec = ask_h, ask_l, ask_c

            mfe, mae, mhp, mhe = _hit_single_numba(fh, fl, ec, t5s, start, end,
                                                     ep, is_buy, levels_price, pip_mult)
            all_mfe[e] = mfe; all_mae[e] = mae; all_mhp[e] = mhp; all_mhe[e] = mhe

        return all_mfe, all_mae, all_mhp, all_mhe


def compute_hit_levels(bid_h, bid_l, bid_c, ask_h, ask_l, ask_c, t5s,
                       ens_arr, prices, dirs, pip_mult, max_hold_ns):
    """Dispatch to Numba or numpy."""
    lp = LEVELS / pip_mult
    if HAS_NUMBA:
        return _compute_hit_levels_numba(bid_h, bid_l, bid_c, ask_h, ask_l, ask_c,
                                          t5s, ens_arr, prices, dirs, pip_mult, lp, max_hold_ns)
    return _compute_hit_levels_numpy(bid_h, bid_l, bid_c, ask_h, ask_l, ask_c,
                                      t5s, ens_arr, prices, dirs, pip_mult, lp, max_hold_ns)


# ======================================================================
# PER-PAIR PIPELINE
# ======================================================================

def process_pair(pair: str, filepath: str, mode: str = 'strict') -> Optional[Dict]:
    """Load → resample → indicators → signals → entries → hit levels."""
    t0 = time_mod.time()
    pip_mult = 100.0 if 'JPY' in pair else 10000.0
    max_hold_ns = np.int64(MAX_HOLD_HOURS * 3600 * 1e9)

    try:
        df_5s = load_pair_data(filepath)
        if len(df_5s) < 10000:
            log.warning(f"{pair}: only {len(df_5s)} 5s bars, skip")
            return None
        log.info(f"{pair}: {len(df_5s):,} 5s bars loaded")

        tfs = build_timeframes(df_5s)
        for tf in tfs:
            if len(tfs[tf]) >= 50:
                tfs[tf] = add_indicators(tfs[tf])

        m5 = tfs['M5']
        if len(m5) < 100:
            log.warning(f"{pair}: only {len(m5)} M5 bars, skip")
            return None

        # Signal components
        mtf_bias = compute_mtf_bias(tfs, m5.index)
        trend_strength = compute_trend_strength(m5)

        if mode == 'strict':
            log.info(f"{pair}: Computing MTF, order flow, Hurst, trend strength...")
            order_flow = compute_order_flow(tfs['M1'], m5.index)
            hurst = compute_rolling_hurst(tfs['H1'], m5.index)
        else:
            log.info(f"{pair}: RELAXED mode — MTF + trend only")
            order_flow = np.zeros(len(m5))
            hurst = np.ones(len(m5))

        # Generate entries
        ens_arr, prices, dirs, windows, years, dows = generate_entries(
            m5, mtf_bias, order_flow, trend_strength, hurst, pair, mode=mode)

        if len(ens_arr) == 0:
            log.warning(f"{pair}: 0 entries")
            return None
        log.info(f"{pair}: {len(ens_arr):,} entries, years={sorted(set(years))}")

        # Hit levels on 5s bid/ask data
        has_ba = 'bid_high' in df_5s.columns
        if has_ba:
            bh = df_5s['bid_high'].values.astype(np.float64)
            bl = df_5s['bid_low'].values.astype(np.float64)
            bc = df_5s['bid_close'].values.astype(np.float64)
            ah = df_5s['ask_high'].values.astype(np.float64)
            al = df_5s['ask_low'].values.astype(np.float64)
            ac = df_5s['ask_close'].values.astype(np.float64)
        else:
            log.warning(f"{pair}: NO bid/ask — falling back to mid")
            bh = df_5s['high'].values.astype(np.float64)
            bl = df_5s['low'].values.astype(np.float64)
            bc = df_5s['close'].values.astype(np.float64)
            ah, al, ac = bh, bl, bc

        t5s = df_5s.index.asi8
        if not isinstance(t5s, np.ndarray):
            t5s = np.asarray(t5s, dtype=np.int64)

        log.info(f"{pair}: Computing hit levels for {len(ens_arr):,} entries...")
        mfe_ns, mae_ns, mh_pnl, mh_exit = compute_hit_levels(
            bh, bl, bc, ah, al, ac, t5s, ens_arr, prices, dirs, pip_mult, max_hold_ns)

        # Filter out entries that had no valid hold window (near end of data).
        # These have mh_exit == 0 (default) and all-SENTINEL hit levels,
        # contributing phantom 0-pip trades that dilute slot averages.
        valid = mh_exit > 0
        n_invalid = (~valid).sum()
        if n_invalid > 0:
            log.info(f"{pair}: Dropping {n_invalid} entries with no hold window (near data end)")
            ens_arr = ens_arr[valid]; prices = prices[valid]; dirs = dirs[valid]
            windows = windows[valid]; years = years[valid]; dows = dows[valid]
            mfe_ns = mfe_ns[valid]; mae_ns = mae_ns[valid]
            mh_pnl = mh_pnl[valid]; mh_exit = mh_exit[valid]

        if len(ens_arr) == 0:
            log.warning(f"{pair}: 0 entries after filtering")
            return None

        elapsed = time_mod.time() - t0
        log.info(f"{pair}: DONE in {elapsed:.1f}s — {len(ens_arr):,} entries")

        return {
            'pair': pair, 'entry_ns': ens_arr, 'prices': prices, 'dirs': dirs,
            'windows': windows, 'years': years, 'dows': dows,
            'mfe_ns': mfe_ns, 'mae_ns': mae_ns, 'mh_pnl': mh_pnl, 'mh_exit_ns': mh_exit,
        }

    except Exception as e:
        log.error(f"{pair}: FAILED — {e}")
        import traceback; traceback.print_exc()
        return None


# ======================================================================
# PHASE 3 — GRID EVALUATION
# ======================================================================

def evaluate_grid(entries: Dict, years_list: List[int]) -> pd.DataFrame:
    """Evaluate all (year, dow, window, TP, SL) combos. No position blocking."""
    log.info("Phase 3: Evaluating grid...")
    all_years = entries['years']; all_dows = entries['dows']; all_windows = entries['windows']
    all_mfe = entries['mfe_ns']; all_mae = entries['mae_ns']; all_mh = entries['mh_pnl']
    rows = []

    for y in years_list:
        for d in range(5):
            for w in range(N_WINDOWS):
                mask = (all_years == y) & (all_dows == d) & (all_windows == w)
                n = mask.sum()
                if n < MIN_TRADES:
                    continue

                mfe_m = all_mfe[mask]; mae_m = all_mae[mask]; mh_m = all_mh[mask]

                for tp_i in range(N_LEVELS):
                    for sl_i in range(N_LEVELS):
                        tp_t = mfe_m[:, tp_i]; sl_t = mae_m[:, sl_i]
                        sl_hit = (sl_t < SENTINEL) & (sl_t <= tp_t)
                        tp_hit = (tp_t < SENTINEL) & (tp_t < sl_t)
                        pnl = np.where(sl_hit, -LEVELS[sl_i],
                               np.where(tp_hit, LEVELS[tp_i], mh_m))
                        total = pnl.sum()
                        avg = pnl.mean()
                        wins = (pnl > 0).sum()
                        gl = abs(pnl[pnl < 0].sum()) if (pnl < 0).any() else 1.0
                        pf = pnl[pnl > 0].sum() / gl if gl > 0 else 0.0

                        rows.append({
                            'year': y, 'dow': d, 'dow_name': DOW_NAMES[d],
                            'window': w, 'window_utc': f"{w//2:02d}:{(w%2)*30:02d}",
                            'tp': int(LEVELS[tp_i]), 'sl': int(LEVELS[sl_i]),
                            'n_trades': n, 'total_pips': round(total, 1),
                            'avg_pips': round(avg, 3), 'win_rate': round(wins / n * 100, 1),
                            'profit_factor': round(pf, 3),
                        })

        log.info(f"  Year {y}: done")

    df = pd.DataFrame(rows)
    log.info(f"Grid: {len(df):,} rows")
    return df


# ======================================================================
# PHASE 4a — WALK-FORWARD PER-SLOT (position-blocked)
# ======================================================================

def walk_forward_blocked(entries: Dict, years_list: List[int]) -> pd.DataFrame:
    """Per-slot walk-forward with optimal TP/SL from training, position blocking on test."""
    log.info("Phase 4a: Walk-forward per-slot (position-blocked)...")
    all_ns = entries['entry_ns']; all_w = entries['windows']; all_y = entries['years']
    all_d = entries['dows']; all_mfe = entries['mfe_ns']; all_mae = entries['mae_ns']
    all_mh = entries['mh_pnl']; all_mhe = entries['mh_exit_ns']; all_p = entries['pair_idxs']
    rows = []

    for ti in range(1, len(years_list)):
        test_yr = years_list[ti]
        train_yrs = years_list[:ti]
        train_m = np.isin(all_y, train_yrs)
        test_m = all_y == test_yr

        for d in range(5):
            for w in range(N_WINDOWS):
                tr = train_m & (all_d == d) & (all_w == w)
                if tr.sum() < MIN_TRADES: continue

                # Best TP/SL on training
                best_avg = -999.0; btp = 0; bsl = 0
                tmfe = all_mfe[tr]; tmae = all_mae[tr]; tmh = all_mh[tr]
                for tp_i in range(N_LEVELS):
                    tp_t = tmfe[:, tp_i]
                    for sl_i in range(N_LEVELS):
                        sl_t = tmae[:, sl_i]
                        sf = (sl_t < SENTINEL) & (sl_t <= tp_t)
                        tf = (tp_t < SENTINEL) & (tp_t < sl_t)
                        pnl = np.where(sf, -LEVELS[sl_i], np.where(tf, LEVELS[tp_i], tmh))
                        a = pnl.mean()
                        if a > best_avg: best_avg = a; btp = tp_i; bsl = sl_i

                if best_avg <= 0: continue

                # Test with position blocking
                te = test_m & (all_d == d) & (all_w == w)
                if te.sum() < 3: continue
                idx = np.where(te)[0]
                idx = idx[np.argsort(all_ns[idx])]

                last_exit = {}; pnls = []
                for e in idx:
                    pid = all_p[e]
                    if pid in last_exit and all_ns[e] < last_exit[pid]: continue
                    tp_t = all_mfe[e, btp]; sl_t = all_mae[e, bsl]
                    if sl_t < SENTINEL and sl_t <= tp_t:
                        p = -LEVELS[bsl]; ex = sl_t
                    elif tp_t < SENTINEL and tp_t < sl_t:
                        p = LEVELS[btp]; ex = tp_t
                    else:
                        p = all_mh[e]; ex = all_mhe[e]
                    pnls.append(p); last_exit[pid] = ex

                if len(pnls) < 3: continue
                pa = np.array(pnls)
                gl = abs(pa[pa < 0].sum()) if (pa < 0).any() else 1.0

                rows.append({
                    'train_years': str(train_yrs), 'test_year': test_yr,
                    'dow': d, 'dow_name': DOW_NAMES[d], 'window': w,
                    'window_utc': f"{w//2:02d}:{(w%2)*30:02d}",
                    'tp': int(LEVELS[btp]), 'sl': int(LEVELS[bsl]),
                    'train_n': int(tr.sum()), 'train_avg_pips': round(best_avg, 3),
                    'test_n': len(pa), 'test_total_pips': round(pa.sum(), 1),
                    'test_avg_pips': round(pa.mean(), 3),
                    'test_win_rate': round((pa > 0).mean() * 100, 1),
                    'test_pf': round(pa[pa > 0].sum() / gl, 3),
                })

    return pd.DataFrame(rows)


# ======================================================================
# PHASE 4b — WALK-FORWARD PORTFOLIO (per-slot TP/SL, combined)
# ======================================================================

def walk_forward_portfolio(entries: Dict, years_list: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-slot optimal TP/SL, walk-forward validated, combined portfolio.
    Returns (slot_detail_df, portfolio_df).
    """
    log.info("Phase 4b: Walk-forward portfolio (per-slot TP/SL)...")
    all_ns = entries['entry_ns']; all_w = entries['windows']; all_y = entries['years']
    all_d = entries['dows']; all_mfe = entries['mfe_ns']; all_mae = entries['mae_ns']
    all_mh = entries['mh_pnl']; all_mhe = entries['mh_exit_ns']; all_p = entries['pair_idxs']
    slot_rows = []; port_rows = []

    for ti in range(1, len(years_list)):
        test_yr = years_list[ti]
        train_yrs = years_list[:ti]
        train_m = np.isin(all_y, train_yrs)
        test_m = all_y == test_yr
        log.info(f"  Train: {train_yrs} → Test: {test_yr}")

        profitable_slots = []

        for d in range(5):
            for w in range(N_WINDOWS):
                tr = train_m & (all_d == d) & (all_w == w)
                if tr.sum() < MIN_TRADES: continue

                # Best TP/SL on training
                best_avg = -999.0; btp = 0; bsl = 0
                tmfe = all_mfe[tr]; tmae = all_mae[tr]; tmh = all_mh[tr]
                for tp_i in range(N_LEVELS):
                    tp_t = tmfe[:, tp_i]
                    for sl_i in range(N_LEVELS):
                        sl_t = tmae[:, sl_i]
                        sf = (sl_t < SENTINEL) & (sl_t <= tp_t)
                        tf = (tp_t < SENTINEL) & (tp_t < sl_t)
                        pnl = np.where(sf, -LEVELS[sl_i], np.where(tf, LEVELS[tp_i], tmh))
                        a = pnl.mean()
                        if a > best_avg: best_avg = a; btp = tp_i; bsl = sl_i

                if best_avg <= 0: continue

                # Test OOS (independent, no position blocking)
                te = test_m & (all_d == d) & (all_w == w)
                nte = te.sum()
                if nte < 3: continue

                temfe = all_mfe[te]; temae = all_mae[te]; temh = all_mh[te]
                tp_t = temfe[:, btp]; sl_t = temae[:, bsl]
                sf = (sl_t < SENTINEL) & (sl_t <= tp_t)
                tf = (tp_t < SENTINEL) & (tp_t < sl_t)
                tpnl = np.where(sf, -LEVELS[bsl], np.where(tf, LEVELS[btp], temh))
                tavg = tpnl.mean()
                gl = abs(tpnl[tpnl < 0].sum()) if (tpnl < 0).any() else 1.0

                slot_rows.append({
                    'train_years': str(train_yrs), 'test_year': test_yr,
                    'dow': d, 'dow_name': DOW_NAMES[d], 'window': w,
                    'window_utc': f"{w//2:02d}:{(w%2)*30:02d}",
                    'tp': int(LEVELS[btp]), 'sl': int(LEVELS[bsl]),
                    'train_n': int(tr.sum()), 'train_avg_pips': round(best_avg, 3),
                    'test_n': nte, 'test_total_pips': round(tpnl.sum(), 1),
                    'test_avg_pips': round(tavg, 3),
                    'test_win_rate': round((tpnl > 0).mean() * 100, 1),
                    'test_pf': round(tpnl[tpnl > 0].sum() / gl, 3),
                    'oos_profitable': tavg > 0,
                })

                if tavg > 0:
                    profitable_slots.append((d, w, btp, bsl))

        log.info(f"    {len(profitable_slots)} / 240 slots profitable OOS")

        if not profitable_slots:
            continue

        # Combined portfolio: position-blocked replay with per-slot TP/SL
        slot_config = {}
        all_idx = []
        for d, w, tp_i, sl_i in profitable_slots:
            te = test_m & (all_d == d) & (all_w == w)
            idxs = np.where(te)[0]
            for e in idxs:
                slot_config[e] = (tp_i, sl_i)
            all_idx.extend(idxs.tolist())

        if not all_idx: continue
        all_idx = np.array(all_idx, dtype=np.int64)
        all_idx = all_idx[np.argsort(all_ns[all_idx])]

        last_exit = {}; pnls = []
        for e in all_idx:
            pid = all_p[e]
            if pid in last_exit and all_ns[e] < last_exit[pid]: continue
            tp_i, sl_i = slot_config[e]
            tp_t = all_mfe[e, tp_i]; sl_t = all_mae[e, sl_i]
            if sl_t < SENTINEL and sl_t <= tp_t:
                p = -LEVELS[sl_i]; ex = sl_t
            elif tp_t < SENTINEL and tp_t < sl_t:
                p = LEVELS[tp_i]; ex = tp_t
            else:
                p = all_mh[e]; ex = all_mhe[e]
            pnls.append(p); last_exit[pid] = ex

        if len(pnls) < 5: continue
        pa = np.array(pnls)
        gl = abs(pa[pa < 0].sum()) if (pa < 0).any() else 1.0
        cum = np.cumsum(pa)
        dd = (np.maximum.accumulate(cum) - cum).max()

        slot_labels = sorted([f"{DOW_NAMES[d]} {w//2:02d}:{(w%2)*30:02d}"
                              for d, w, _, _ in profitable_slots])
        port_rows.append({
            'train_years': str(train_yrs), 'test_year': test_yr,
            'n_slots': len(profitable_slots),
            'slots': ', '.join(slot_labels[:20]) + ('...' if len(slot_labels) > 20 else ''),
            'test_n_signals': len(all_idx), 'test_n_traded': len(pa),
            'test_n_blocked': len(all_idx) - len(pa),
            'test_total_pips': round(pa.sum(), 1), 'test_avg_pips': round(pa.mean(), 3),
            'test_win_rate': round((pa > 0).mean() * 100, 1),
            'test_pf': round(pa[pa > 0].sum() / gl, 3),
            'test_max_dd': round(dd, 1),
        })

    return pd.DataFrame(slot_rows), pd.DataFrame(port_rows)


# ======================================================================
# OUTPUT
# ======================================================================

def save_results(grid_df, wf_slot_df, wf_oos_df, wf_port_df, output_dir, mode):
    ts = time_mod.strftime('%Y%m%d_%H%M%S')
    pre = f"{mode}_{ts}"
    os.makedirs(output_dir, exist_ok=True)

    for name, df in [('grid_results', grid_df), ('best_per_slot', None),
                     ('wf_perslot_blocked', wf_slot_df),
                     ('wf_perslot_oos', wf_oos_df),
                     ('wf_portfolio', wf_port_df)]:
        if name == 'best_per_slot':
            if len(grid_df) > 0:
                df = grid_df.loc[grid_df.groupby(['year', 'dow', 'window'])['avg_pips'].idxmax()]
            else:
                continue
        if df is None or len(df) == 0:
            continue
        path = os.path.join(output_dir, f'{name}_{pre}.csv')
        df.to_csv(path, index=False)
        log.info(f"Saved: {path} ({len(df):,} rows)")


def print_summary(grid_df, wf_slot_df, wf_oos_df, wf_port_df, years_list):
    print("\n" + "=" * 80)
    print("GRID SEARCH RESULTS SUMMARY")
    print("=" * 80)

    if len(grid_df) == 0:
        print("No results!"); return

    for y in years_list:
        ydf = grid_df[grid_df['year'] == y]
        if len(ydf) == 0: continue
        best = ydf.nlargest(5, 'avg_pips')
        print(f"\n── {y}: Top 5 ──")
        for _, r in best.iterrows():
            print(f"  {r['dow_name']} {r['window_utc']} | TP={r['tp']:3d} SL={r['sl']:3d} | "
                  f"{r['n_trades']:5d} trades | {r['total_pips']:+8.1f} total | "
                  f"{r['avg_pips']:+.3f} avg | {r['win_rate']:.1f}% WR | PF {r['profit_factor']:.3f}")

    if len(wf_port_df) > 0:
        print(f"\n── Walk-Forward PORTFOLIO (per-slot TP/SL, position-blocked) ──")
        for _, r in wf_port_df.iterrows():
            print(f"  Train {r['train_years']} → Test {r['test_year']}: "
                  f"{r['n_slots']} slots | {r['test_n_traded']}/{r['test_n_signals']} traded | "
                  f"{r['test_total_pips']:+.1f} pips | {r['test_avg_pips']:+.3f} avg | "
                  f"{r['test_win_rate']:.1f}% WR | PF {r['test_pf']:.3f} | DD {r['test_max_dd']:.0f}")

        total_oos = wf_port_df['test_total_pips'].sum()
        total_tr = wf_port_df['test_n_traded'].sum()
        print(f"\n  CUMULATIVE OOS: {total_oos:+,.1f} pips / {total_tr} trades "
              f"({total_oos/total_tr:+.3f} avg)" if total_tr > 0 else "")

    if len(wf_oos_df) > 0 and 'oos_profitable' in wf_oos_df.columns:
        prof = wf_oos_df[wf_oos_df['oos_profitable']]
        if len(prof) > 0:
            cons = prof.groupby(['dow_name', 'window_utc']).agg(
                yrs=('test_year', 'count'), avg=('test_avg_pips', 'mean'),
                total=('test_total_pips', 'sum')).sort_values('yrs', ascending=False)
            nf = wf_port_df['test_year'].nunique() if len(wf_port_df) > 0 else 1
            print(f"\n── Most Consistent OOS Slots ──")
            for (dn, wl), r in cons.head(15).iterrows():
                print(f"  {dn} {wl} | {int(r['yrs'])}/{nf} yrs | avg {r['avg']:+.3f} | total {r['total']:+.1f}")

    print("=" * 80)


# ======================================================================
# MAIN
# ======================================================================

def find_data_dir():
    for c in [r'C:\Forex_Projects\5_year_5s_data',
              r'C:\Forex_Projects\Forex_bot_original',
              r'C:\Forex_Projects',
              os.path.join(os.path.expanduser('~'), 'Desktop', 'Forex_Projects'),
              os.getcwd()]:
        if os.path.isdir(c):
            for pat in ['*_S5_*.parquet', '*_5s_*.parquet']:
                if glob.glob(os.path.join(c, pat)):
                    return c
    return os.getcwd()


def discover_pairs(data_dir):
    pairs = {}
    ALL_PAIRS = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'EUR_JPY', 'GBP_JPY',
                 'AUD_USD', 'NZD_USD', 'USD_CAD', 'USD_CHF', 'EUR_GBP',
                 'EUR_AUD', 'GBP_AUD', 'AUD_JPY', 'CAD_JPY', 'CHF_JPY',
                 'NZD_JPY', 'AUD_CAD', 'NZD_CAD', 'AUD_NZD']
    for pat in ['*_S5_*.parquet', '*_5s_*.parquet', '*_S5_*.pkl', '*_5s_*.pkl']:
        for f in glob.glob(os.path.join(data_dir, pat)):
            base = os.path.basename(f)
            for pair in ALL_PAIRS:
                if pair in base and pair not in pairs:
                    pairs[pair] = f
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Grid Search Backtester v2')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--mode', type=str, default='strict', choices=['strict', 'relaxed'])
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    data_dir = args.data_dir or find_data_dir()
    output_dir = args.output_dir or data_dir
    mode = args.mode
    t_start = time_mod.time()

    log.info(f"Data dir: {data_dir}")
    log.info(f"Output dir: {output_dir}")
    log.info(f"Mode: {mode.upper()}")
    log.info(f"Numba: {'YES' if HAS_NUMBA else 'NO'}")
    log.info(f"TA-Lib: {'YES' if HAS_TALIB else 'NO (numpy fallback)'}")
    log.info(f"Grid: {N_LEVELS}×{N_LEVELS} TP/SL × 5 days × {N_WINDOWS} windows = "
             f"{N_LEVELS*N_LEVELS*5*N_WINDOWS:,} combos/year")

    pairs = discover_pairs(data_dir)
    if not pairs:
        log.error(f"No parquet files found in {data_dir}")
        sys.exit(1)

    pair_names = sorted(pairs.keys())
    pair_to_idx = {p: i for i, p in enumerate(pair_names)}
    log.info(f"Found {len(pairs)} pairs: {', '.join(pair_names)}")

    # ── Phase 1+2: Process pairs ──
    all_entries = {k: [] for k in ['entry_ns', 'prices', 'dirs', 'windows',
                                    'years', 'dows', 'mfe_ns', 'mae_ns',
                                    'mh_pnl', 'mh_exit_ns', 'pair_idxs']}

    for pi, pname in enumerate(pair_names):
        log.info(f"\n{'='*60}")
        log.info(f"Pair {pi+1}/{len(pair_names)}: {pname}")
        result = process_pair(pname, pairs[pname], mode)
        if result:
            n = len(result['entry_ns'])
            for k in ['entry_ns', 'prices', 'dirs', 'windows', 'years', 'dows',
                       'mfe_ns', 'mae_ns', 'mh_pnl', 'mh_exit_ns']:
                all_entries[k].append(result[k])
            all_entries['pair_idxs'].append(np.full(n, pair_to_idx[pname], dtype=np.int32))

    # Concatenate
    if not all_entries['entry_ns']:
        log.error("No entries generated"); sys.exit(1)

    for k in all_entries:
        all_entries[k] = np.concatenate(all_entries[k])

    total = len(all_entries['entry_ns'])
    years_list = sorted(set(all_entries['years']))
    log.info(f"\nPhase 1+2 complete: {total:,} entries in {time_mod.time()-t_start:.1f}s")
    log.info(f"Year range: {years_list}")
    for pname in pair_names:
        pidx = pair_to_idx[pname]
        cnt = (all_entries['pair_idxs'] == pidx).sum()
        if cnt > 0:
            log.info(f"  {pname}: {cnt:,} entries")

    # ── Phase 3: Grid ──
    grid_df = evaluate_grid(all_entries, years_list)

    # ── Phase 4: Walk-forward ──
    wf_slot_df = pd.DataFrame()
    wf_oos_df = pd.DataFrame()
    wf_port_df = pd.DataFrame()

    if len(years_list) >= 2:
        wf_slot_df = walk_forward_blocked(all_entries, years_list)
        wf_oos_df, wf_port_df = walk_forward_portfolio(all_entries, years_list)
    else:
        log.warning("Only 1 year — skipping walk-forward. Need 2+ years.")

    # ── Output ──
    save_results(grid_df, wf_slot_df, wf_oos_df, wf_port_df, output_dir, mode)
    print_summary(grid_df, wf_slot_df, wf_oos_df, wf_port_df, years_list)

    log.info(f"\nTotal runtime: {time_mod.time()-t_start:.1f}s ({(time_mod.time()-t_start)/60:.1f} min)")


if __name__ == '__main__':
    main()
