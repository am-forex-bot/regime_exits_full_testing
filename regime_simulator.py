#!/usr/bin/env python3
"""
Regime v2 — Live Simulator
────────────────────────────
Replays regime transition trades with:
  • Margin-constrained position limits
  • Configurable slippage per side
  • Spread filter (skip entries where spread > max_spread_pips)
  • Concurrent position tracking per bar
  • Equity curve + drawdown
  • Per-pair and per-slot breakdown

Reads the regime_v2_slots CSV to get per-slot configs, then replays
against the raw 5s→M5 data.

Usage:
  python regime_simulator.py --data-dir "C:\\path" --slots-csv "regime_v2_slots_xxx.csv"
  python regime_simulator.py --data-dir "C:\\path" --slots-csv "path.csv" --max-positions 10 --slippage 0.3
"""

import argparse
import glob
import logging
import os
import sys
import time as time_mod
from typing import Dict, List, Tuple

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ======================================================================
# CONFIG
# ======================================================================

MTF_WEIGHTS = {'M1': 0.05, 'M5': 0.20, 'M15': 0.30, 'H1': 0.25, 'H4': 0.20}
N_WINDOWS = 48             # 30-min UTC windows
MAX_SPREAD_PIPS = 5.0      # skip entries with spread > this
DOW_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

ALL_PAIRS = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'EUR_JPY', 'GBP_JPY',
             'AUD_USD', 'NZD_USD', 'USD_CAD', 'USD_CHF', 'EUR_GBP',
             'EUR_AUD', 'GBP_AUD', 'AUD_JPY', 'CAD_JPY', 'CHF_JPY',
             'NZD_JPY', 'AUD_CAD', 'NZD_CAD', 'AUD_NZD']

# Default OANDA-style margin percentages (indicative)
MARGIN_PCT = {
    'EUR_USD': 0.02, 'GBP_USD': 0.02, 'USD_JPY': 0.02, 'EUR_JPY': 0.02,
    'GBP_JPY': 0.05, 'AUD_USD': 0.02, 'NZD_USD': 0.02, 'USD_CAD': 0.02,
    'USD_CHF': 0.02, 'EUR_GBP': 0.02, 'EUR_AUD': 0.05, 'GBP_AUD': 0.05,
    'AUD_JPY': 0.05, 'CAD_JPY': 0.05, 'CHF_JPY': 0.05, 'NZD_JPY': 0.05,
    'AUD_CAD': 0.05, 'NZD_CAD': 0.05, 'AUD_NZD': 0.05,
}

DEFAULT_UNIT_SIZE = 10000    # micro lot in units
DEFAULT_ACCOUNT = 10000.0    # starting balance
DEFAULT_MAX_POS = 15         # max concurrent positions
DEFAULT_SLIPPAGE = 0.3       # pips per side


# ======================================================================
# DATA LOADING (same as regime_backtest_v2 but flexible discovery)
# ======================================================================

def load_pair_data(filepath):
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
    return df


def build_timeframes(df_5s):
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'volume' in df_5s.columns:
        agg['volume'] = 'sum'
    has_ba = all(c in df_5s.columns for c in ['bid_high', 'bid_low', 'bid_close',
                                                'ask_high', 'ask_low', 'ask_close'])
    if has_ba:
        for c, f in {'bid_open': 'first', 'bid_high': 'max', 'bid_low': 'min', 'bid_close': 'last',
                      'ask_open': 'first', 'ask_high': 'max', 'ask_low': 'min', 'ask_close': 'last'}.items():
            if c in df_5s.columns:
                agg[c] = f
    tfs = {}
    for label, rule in [('M1','1min'),('M5','5min'),('M15','15min'),('H1','1h'),('H4','4h')]:
        df = df_5s.resample(rule).agg(agg).dropna(subset=['close'])
        if 'volume' not in df.columns:
            df['volume'] = 0.0
        tfs[label] = df
    return tfs


def _ema_np(arr, span):
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr); out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out

def add_ema(df):
    if len(df) < 22: return df
    c = df['close'].values.astype(np.float64)
    if HAS_TALIB:
        df['ema_9'] = talib.EMA(c, timeperiod=9)
        df['ema_21'] = talib.EMA(c, timeperiod=21)
    else:
        df['ema_9'] = _ema_np(c, 9); df['ema_21'] = _ema_np(c, 21)
    return df

def compute_mtf_bias(tfs, m5_index):
    total_w = sum(MTF_WEIGHTS.values())
    n = len(m5_index); bias = np.zeros(n, dtype=np.float64)
    tf_data = {}
    for tf_name in MTF_WEIGHTS:
        df = tfs.get(tf_name)
        if df is None or len(df) < 22 or 'ema_9' not in df.columns: continue
        e9 = df['ema_9'].values; e21 = df['ema_21'].values; cl = df['close'].values
        sig = np.where((e9>e21)&(cl>e9), 1.0, np.where((e9<e21)&(cl<e9), -1.0, 0.0))
        tf_data[tf_name] = (df.index.asi8, sig)
    m5_ns = m5_index.asi8
    for tf_name, w in MTF_WEIGHTS.items():
        if tf_name not in tf_data: continue
        tt, ts = tf_data[tf_name]
        idx = np.clip(np.searchsorted(tt, m5_ns, side='right') - 1, 0, len(ts) - 1)
        bias += ts[idx] * w
    if total_w > 0 and total_w != 1.0: bias /= total_w
    return np.clip(bias, -1.0, 1.0)


# ======================================================================
# REGIME STATE MACHINE
# ======================================================================

def _regime_py(bias, et, xt):
    n = len(bias); state = np.zeros(n, dtype=np.int32); cur = 0
    for i in range(n):
        b = bias[i]
        if cur == 0:
            if b > et: cur = 1
            elif b < -et: cur = -1
        elif cur == 1:
            if b < xt:
                cur = 0
                if b < -et: cur = -1
        else:
            if b > -xt:
                cur = 0
                if b > et: cur = 1
        state[i] = cur
    return state

if HAS_NUMBA:
    @njit(cache=True)
    def regime_hysteresis(bias, et, xt):
        n = len(bias); state = np.zeros(n, dtype=np.int32); cur = np.int32(0)
        for i in range(n):
            b = bias[i]
            if cur == 0:
                if b > et: cur = np.int32(1)
                elif b < -et: cur = np.int32(-1)
            elif cur == 1:
                if b < xt:
                    cur = np.int32(0)
                    if b < -et: cur = np.int32(-1)
            else:
                if b > -xt:
                    cur = np.int32(0)
                    if b > et: cur = np.int32(1)
            state[i] = cur
        return state
else:
    regime_hysteresis = _regime_py


# ======================================================================
# FLEXIBLE PAIR DISCOVERY
# ======================================================================

def discover_pairs(data_dir):
    """Find parquet files matching any of the known pair names."""
    pairs = {}
    # Try all parquet files in the directory
    for f in glob.glob(os.path.join(data_dir, '*.parquet')):
        base = os.path.basename(f).upper()
        for pair in ALL_PAIRS:
            # Match pair name with or without underscores
            if pair in base or pair.replace('_', '') in base:
                if pair not in pairs:
                    pairs[pair] = f
    # Also try .pkl
    for f in glob.glob(os.path.join(data_dir, '*.pkl')):
        base = os.path.basename(f).upper()
        for pair in ALL_PAIRS:
            if pair in base or pair.replace('_', '') in base:
                if pair not in pairs:
                    pairs[pair] = f
    return pairs


# ======================================================================
# TRADE GENERATION PER PAIR (full event lifecycle)
# ======================================================================

def generate_pair_trades(pair: str, m5: pd.DataFrame, state: np.ndarray,
                         slot_configs: dict, test_years: list,
                         slippage_pips: float,
                         max_spread_pips: float = MAX_SPREAD_PIPS) -> List[dict]:
    """
    Generate trades for one pair using per-slot configs from walk-forward.

    slot_configs: {(test_year, dow, window): {ec, xc, te_bars, train_years}}
    Skips entries where spread > max_spread_pips.

    Returns list of trade dicts with entry/exit times, PnL, etc.
    """
    pip_mult = 100.0 if 'JPY' in pair else 10000.0
    n = len(m5)
    m5_ns = m5.index.asi8

    has_ba = 'bid_close' in m5.columns
    bid_c = m5['bid_close'].values.astype(np.float64) if has_ba else m5['close'].values.astype(np.float64)
    ask_c = m5['ask_close'].values.astype(np.float64) if has_ba else m5['close'].values.astype(np.float64)

    trades = []
    spread_rejected = 0

    # Find ON transitions
    for i in range(1, n):
        if state[i-1] == 0 and state[i] != 0:
            ts = m5.index[i]
            dow = ts.dayofweek
            if dow >= 5:
                continue
            # ── Spread filter ──
            spread = (ask_c[i] - bid_c[i]) * pip_mult
            if spread > max_spread_pips:
                spread_rejected += 1
                continue
            year = ts.year
            window = ts.hour * 2 + (1 if ts.minute >= 30 else 0)

            # Check if we have a config for this slot in this test year
            key = (year, dow, window)
            if key not in slot_configs:
                continue

            cfg = slot_configs[key]
            ec = cfg['ec']       # entry confirm bars
            xc = cfg['xc']       # exit confirm bars
            te = cfg['te_bars']  # timed exit bars

            d = int(state[i])

            # ── Entry confirmation ──
            entry_bar = i + ec
            if entry_bar >= n:
                continue

            # Check regime still on at entry_bar
            still_on = True
            for b in range(i, entry_bar + 1):
                if b >= n or state[b] != d:
                    still_on = False
                    break
            if not still_on:
                continue

            # ── Spread filter at entry bar ──
            entry_spread = (ask_c[entry_bar] - bid_c[entry_bar]) * pip_mult
            if entry_spread > max_spread_pips:
                spread_rejected += 1
                continue

            # Entry price (with slippage)
            if d == 1:
                entry_price = float(ask_c[entry_bar]) + slippage_pips / pip_mult
            else:
                entry_price = float(bid_c[entry_bar]) - slippage_pips / pip_mult

            # ── Find regime exit with confirmation ──
            regime_exit_bar = None
            off_run = 0
            max_bar = min(entry_bar + te, n - 1)  # timed exit cap

            for b in range(entry_bar + 1, max_bar + 1):
                if state[b] != d:
                    off_run += 1
                else:
                    off_run = 0

                if off_run > xc:
                    regime_exit_bar = b
                    break

            # ── Determine actual exit bar ──
            timed_exit_bar = min(entry_bar + te, n - 1)

            if regime_exit_bar is not None:
                exit_bar = min(regime_exit_bar, timed_exit_bar)
            else:
                exit_bar = timed_exit_bar

            if exit_bar <= entry_bar:
                exit_bar = min(entry_bar + 1, n - 1)

            # Exit price (with slippage)
            if d == 1:
                exit_price = float(bid_c[exit_bar]) - slippage_pips / pip_mult
            else:
                exit_price = float(ask_c[exit_bar]) + slippage_pips / pip_mult

            pnl_pips = d * (exit_price - entry_price) * pip_mult

            trades.append({
                'pair': pair,
                'direction': d,
                'entry_bar': entry_bar,
                'exit_bar': exit_bar,
                'entry_ns': int(m5_ns[entry_bar]),
                'exit_ns': int(m5_ns[exit_bar]),
                'entry_time': m5.index[entry_bar],
                'exit_time': m5.index[exit_bar],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pips': round(pnl_pips, 2),
                'hold_bars': exit_bar - entry_bar,
                'year': year,
                'dow': dow,
                'dow_name': DOW_NAMES[dow],
                'window': window,
                'window_utc': f"{window//2:02d}:{(window%2)*30:02d}",
                'exit_type': 'regime' if (regime_exit_bar is not None and
                                           regime_exit_bar <= timed_exit_bar) else 'timed',
            })

    return trades


# ======================================================================
# MARGIN-CONSTRAINED PORTFOLIO SIMULATION
# ======================================================================

def simulate_portfolio(all_trades: List[dict], max_positions: int,
                       account_balance: float, unit_size: int) -> dict:
    """
    Replay all trades chronologically with:
    - Max concurrent position limit
    - Per-pair single position constraint (no stacking)
    - Margin tracking
    - Equity curve construction

    Returns comprehensive results dict.
    """
    if not all_trades:
        return None

    # Sort by entry time
    all_trades.sort(key=lambda t: t['entry_ns'])

    # ── Simulation ──
    open_positions = {}   # pair → trade dict (one per pair max)
    equity = account_balance
    peak_equity = equity
    max_dd_pips = 0.0
    max_dd_pct = 0.0

    executed = []
    rejected_margin = []
    rejected_pair_busy = []

    # Track concurrent positions over time
    concurrent_history = []
    pnl_by_year = {}
    trades_by_year = {}

    for trade in all_trades:
        pair = trade['pair']
        entry_ns = trade['entry_ns']
        yr = trade['year']

        # ── Close any expired positions first ──
        to_close = []
        for p, pos in open_positions.items():
            if pos['exit_ns'] <= entry_ns:
                to_close.append(p)
        for p in to_close:
            pos = open_positions.pop(p)
            # PnL already computed — just track it
            # (Actual PnL application happens when trade was opened)

        n_open = len(open_positions)

        # ── Check if pair already has an open position ──
        if pair in open_positions:
            if open_positions[pair]['exit_ns'] > entry_ns:
                rejected_pair_busy.append(trade)
                continue

        # ── Check margin/position limit ──
        if n_open >= max_positions:
            rejected_margin.append(trade)
            continue

        # ── Execute ──
        open_positions[pair] = trade
        executed.append(trade)

        concurrent_history.append((entry_ns, n_open + 1))

        if yr not in pnl_by_year:
            pnl_by_year[yr] = 0.0
            trades_by_year[yr] = 0
        pnl_by_year[yr] += trade['pnl_pips']
        trades_by_year[yr] += 1

    # ── Build equity curve from executed trades (by exit time) ──
    exec_sorted = sorted(executed, key=lambda t: t['exit_ns'])
    equity_curve = []
    cum_pnl = 0.0
    peak_pnl = 0.0
    max_dd = 0.0

    for t in exec_sorted:
        cum_pnl += t['pnl_pips']
        if cum_pnl > peak_pnl:
            peak_pnl = cum_pnl
        dd = peak_pnl - cum_pnl
        if dd > max_dd:
            max_dd = dd
        equity_curve.append({
            'exit_time': t['exit_time'],
            'exit_ns': t['exit_ns'],
            'pnl_pips': t['pnl_pips'],
            'cum_pnl': round(cum_pnl, 2),
            'drawdown': round(dd, 2),
            'pair': t['pair'],
            'year': t['year'],
        })

    # ── Concurrent position stats ──
    if concurrent_history:
        conc_vals = [c[1] for c in concurrent_history]
        peak_concurrent = max(conc_vals)
        avg_concurrent = np.mean(conc_vals)
    else:
        peak_concurrent = 0
        avg_concurrent = 0

    # ── Per-pair stats ──
    pair_stats = {}
    for t in executed:
        p = t['pair']
        if p not in pair_stats:
            pair_stats[p] = {'n': 0, 'pips': 0.0, 'wins': 0}
        pair_stats[p]['n'] += 1
        pair_stats[p]['pips'] += t['pnl_pips']
        if t['pnl_pips'] > 0:
            pair_stats[p]['wins'] += 1

    # ── Exit type breakdown ──
    regime_exits = sum(1 for t in executed if t['exit_type'] == 'regime')
    timed_exits = sum(1 for t in executed if t['exit_type'] == 'timed')
    regime_pnl = sum(t['pnl_pips'] for t in executed if t['exit_type'] == 'regime')
    timed_pnl = sum(t['pnl_pips'] for t in executed if t['exit_type'] == 'timed')

    # ── Hold time stats ──
    hold_bars = [t['hold_bars'] for t in executed]

    return {
        'executed': executed,
        'rejected_margin': rejected_margin,
        'rejected_pair_busy': rejected_pair_busy,
        'equity_curve': equity_curve,
        'pnl_by_year': pnl_by_year,
        'trades_by_year': trades_by_year,
        'pair_stats': pair_stats,
        'peak_concurrent': peak_concurrent,
        'avg_concurrent': avg_concurrent,
        'max_dd': max_dd,
        'total_pnl': cum_pnl,
        'regime_exits': regime_exits,
        'timed_exits': timed_exits,
        'regime_pnl': regime_pnl,
        'timed_pnl': timed_pnl,
        'hold_bars': hold_bars,
    }


# ======================================================================
# OUTPUT
# ======================================================================

def save_results(results, output_dir, max_positions, slippage):
    ts = time_mod.strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)
    tag = f"maxpos{max_positions}_slip{slippage}"

    # Equity curve
    eq_df = pd.DataFrame(results['equity_curve'])
    eq_path = os.path.join(output_dir, f'regime_sim_equity_{tag}_{ts}.csv')
    eq_df.to_csv(eq_path, index=False)
    log.info(f"Saved: {eq_path} ({len(eq_df):,} rows)")

    # All executed trades
    exec_df = pd.DataFrame(results['executed'])
    cols_out = ['pair','direction','entry_time','exit_time','entry_price','exit_price',
                'pnl_pips','hold_bars','year','dow_name','window_utc','exit_type']
    cols_out = [c for c in cols_out if c in exec_df.columns]
    exec_path = os.path.join(output_dir, f'regime_sim_trades_{tag}_{ts}.csv')
    exec_df[cols_out].to_csv(exec_path, index=False)
    log.info(f"Saved: {exec_path} ({len(exec_df):,} rows)")

    # Per-year summary
    yearly = []
    for yr in sorted(results['pnl_by_year'].keys()):
        yr_trades = [t for t in results['executed'] if t['year'] == yr]
        pa = np.array([t['pnl_pips'] for t in yr_trades])
        gl = abs(pa[pa<0].sum()) if (pa<0).any() else 1.0
        cum = np.cumsum(pa)
        dd = float((np.maximum.accumulate(cum) - cum).max()) if len(cum) > 0 else 0
        yearly.append({
            'year': yr,
            'trades': len(pa),
            'total_pips': round(pa.sum(), 1),
            'avg_pips': round(pa.mean(), 3),
            'win_rate': round((pa>0).mean()*100, 1),
            'profit_factor': round(pa[pa>0].sum()/gl, 3) if gl > 0 else 0,
            'max_dd': round(dd, 1),
        })
    yr_df = pd.DataFrame(yearly)
    yr_path = os.path.join(output_dir, f'regime_sim_yearly_{tag}_{ts}.csv')
    yr_df.to_csv(yr_path, index=False)
    log.info(f"Saved: {yr_path}")

    return eq_path, exec_path, yr_path


def print_results(results, max_positions, slippage):
    print("\n" + "=" * 100)
    print(f"REGIME v2 — LIVE SIMULATION RESULTS")
    print(f"Max positions: {max_positions} | Slippage: {slippage} pips/side")
    print("=" * 100)

    ex = results['executed']
    rej_m = results['rejected_margin']
    rej_p = results['rejected_pair_busy']
    total_signals = len(ex) + len(rej_m) + len(rej_p)

    print(f"\n── EXECUTION ──")
    print(f"  Total signals:       {total_signals:,}")
    print(f"  Executed:            {len(ex):,} ({len(ex)/total_signals*100:.1f}%)")
    print(f"  Rejected (margin):   {len(rej_m):,} ({len(rej_m)/total_signals*100:.1f}%)")
    print(f"  Rejected (pair busy):{len(rej_p):,} ({len(rej_p)/total_signals*100:.1f}%)")

    print(f"\n── PERFORMANCE ──")
    pa = np.array([t['pnl_pips'] for t in ex])
    gl = abs(pa[pa<0].sum()) if (pa<0).any() else 1.0
    print(f"  Total P&L:           {results['total_pnl']:+,.1f} pips")
    print(f"  Avg per trade:       {pa.mean():+.3f} pips")
    print(f"  Win rate:            {(pa>0).mean()*100:.1f}%")
    print(f"  Profit factor:       {pa[pa>0].sum()/gl:.3f}")
    print(f"  Max drawdown:        {results['max_dd']:,.1f} pips")
    print(f"  Avg win:             {pa[pa>0].mean():+.1f} pips" if (pa>0).any() else "")
    print(f"  Avg loss:            {pa[pa<0].mean():+.1f} pips" if (pa<0).any() else "")

    print(f"\n── CONCURRENT POSITIONS ──")
    print(f"  Peak concurrent:     {results['peak_concurrent']}")
    print(f"  Avg concurrent:      {results['avg_concurrent']:.1f}")

    print(f"\n── EXIT TYPE BREAKDOWN ──")
    re = results['regime_exits']; te = results['timed_exits']
    total_ex = re + te
    print(f"  Regime exits:        {re:,} ({re/total_ex*100:.1f}%) | {results['regime_pnl']:+,.1f} pips "
          f"({results['regime_pnl']/re:+.3f} avg)" if re > 0 else "")
    print(f"  Timed exits:         {te:,} ({te/total_ex*100:.1f}%) | {results['timed_pnl']:+,.1f} pips "
          f"({results['timed_pnl']/te:+.3f} avg)" if te > 0 else "")

    print(f"\n── HOLD TIME ──")
    hb = np.array(results['hold_bars'])
    print(f"  Mean:                {hb.mean():.0f} bars ({hb.mean()*5/60:.1f}h)")
    print(f"  Median:              {np.median(hb):.0f} bars ({np.median(hb)*5/60:.1f}h)")
    print(f"  P25:                 {np.percentile(hb,25):.0f} bars ({np.percentile(hb,25)*5/60:.1f}h)")
    print(f"  P75:                 {np.percentile(hb,75):.0f} bars ({np.percentile(hb,75)*5/60:.1f}h)")
    print(f"  Max:                 {hb.max():.0f} bars ({hb.max()*5/60:.1f}h)")

    print(f"\n── PER YEAR ──")
    losing = 0
    for yr in sorted(results['pnl_by_year'].keys()):
        pips = results['pnl_by_year'][yr]
        n = results['trades_by_year'][yr]
        avg = pips / n if n > 0 else 0
        if pips < 0: losing += 1
        yr_trades = [t['pnl_pips'] for t in ex if t['year'] == yr]
        pa_yr = np.array(yr_trades)
        wr = (pa_yr>0).mean()*100 if len(pa_yr) > 0 else 0
        gl_yr = abs(pa_yr[pa_yr<0].sum()) if (pa_yr<0).any() else 1.0
        pf = pa_yr[pa_yr>0].sum()/gl_yr if gl_yr > 0 else 0
        cum_yr = np.cumsum(pa_yr)
        dd_yr = float((np.maximum.accumulate(cum_yr) - cum_yr).max()) if len(cum_yr) > 0 else 0
        print(f"  {yr}: {n:>6,} trades | {pips:>+10,.1f} pips | {avg:>+7.3f} avg | "
              f"{wr:.1f}% WR | PF {pf:.3f} | DD {dd_yr:,.0f}")
    print(f"  Losing years: {losing}")

    print(f"\n── PER PAIR ──")
    ps = results['pair_stats']
    for p in sorted(ps.keys(), key=lambda x: ps[x]['pips'], reverse=True):
        s = ps[p]
        wr = s['wins']/s['n']*100 if s['n'] > 0 else 0
        avg = s['pips']/s['n'] if s['n'] > 0 else 0
        print(f"  {p:>8}: {s['n']:>5,} trades | {s['pips']:>+10,.1f} pips | {avg:>+7.3f} avg | {wr:.1f}% WR")

    # ── Sensitivity: what if fewer positions? ──
    print(f"\n── POSITION LIMIT SENSITIVITY ──")
    for mp in [5, 8, 10, 12, 15, 20]:
        # Quick re-sim
        _sim = _quick_sim(ex + rej_m + rej_p, mp)
        if _sim:
            print(f"  Max {mp:>2} pos: {_sim['n']:>6,} trades | {_sim['pips']:>+10,.1f} pips | "
                  f"{_sim['avg']:>+7.3f} avg | {_sim['wr']:.1f}% WR | DD {_sim['dd']:,.0f}")

    print("\n" + "=" * 100)


def _quick_sim(all_trades_list, max_pos):
    """Quick position-limited replay for sensitivity analysis."""
    trades = sorted(all_trades_list, key=lambda t: t['entry_ns'])
    open_pos = {}
    pnls = []

    for t in trades:
        pair = t['pair']
        entry_ns = t['entry_ns']

        # Close expired
        to_close = [p for p, pos in open_pos.items() if pos['exit_ns'] <= entry_ns]
        for p in to_close:
            del open_pos[p]

        if pair in open_pos and open_pos[pair]['exit_ns'] > entry_ns:
            continue
        if len(open_pos) >= max_pos:
            continue

        open_pos[pair] = t
        pnls.append(t['pnl_pips'])

    if not pnls:
        return None
    pa = np.array(pnls)
    gl = abs(pa[pa<0].sum()) if (pa<0).any() else 1.0
    cum = np.cumsum(pa)
    dd = float((np.maximum.accumulate(cum) - cum).max())
    return {
        'n': len(pa), 'pips': round(pa.sum(), 1),
        'avg': round(pa.mean(), 3), 'wr': round((pa>0).mean()*100, 1),
        'dd': round(dd, 1),
    }


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Regime v2 Live Simulator')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--slots-csv', type=str, required=True,
                        help='Path to regime_v2_slots CSV from backtester')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--max-positions', type=int, default=DEFAULT_MAX_POS)
    parser.add_argument('--slippage', type=float, default=DEFAULT_SLIPPAGE,
                        help='Slippage in pips per side')
    parser.add_argument('--account', type=float, default=DEFAULT_ACCOUNT)
    parser.add_argument('--units', type=int, default=DEFAULT_UNIT_SIZE)
    parser.add_argument('--test-years', type=str, default=None,
                        help='Comma-separated test years to simulate (default: all from slots CSV)')
    parser.add_argument('--max-spread', type=float, default=MAX_SPREAD_PIPS,
                        help='Max spread in pips to allow entry (default: 5.0)')
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir or data_dir
    max_pos = args.max_positions
    slippage = args.slippage
    max_spread = args.max_spread

    t_start = time_mod.time()

    # ── Load slot configs ──
    log.info(f"Loading slot configs: {args.slots_csv}")
    slots_df = pd.read_csv(args.slots_csv)

    # Only use OOS profitable slots
    prof_slots = slots_df[slots_df['oos_profitable'] == True].copy()
    log.info(f"  {len(prof_slots)} profitable slots from {len(slots_df)} total")

    # Get thresholds from the slots
    et = prof_slots['entry_thresh'].iloc[0]
    xt = prof_slots['exit_thresh'].iloc[0]
    log.info(f"  Thresholds: ET={et:.1f}, XT={xt:.1f}")

    # Build slot config lookup
    test_years = sorted(prof_slots['test_year'].unique())
    if args.test_years:
        test_years = [int(y) for y in args.test_years.split(',')]
    log.info(f"  Test years: {test_years}")

    slot_configs = {}
    for _, r in prof_slots.iterrows():
        yr = int(r['test_year'])
        if yr not in test_years:
            continue
        d = int(r['dow'])
        w = int(r['window'])
        ec_min = int(r['entry_confirm_min'])
        xc_min = int(r['exit_confirm_min'])
        te_bars = int(r['timed_exit_bars'])
        slot_configs[(yr, d, w)] = {
            'ec': ec_min // 5,       # convert mins to M5 bars
            'xc': xc_min // 5,
            'te_bars': te_bars,
        }
    log.info(f"  {len(slot_configs)} slot configs loaded")

    log.info(f"\nSimulation config:")
    log.info(f"  Max positions: {max_pos}")
    log.info(f"  Slippage:      {slippage} pips/side")
    log.info(f"  Max spread:    {max_spread} pips")
    log.info(f"  Windows:       {N_WINDOWS} (30-min UTC slots)")
    log.info(f"  Numba:         {'YES' if HAS_NUMBA else 'NO'}")

    # ── Discover and load pairs ──
    pairs = discover_pairs(data_dir)
    if not pairs:
        log.error(f"No parquet files found in {data_dir}")
        sys.exit(1)
    pair_names = sorted(pairs.keys())
    log.info(f"  Pairs: {len(pairs)} — {', '.join(pair_names)}")

    # ── Process each pair ──
    all_trades = []
    for pi, pname in enumerate(pair_names):
        t0 = time_mod.time()
        df_5s = load_pair_data(pairs[pname])
        tfs = build_timeframes(df_5s)
        for tf in tfs:
            if len(tfs[tf]) >= 22:
                tfs[tf] = add_ema(tfs[tf])
        m5 = tfs['M5']
        if len(m5) < 100:
            log.warning(f"  {pname}: skip ({len(m5)} M5 bars)")
            continue

        bias = compute_mtf_bias(tfs, m5.index)
        state = regime_hysteresis(bias, et, xt)

        trades = generate_pair_trades(pname, m5, state, slot_configs,
                                       test_years, slippage,
                                       max_spread_pips=max_spread)
        all_trades.extend(trades)

        elapsed = time_mod.time() - t0
        log.info(f"  {pname}: {len(trades):,} trades ({elapsed:.1f}s)")

    del df_5s
    log.info(f"\nTotal trades generated: {len(all_trades):,}")

    # ── Run simulation ──
    log.info(f"\nRunning portfolio simulation...")
    results = simulate_portfolio(all_trades, max_pos, args.account, args.units)

    if results is None:
        log.error("No trades executed!")
        sys.exit(1)

    # ── Output ──
    save_results(results, output_dir, max_pos, slippage)
    print_results(results, max_pos, slippage)

    elapsed = time_mod.time() - t_start
    log.info(f"\nTotal runtime: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == '__main__':
    main()
