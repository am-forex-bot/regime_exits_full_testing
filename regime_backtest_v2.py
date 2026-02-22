#!/usr/bin/env python3
"""
Regime Transition Backtester v2
────────────────────────────────
Grid dimensions:
  • Entry threshold   (MTF bias level to trigger entry)
  • Exit threshold    (MTF bias level to trigger exit — hysteresis)
  • Entry confirmation bars  (regime must stay ON for N M5 bars before entry)
  • Exit confirmation bars   (regime must stay OFF for N M5 bars before exit)
  • Timed exit        (hard exit after N M5 bars regardless of regime)

No SL. No TP. Entry on regime transition, exit on counter-transition or timer.
Walk-forward validated by (dow × 30-min UTC window) slots.
Bid/ask pricing throughout.

Statistical robustness:
  • Reduced parameter space (3 EC × 3 XC × ~28 TE = 252 combos/slot)
  • Bonferroni-corrected significance test on OOS results
  • Min 50 training trades, 20 test trades per slot
  • Spread filter: skip entries where spread > max_spread_pips

Usage:
  python regime_backtest_v2.py --data-dir "C:\\path\\to\\parquets"
  python regime_backtest_v2.py --data-dir "C:\\path" --fine-grid
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
from scipy import stats as scipy_stats

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

DEFAULT_ENTRY_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
DEFAULT_EXIT_THRESHOLDS  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

ENTRY_CONFIRM_VALUES = [0, 2, 6]     # 0, 10, 30 min (reduced from 0-12)
EXIT_CONFIRM_VALUES  = [0, 2, 6]     # 0, 10, 30 min (reduced from 0-12)
MAX_HOLD_BARS = 576        # 48h of M5 bars
MIN_TRAIN_TRADES = 50      # min trades in training (was 15)
MIN_TEST_TRADES  = 20      # min trades in test (was 3)
N_WINDOWS = 48             # 30-min UTC windows
MAX_SPREAD_PIPS = 5.0      # skip entries with spread > this
DOW_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

ALL_PAIRS = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'EUR_JPY', 'GBP_JPY',
             'AUD_USD', 'NZD_USD', 'USD_CAD', 'USD_CHF', 'EUR_GBP',
             'EUR_AUD', 'GBP_AUD', 'AUD_JPY', 'CAD_JPY', 'CHF_JPY',
             'NZD_JPY', 'AUD_CAD', 'NZD_CAD', 'AUD_NZD']


def make_timed_exit_grid(fine=False):
    if fine:
        return np.arange(1, MAX_HOLD_BARS + 1, dtype=np.int32)
    # Reduced grid: ~30 values instead of ~120
    bars = set()
    for b in range(6, 49, 6):                    bars.add(b)   # 30m-4h every 30m (8 vals)
    for b in range(60, 145, 12):                  bars.add(b)   # 5h-12h every 1h  (8 vals)
    for b in range(168, 289, 24):                 bars.add(b)   # 14h-24h every 2h (6 vals)
    for b in range(336, MAX_HOLD_BARS + 1, 48):   bars.add(b)   # 28h-48h every 4h (6 vals)
    bars.add(MAX_HOLD_BARS)
    return np.array(sorted(bars), dtype=np.int32)


# ======================================================================
# DATA LOADING
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


# ======================================================================
# INDICATORS
# ======================================================================

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


# ======================================================================
# MTF BIAS
# ======================================================================

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
# EVENT EXTRACTION + PRECOMPUTATION
# ======================================================================

def extract_events(state, m5_index, bid_c, ask_c,
                   pair_idx, pip_mult, max_ec, max_xc, max_hold,
                   max_spread_pips=MAX_SPREAD_PIPS):
    """Find regime ON transitions and precompute per-event arrays.
    Skips entries where spread > max_spread_pips."""
    n = len(state)
    m5_ns = m5_index.asi8

    # Find ON transitions
    events = []
    for i in range(1, n):
        if state[i-1] == 0 and state[i] != 0:
            ts = m5_index[i]
            if ts.dayofweek >= 5: continue
            # ── Spread filter ──
            spread = (ask_c[i] - bid_c[i]) * pip_mult
            if spread > max_spread_pips:
                continue
            events.append({
                'bar': i, 'dir': int(state[i]),
                'year': ts.year, 'dow': ts.dayofweek,
                'window': ts.hour * 2 + (1 if ts.minute >= 30 else 0),
            })

    if not events: return None
    ne = len(events)

    # Preallocate
    entry_valid  = np.zeros((ne, max_ec + 1), dtype=np.bool_)
    entry_prices = np.zeros((ne, max_ec + 1), dtype=np.float64)
    exit_prices  = np.zeros((ne, max_hold + 1), dtype=np.float64)
    exit_bar_xc  = np.full((ne, max_xc + 1), max_hold, dtype=np.int32)
    entry_ns_all = np.zeros((ne, max_ec + 1), dtype=np.int64)
    exit_ns_all  = np.zeros((ne, max_hold + 1), dtype=np.int64)
    directions   = np.array([e['dir'] for e in events], dtype=np.int32)
    years        = np.array([e['year'] for e in events], dtype=np.int32)
    dows         = np.array([e['dow'] for e in events], dtype=np.int32)
    windows      = np.array([e['window'] for e in events], dtype=np.int32)
    pip_mults    = np.full(ne, pip_mult, dtype=np.float64)
    pair_indices = np.full(ne, pair_idx, dtype=np.int32)

    for ei, ev in enumerate(events):
        bar = ev['bar']; d = ev['dir']
        end = min(bar + max_hold, n - 1)

        # Entry valid + prices
        for ec in range(min(max_ec + 1, end - bar + 1)):
            b = bar + ec
            if state[b] == d:
                entry_valid[ei, ec] = True
                entry_prices[ei, ec] = ask_c[b] if d == 1 else bid_c[b]
                entry_ns_all[ei, ec] = m5_ns[b]
            else:
                break

        # Exit prices (correct side)
        for off in range(min(max_hold + 1, n - bar)):
            b = bar + off
            exit_prices[ei, off] = bid_c[b] if d == 1 else ask_c[b]
            exit_ns_all[ei, off] = m5_ns[b]
        last = min(max_hold, n - bar - 1)
        if last < max_hold:
            exit_prices[ei, last+1:] = exit_prices[ei, last]
            exit_ns_all[ei, last+1:] = exit_ns_all[ei, last]

        # Exit bar for each exit_confirm xc
        # xc=0: first bar where regime off. xc=N: off for N+1 consecutive bars
        off_run = 0; found = np.zeros(max_xc + 1, dtype=np.bool_)
        for off in range(1, min(max_hold + 1, n - bar)):
            if state[bar + off] != d:
                off_run += 1
            else:
                off_run = 0
            for xc in range(min(off_run, max_xc + 1)):
                if not found[xc]:
                    exit_bar_xc[ei, xc] = off
                    found[xc] = True
            if found.all(): break

    return {
        'entry_valid': entry_valid, 'entry_prices': entry_prices,
        'exit_prices': exit_prices, 'exit_bar_xc': exit_bar_xc,
        'directions': directions, 'pip_mults': pip_mults,
        'years': years, 'dows': dows, 'windows': windows,
        'pair_indices': pair_indices,
        'entry_ns_all': entry_ns_all, 'exit_ns_all': exit_ns_all,
        'n_events': ne,
    }


# ======================================================================
# GRID SWEEP KERNEL
# ======================================================================

def _sweep_py(ev, ep, xp, xb, dirs, pms, yidx,
              ecs, xcs, tbs, mh, ny):
    ne = len(dirs); nec = len(ecs); nxc = len(xcs); nte = len(tbs)
    sums = np.zeros((ny, nec, nxc, nte))
    sumsq = np.zeros((ny, nec, nxc, nte))
    counts = np.zeros((ny, nec, nxc, nte), dtype=np.int32)
    for e in range(ne):
        d=dirs[e]; pm=pms[e]; yr=yidx[e]
        for eci in range(nec):
            ec = ecs[eci]
            if not ev[e, ec]: continue
            epr = ep[e, ec]
            for xci in range(nxc):
                xce = xb[e, xci]
                for tei in range(nte):
                    te = tbs[tei]; eb = min(xce, ec + te)
                    if eb <= ec: eb = ec + 1
                    if eb > mh: eb = mh
                    pnl = d * (xp[e, eb] - epr) * pm
                    sums[yr, eci, xci, tei] += pnl
                    sumsq[yr, eci, xci, tei] += pnl * pnl
                    counts[yr, eci, xci, tei] += 1
    return sums, sumsq, counts

if HAS_NUMBA:
    @njit(cache=True)
    def sweep_slot(ev, ep, xp, xb, dirs, pms, yidx,
                   ecs, xcs, tbs, mh, ny):
        ne = len(dirs); nec = len(ecs); nxc = len(xcs); nte = len(tbs)
        sums = np.zeros((ny, nec, nxc, nte))
        sumsq = np.zeros((ny, nec, nxc, nte))
        counts = np.zeros((ny, nec, nxc, nte), dtype=np.int32)
        for e in range(ne):
            d=dirs[e]; pm=pms[e]; yr=yidx[e]
            for eci in range(nec):
                ec = ecs[eci]
                if not ev[e, ec]: continue
                epr = ep[e, ec]
                for xci in range(nxc):
                    xce = xb[e, xci]
                    for tei in range(nte):
                        te = tbs[tei]; eb = min(xce, ec + te)
                        if eb <= ec: eb = ec + 1
                        if eb > mh: eb = mh
                        pnl = d * (xp[e, eb] - epr) * pm
                        sums[yr, eci, xci, tei] += pnl
                        sumsq[yr, eci, xci, tei] += pnl * pnl
                        counts[yr, eci, xci, tei] += 1
        return sums, sumsq, counts
else:
    sweep_slot = _sweep_py


# ======================================================================
# VECTORISED WALK-FORWARD (all combos at once per slot)
# ======================================================================

def _bonferroni_significant(train_avg, train_std, train_n, n_combos_tested,
                            alpha=0.05):
    """Check if training performance is significant after Bonferroni correction.
    Uses a one-sided t-test: H0 mean <= 0, H1 mean > 0.
    Adjusted alpha = alpha / n_combos_tested."""
    if train_n < 2 or train_std <= 0 or train_avg <= 0:
        return False
    t_stat = train_avg / (train_std / np.sqrt(train_n))
    p_value = 1.0 - scipy_stats.t.cdf(t_stat, df=train_n - 1)
    adjusted_alpha = alpha / max(n_combos_tested, 1)
    return p_value < adjusted_alpha


def walk_forward_all_combos(slot_sums_dict, slot_counts_dict,
                            slot_sumsq_dict, n_years,
                            min_train_trades, min_test_trades):
    """
    Vectorised walk-forward across ALL combos simultaneously.
    Returns (total_pips, total_trades) matrices of shape (n_ec, n_xc, n_te).
    Also returns per-slot best combo results for portfolio simulation.

    Applies Bonferroni-corrected significance testing:
    - A combo must pass a one-sided t-test (H0: mean<=0) at
      alpha / n_combos_tested to be selected.
    """
    if not slot_sums_dict:
        return None, None, {}, {}

    # Get combo shape from first slot
    first_key = next(iter(slot_sums_dict))
    combo_shape = slot_sums_dict[first_key].shape[1:]  # (n_ec, n_xc, n_te)
    n_combos = int(np.prod(combo_shape))

    # Global combo-level accumulators
    total_pips = np.zeros(combo_shape, dtype=np.float64)
    total_trades = np.zeros(combo_shape, dtype=np.int64)

    # Per-slot best combo results (for portfolio sim)
    slot_best_results = {}    # (dow, window) → list of fold dicts
    slot_best_configs = {}    # (test_year_idx, dow, window) → combo config

    # Total number of hypotheses = n_slots × n_combos
    n_slots = len(slot_sums_dict)
    n_hypotheses = n_slots * n_combos

    for (d, w), sums in slot_sums_dict.items():
        counts = slot_counts_dict[(d, w)]
        sumsq = slot_sumsq_dict[(d, w)]
        # sums: (n_years, n_ec, n_xc, n_te)

        cum_sums = np.cumsum(sums, axis=0)
        cum_counts = np.cumsum(counts, axis=0)
        cum_sumsq = np.cumsum(sumsq, axis=0)

        fold_results = []

        for ti in range(1, n_years):
            # Training: cumulative to year ti-1
            tr_s = cum_sums[ti - 1]
            tr_c = cum_counts[ti - 1]
            tr_sq = cum_sumsq[ti - 1]

            valid_train = tr_c >= min_train_trades
            tr_avg = np.where(valid_train, tr_s / np.maximum(tr_c, 1), -1e10)
            # Compute training variance for t-test
            tr_var = np.where(
                tr_c >= 2,
                (tr_sq - tr_s**2 / np.maximum(tr_c, 1)) / np.maximum(tr_c - 1, 1),
                0.0
            )
            tr_std = np.sqrt(np.maximum(tr_var, 0.0))

            # Test year ti
            te_s = sums[ti]
            te_c = counts[ti]
            valid_test = te_c >= min_test_trades
            te_avg = np.where(valid_test, te_s / np.maximum(te_c, 1), -1e10)

            # ── Bonferroni-corrected significance for global accumulator ──
            # For each combo: must be significant after correcting for all hypotheses
            adjusted_alpha = 0.05 / max(n_hypotheses, 1)

            t_stat = np.where(
                valid_train & (tr_std > 0),
                tr_avg / (tr_std / np.sqrt(np.maximum(tr_c, 1).astype(np.float64))),
                0.0
            )
            p_value = np.where(
                t_stat > 0,
                1.0 - scipy_stats.t.cdf(t_stat, df=np.maximum(tr_c - 1, 1)),
                1.0
            )
            sig_train = (p_value < adjusted_alpha) & valid_train
            oos_prof = sig_train & valid_test & (te_avg > 0)

            # ── Global combo accumulator ──
            total_pips += np.where(oos_prof, te_s, 0.0)
            total_trades += np.where(oos_prof, te_c, 0)

            # ── Per-slot best combo (for portfolio) ──
            # Also requires Bonferroni significance
            if not sig_train.any():
                continue
            # Among significant combos, pick the one with best train avg
            masked_avg = np.where(sig_train, tr_avg, -1e10)
            best_idx = np.unravel_index(masked_avg.argmax(), combo_shape)
            best_train_avg = float(tr_avg[best_idx])
            if best_train_avg <= 0:
                continue
            test_n = int(te_c[best_idx])
            if test_n < min_test_trades:
                continue
            test_total = float(te_s[best_idx])
            test_avg = test_total / test_n
            best_train_n = int(tr_c[best_idx])
            best_train_std = float(tr_std[best_idx])

            fold_results.append({
                'test_year_idx': ti,
                'best_combo': best_idx,
                'train_n': best_train_n,
                'train_avg': round(best_train_avg, 4),
                'train_std': round(best_train_std, 4),
                'test_n': test_n,
                'test_total': round(test_total, 2),
                'test_avg': round(test_avg, 4),
                'oos_profitable': test_avg > 0,
                'bonferroni_alpha': adjusted_alpha,
            })

            if test_avg > 0:
                slot_best_configs[(ti, d, w)] = best_idx

        if fold_results:
            slot_best_results[(d, w)] = fold_results

    return total_pips, total_trades, slot_best_results, slot_best_configs


# ======================================================================
# PORTFOLIO SIMULATION
# ======================================================================

def simulate_portfolio(all_ev, slot_configs, years_list,
                       entry_confirms, exit_confirms, timed_bars):
    """Replay test years with per-slot best configs, position-blocked."""
    rows = []

    for ti in range(1, len(years_list)):
        test_yr = years_list[ti]
        train_yrs = years_list[:ti]
        candidates = []

        for (ty_idx, dow, win), combo_idx in slot_configs.items():
            if ty_idx != ti: continue
            eci, xci, tei = combo_idx
            ec_val = int(entry_confirms[eci])
            xc_val = int(exit_confirms[xci])
            te_val = int(timed_bars[tei])

            mask = ((all_ev['years'] == test_yr) &
                    (all_ev['dows'] == dow) & (all_ev['windows'] == win))
            for idx in np.where(mask)[0]:
                if not all_ev['entry_valid'][idx, ec_val]: continue

                xc_exit = int(all_ev['exit_bar_xc'][idx, xc_val])
                exit_bar = min(xc_exit, ec_val + te_val)
                if exit_bar <= ec_val: exit_bar = ec_val + 1
                if exit_bar > MAX_HOLD_BARS: exit_bar = MAX_HOLD_BARS

                d = int(all_ev['directions'][idx])
                pm = float(all_ev['pip_mults'][idx])
                ep = float(all_ev['entry_prices'][idx, ec_val])
                xp = float(all_ev['exit_prices'][idx, exit_bar])
                pnl = d * (xp - ep) * pm

                ens = int(all_ev['entry_ns_all'][idx, ec_val])
                xns = int(all_ev['exit_ns_all'][idx, exit_bar])
                pid = int(all_ev['pair_indices'][idx])
                candidates.append((ens, xns, pid, pnl))

        if not candidates: continue
        candidates.sort()

        last_exit = {}; pnls = []
        for ens, xns, pid, pnl in candidates:
            if pid in last_exit and ens < last_exit[pid]: continue
            pnls.append(pnl); last_exit[pid] = xns

        if len(pnls) < 3: continue
        pa = np.array(pnls)
        gl = abs(pa[pa<0].sum()) if (pa<0).any() else 1.0
        cum = np.cumsum(pa); dd = float((np.maximum.accumulate(cum) - cum).max())
        n_slots = sum(1 for (ty,_,_) in slot_configs if ty == ti)

        rows.append({
            'train_years': str(train_yrs), 'test_year': test_yr,
            'n_slots': n_slots,
            'test_n_signals': len(candidates), 'test_n_traded': len(pa),
            'test_n_blocked': len(candidates) - len(pa),
            'test_total_pips': round(pa.sum(), 1),
            'test_avg_pips': round(pa.mean(), 3),
            'test_win_rate': round((pa>0).mean()*100, 1),
            'test_pf': round(pa[pa>0].sum()/gl, 3) if gl > 0 else 0.0,
            'test_max_dd': round(dd, 1),
        })

    return pd.DataFrame(rows)


# ======================================================================
# OUTPUT
# ======================================================================

def save_outputs(slot_df, port_df, summary_df, output_dir):
    ts = time_mod.strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)
    for name, df in [('regime_v2_slots', slot_df),
                     ('regime_v2_portfolio', port_df),
                     ('regime_v2_summary', summary_df)]:
        if df is not None and len(df) > 0:
            path = os.path.join(output_dir, f'{name}_{ts}.csv')
            df.to_csv(path, index=False)
            log.info(f"Saved: {path} ({len(df):,} rows)")


def print_summary(summary_df, port_df, slot_df, years_list,
                  entry_confirms, exit_confirms, timed_bars):
    print("\n" + "=" * 100)
    print("REGIME BACKTEST v2 — RESULTS")
    print("=" * 100)

    if summary_df is None or len(summary_df) == 0:
        print("No results!"); return

    best = summary_df.loc[summary_df['total_oos_pips'].idxmax()]
    ecb = int(best['entry_confirm_bars']); xcb = int(best['exit_confirm_bars'])
    teb = int(best['timed_exit_bars'])

    print(f"\n── BEST CONFIG ──")
    print(f"  Entry threshold:  {best['entry_thresh']:.1f}")
    print(f"  Exit threshold:   {best['exit_thresh']:.1f}")
    print(f"  Entry confirm:    {ecb} bars ({ecb*5} min)")
    print(f"  Exit confirm:     {xcb} bars ({xcb*5} min)")
    print(f"  Timed exit:       {teb} bars ({teb*5}min / {teb*5/60:.1f}h)")
    print(f"  OOS total:        {best['total_oos_pips']:+,.1f} pips")
    print(f"  OOS trades:       {int(best['total_trades']):,}")
    print(f"  OOS avg:          {best['avg_oos_pips']:+.4f} pips/trade")

    print(f"\n── TOP 20 CONFIGS ──")
    hdr = f"  {'ET':>4} {'XT':>4} {'EC':>6} {'XC':>6} {'TE':>8} │ {'OOS Pips':>12} {'Trades':>8} {'Avg':>8}"
    print(hdr); print("  " + "─" * len(hdr))
    for _, r in summary_df.nlargest(20, 'total_oos_pips').iterrows():
        te_min = int(r['timed_exit_bars']) * 5
        te_str = f"{te_min/60:.1f}h" if te_min >= 60 else f"{te_min}m"
        print(f"  {r['entry_thresh']:>4.1f} {r['exit_thresh']:>4.1f} "
              f"{int(r['entry_confirm_bars'])*5:>5}m {int(r['exit_confirm_bars'])*5:>5}m "
              f"{te_str:>8} │ {r['total_oos_pips']:>+12,.1f} {int(r['total_trades']):>8,} "
              f"{r['avg_oos_pips']:>+8.4f}")

    # ── Dimension impact charts ──
    bt_e = best['entry_thresh']; bt_x = best['exit_thresh']
    bt_rows = summary_df[(summary_df['entry_thresh']==bt_e) & (summary_df['exit_thresh']==bt_x)]

    for dim_name, dim_col, fmt_fn in [
        ('ENTRY CONFIRM', 'entry_confirm_bars', lambda x: f"{int(x)*5:>3}min"),
        ('EXIT CONFIRM', 'exit_confirm_bars', lambda x: f"{int(x)*5:>3}min"),
    ]:
        print(f"\n── {dim_name} IMPACT (ET={bt_e:.1f}, XT={bt_x:.1f}) ──")
        impact = bt_rows.groupby(dim_col)['total_oos_pips'].max().sort_index()
        mx = max(impact.max(), 1)
        for v, p in impact.items():
            bar = "█" * max(1, int(max(0, p) / mx * 40))
            print(f"  {fmt_fn(v)}: {p:>+12,.1f} {bar}")

    print(f"\n── TIMED EXIT IMPACT (ET={bt_e:.1f}, XT={bt_x:.1f}, top 25) ──")
    te_imp = bt_rows.groupby('timed_exit_bars')['total_oos_pips'].max().sort_values(ascending=False)
    mx = max(te_imp.max(), 1)
    for te, p in te_imp.head(25).items():
        mins = int(te)*5
        label = f"{mins/60:.1f}h" if mins >= 60 else f"{mins}m"
        bar = "█" * max(1, int(max(0, p) / mx * 40))
        print(f"  {label:>8}: {p:>+12,.1f} {bar}")

    # ── Threshold pair impact ──
    print(f"\n── THRESHOLD PAIR IMPACT (best combo per pair) ──")
    tp = summary_df.groupby(['entry_thresh','exit_thresh'])['total_oos_pips'].max()
    tp = tp.sort_values(ascending=False)
    for (et2, xt2), p in tp.head(15).items():
        print(f"  ET={et2:.1f} XT={xt2:.1f}: {p:>+12,.1f} pips")

    # ── Portfolio ──
    if port_df is not None and len(port_df) > 0:
        print(f"\n── BEST CONFIG — PORTFOLIO (position-blocked) ──")
        for _, r in port_df.iterrows():
            print(f"  Test {r['test_year']}: {r['n_slots']} slots | "
                  f"{r['test_n_traded']}/{r['test_n_signals']} traded | "
                  f"{r['test_total_pips']:+,.1f} pips | {r['test_avg_pips']:+.3f} avg | "
                  f"{r['test_win_rate']:.1f}% WR | PF {r['test_pf']:.3f} | DD {r['test_max_dd']:.0f}")
        total = port_df['test_total_pips'].sum()
        trades = port_df['test_n_traded'].sum()
        print(f"  TOTAL: {total:+,.1f} pips / {trades:,} trades ({total/trades:+.3f} avg)")

    # ── Consistent slots ──
    if slot_df is not None and len(slot_df) > 0:
        prof = slot_df[slot_df['oos_profitable'] == True]
        if len(prof) > 0:
            cons = prof.groupby(['dow_name','window_utc']).agg(
                yrs=('test_year','count'), avg=('test_avg_pips','mean'),
                total=('test_total_pips','sum'), avg_trades=('test_n','mean'),
            ).sort_values('yrs', ascending=False)
            n_test = len(years_list) - 1
            print(f"\n── MOST CONSISTENT SLOTS ──")
            for (dn, wl), r in cons.head(25).iterrows():
                print(f"  {dn} {wl} | {int(r['yrs'])}/{n_test} yrs | "
                      f"avg {r['avg']:+.3f} | total {r['total']:+,.1f} | ~{r['avg_trades']:.0f}/yr")
            for t in [7,6,5,4,3]:
                ct = (cons['yrs']>=t).sum()
                if ct > 0: print(f"\n  Slots profitable {t}+/{n_test} yrs: {ct}")

    print("\n" + "=" * 100)


# ======================================================================
# PAIR DISCOVERY
# ======================================================================

def discover_pairs(data_dir):
    pairs = {}
    for ext in ['*.parquet', '*.pkl']:
        for f in glob.glob(os.path.join(data_dir, ext)):
            base = os.path.basename(f).upper()
            for p in ALL_PAIRS:
                if (p in base or p.replace('_', '') in base) and p not in pairs:
                    pairs[p] = f
    return pairs


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Regime Transition Backtester v2')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--fine-grid', action='store_true',
                        help='Full 5-min resolution timed exit (576 values)')
    parser.add_argument('--entry-thresholds', type=str, default=None)
    parser.add_argument('--exit-thresholds', type=str, default=None)
    parser.add_argument('--max-spread', type=float, default=MAX_SPREAD_PIPS,
                        help='Max spread in pips to allow entry (default: 5.0)')
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir:
        for c in [r'C:\Forex_Projects\5_year_5s_data', r'C:\Forex_Projects\Forex_bot_original']:
            if os.path.isdir(c): data_dir = c; break
        if not data_dir: data_dir = os.getcwd()
    output_dir = args.output_dir or data_dir

    if args.entry_thresholds:
        entry_thresholds = [float(x) for x in args.entry_thresholds.split(',')]
    else:
        entry_thresholds = DEFAULT_ENTRY_THRESHOLDS
    if args.exit_thresholds:
        exit_thresholds = [float(x) for x in args.exit_thresholds.split(',')]
    else:
        exit_thresholds = DEFAULT_EXIT_THRESHOLDS

    entry_confirms = np.array(ENTRY_CONFIRM_VALUES, dtype=np.int32)
    exit_confirms = np.array(EXIT_CONFIRM_VALUES, dtype=np.int32)
    max_ec = int(entry_confirms.max())
    max_xc = int(exit_confirms.max())
    max_spread = args.max_spread
    timed_bars = make_timed_exit_grid(fine=args.fine_grid)

    threshold_pairs = [(et, xt) for et in entry_thresholds
                       for xt in exit_thresholds if xt <= et]

    n_combos = len(entry_confirms) * len(exit_confirms) * len(timed_bars)
    total = len(threshold_pairs) * n_combos

    t_start = time_mod.time()
    log.info(f"Data dir:          {data_dir}")
    log.info(f"Output dir:        {output_dir}")
    log.info(f"Threshold pairs:   {len(threshold_pairs)} (exit ≤ entry)")
    log.info(f"Entry confirm:     {ENTRY_CONFIRM_VALUES} bars")
    log.info(f"Exit confirm:      {EXIT_CONFIRM_VALUES} bars")
    log.info(f"Timed exit values: {len(timed_bars)} ({timed_bars[0]*5}min to {timed_bars[-1]*5/60:.0f}h)")
    log.info(f"Windows:           {N_WINDOWS} (30-min UTC slots)")
    log.info(f"Max spread:        {max_spread} pips")
    log.info(f"Min train trades:  {MIN_TRAIN_TRADES}")
    log.info(f"Min test trades:   {MIN_TEST_TRADES}")
    log.info(f"Combos/threshold:  {n_combos:,}  |  Total: {total:,}")
    log.info(f"Bonferroni:        YES (alpha=0.05 / n_hypotheses)")
    log.info(f"Numba:             {'YES' if HAS_NUMBA else 'NO (slow)'}")
    log.info(f"TA-Lib:            {'YES' if HAS_TALIB else 'NO (numpy fallback)'}")

    pairs = discover_pairs(data_dir)
    if not pairs: log.error(f"No parquet files in {data_dir}"); sys.exit(1)
    pair_names = sorted(pairs.keys())
    pair_to_idx = {p: i for i, p in enumerate(pair_names)}
    log.info(f"Pairs:             {len(pairs)} — {', '.join(pair_names)}")

    # ══ PHASE 1: Load + MTF bias ══
    log.info(f"\n{'='*60}\nPHASE 1: Loading data")
    pair_bias = {}; pair_m5 = {}; pair_ba = {}

    for pname in pair_names:
        t0 = time_mod.time()
        df_5s = load_pair_data(pairs[pname])
        tfs = build_timeframes(df_5s)
        for tf in tfs:
            if len(tfs[tf]) >= 22: tfs[tf] = add_ema(tfs[tf])
        m5 = tfs['M5']
        if len(m5) < 100:
            log.warning(f"  {pname}: skip ({len(m5)} M5 bars)"); continue
        bias = compute_mtf_bias(tfs, m5.index)
        has_ba = 'bid_close' in m5.columns
        bc = m5['bid_close'].values.astype(np.float64) if has_ba else m5['close'].values.astype(np.float64)
        ac = m5['ask_close'].values.astype(np.float64) if has_ba else m5['close'].values.astype(np.float64)
        pair_bias[pname] = bias; pair_m5[pname] = m5; pair_ba[pname] = (bc, ac)
        log.info(f"  {pname}: {len(m5):,} M5 bars ({time_mod.time()-t0:.1f}s)")
    del df_5s

    # ══ PHASE 2: Regime events + sweep per threshold pair ══
    log.info(f"\n{'='*60}\nPHASE 2: Grid sweep")

    best_overall_pips = -1e10
    best_thresh_pair = None
    best_events_data = None
    best_slot_configs = None
    best_slot_results = None
    all_summary_rows = []

    for tp_i, (et, xt) in enumerate(threshold_pairs):
        t0 = time_mod.time()
        log.info(f"\n  [{tp_i+1}/{len(threshold_pairs)}] ET={et:.1f} XT={xt:.1f}")

        # Extract events across all pairs
        all_ev = {k: [] for k in ['entry_valid','entry_prices','exit_prices','exit_bar_xc',
                                   'directions','pip_mults','years','dows','windows',
                                   'pair_indices','entry_ns_all','exit_ns_all']}
        total_ev = 0

        for pname in pair_names:
            if pname not in pair_bias: continue
            state = regime_hysteresis(pair_bias[pname], et, xt)
            bc, ac = pair_ba[pname]
            ev = extract_events(state, pair_m5[pname].index, bc, ac,
                                pair_to_idx[pname],
                                100.0 if 'JPY' in pname else 10000.0,
                                max_ec, max_xc, MAX_HOLD_BARS,
                                max_spread_pips=max_spread)
            if ev is None: continue
            total_ev += ev['n_events']
            for k in all_ev: all_ev[k].append(ev[k])

        if total_ev == 0:
            log.info(f"    0 events"); continue
        for k in all_ev: all_ev[k] = np.concatenate(all_ev[k])

        yrs = sorted(set(all_ev['years'].tolist()))
        y2i = {y: i for i, y in enumerate(yrs)}
        ny = len(yrs)
        yidx = np.array([y2i[y] for y in all_ev['years']], dtype=np.int32)

        log.info(f"    {total_ev:,} events, {ny} years: {yrs}")
        if ny < 2: continue

        # Sweep per slot
        slot_sums = {}; slot_counts = {}; slot_sumsq = {}; ns = 0
        for d in range(5):
            for w in range(N_WINDOWS):
                mask = (all_ev['dows'] == d) & (all_ev['windows'] == w)
                if mask.sum() < MIN_TRAIN_TRADES: continue
                idx = np.where(mask)[0]
                s, sq, c = sweep_slot(
                    all_ev['entry_valid'][idx], all_ev['entry_prices'][idx],
                    all_ev['exit_prices'][idx], all_ev['exit_bar_xc'][idx],
                    all_ev['directions'][idx], all_ev['pip_mults'][idx],
                    yidx[idx], entry_confirms, exit_confirms, timed_bars,
                    MAX_HOLD_BARS, ny)
                slot_sums[(d,w)] = s; slot_counts[(d,w)] = c
                slot_sumsq[(d,w)] = sq; ns += 1

        # Vectorised walk-forward with Bonferroni correction
        tp_mat, tt_mat, slot_results, slot_configs = walk_forward_all_combos(
            slot_sums, slot_counts, slot_sumsq, ny,
            MIN_TRAIN_TRADES, MIN_TEST_TRADES)

        if tp_mat is None:
            log.info(f"    No results"); continue

        # Build summary rows from the combo matrix
        # Find top N combos to save (not all — could be 20K+)
        flat = tp_mat.ravel()
        top_n = min(500, len(flat))
        top_idx = np.argpartition(flat, -top_n)[-top_n:]
        top_idx = top_idx[flat[top_idx] > 0]  # only profitable

        for fi in top_idx:
            ci = np.unravel_index(fi, tp_mat.shape)
            pips = float(tp_mat[ci])
            trades = int(tt_mat[ci])
            if trades == 0: continue
            all_summary_rows.append({
                'entry_thresh': et, 'exit_thresh': xt,
                'entry_confirm_bars': int(entry_confirms[ci[0]]),
                'exit_confirm_bars': int(exit_confirms[ci[1]]),
                'timed_exit_bars': int(timed_bars[ci[2]]),
                'total_oos_pips': round(pips, 1),
                'total_trades': trades,
                'avg_oos_pips': round(pips / trades, 4),
                'losing_years': 0,  # computed at portfolio level
            })

        best_pips = float(tp_mat.max())
        log.info(f"    {ns} slots | best combo OOS: {best_pips:+,.1f} pips ({time_mod.time()-t0:.1f}s)")

        if best_pips > best_overall_pips:
            best_overall_pips = best_pips
            best_thresh_pair = (et, xt)
            best_events_data = {k: v.copy() for k, v in all_ev.items()}
            best_events_data['year_list'] = yrs
            best_slot_configs = dict(slot_configs)
            best_slot_results = dict(slot_results)

    # ══ PHASE 3: Portfolio simulation ══
    log.info(f"\n{'='*60}\nPHASE 3: Portfolio simulation")

    port_df = pd.DataFrame()
    best_slot_df = pd.DataFrame()

    if best_events_data is not None:
        et, xt = best_thresh_pair
        yl = best_events_data['year_list']
        log.info(f"  Best: ET={et:.1f} XT={xt:.1f}")

        port_df = simulate_portfolio(best_events_data, best_slot_configs,
                                     yl, entry_confirms, exit_confirms, timed_bars)
        if len(port_df) > 0:
            t = port_df['test_total_pips'].sum()
            n = port_df['test_n_traded'].sum()
            log.info(f"  Portfolio: {t:+,.1f} pips / {n:,} trades ({t/n:+.3f} avg)")

        # Build slot detail
        srows = []
        for (d, w), results in best_slot_results.items():
            for r in results:
                bi = r['best_combo']
                srows.append({
                    'train_years': str([yl[i] for i in range(r['test_year_idx'])]),
                    'test_year': yl[r['test_year_idx']],
                    'dow': d, 'dow_name': DOW_NAMES[d],
                    'window': w, 'window_utc': f"{w//2:02d}:{(w%2)*30:02d}",
                    'entry_thresh': et, 'exit_thresh': xt,
                    'entry_confirm_min': int(entry_confirms[bi[0]]) * 5,
                    'exit_confirm_min': int(exit_confirms[bi[1]]) * 5,
                    'timed_exit_bars': int(timed_bars[bi[2]]),
                    'timed_exit_min': int(timed_bars[bi[2]]) * 5,
                    'train_n': r['train_n'], 'train_avg_pips': r['train_avg'],
                    'test_n': r['test_n'], 'test_total_pips': r['test_total'],
                    'test_avg_pips': r['test_avg'], 'oos_profitable': r['oos_profitable'],
                })
        best_slot_df = pd.DataFrame(srows)

    # ══ PHASE 4: Output ══
    summary_df = pd.DataFrame(all_summary_rows) if all_summary_rows else pd.DataFrame()
    save_outputs(best_slot_df, port_df, summary_df, output_dir)
    print_summary(summary_df, port_df, best_slot_df,
                  best_events_data['year_list'] if best_events_data else [],
                  entry_confirms, exit_confirms, timed_bars)

    elapsed = time_mod.time() - t_start
    log.info(f"\nTotal runtime: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == '__main__':
    main()
