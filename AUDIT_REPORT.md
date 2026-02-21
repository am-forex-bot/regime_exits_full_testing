# Forensic Audit Report: regime_backtest_v2.py

## Executive Summary

Line-by-line forensic audit of `regime_backtest_v2.py` (869 lines) and
`regime_simulator.py` (785 lines), plus all output CSVs.

**The code is structurally legitimate — no outright cheating, no hardcoded
returns, no explicit future data access.** However, there are **significant
methodological concerns** (selection bias from massive parameter grid) that
inflate the reported results and make them unreliable as a prediction of live
performance.

---

## 1. Lookahead Bias Audit

### 1a. MTF Bias Computation — `compute_mtf_bias()` (line 146)

**CLEAN.** The alignment at line 160 uses `searchsorted(tt, m5_ns, side='right') - 1`,
which maps each M5 bar to the most recent completed higher-timeframe bar. Correct
causal approach.

### 1b. EMA Computation — `_ema_np()` (lines 124-129)

**CLEAN.** Sequential forward computation. `out[i]` depends only on `arr[i]` and
`out[i-1]`. No future data.

### 1c. Regime State Machine — `regime_hysteresis()` (line 170)

**CLEAN.** Forward loop. `state[i]` computed from `bias[i]` and previous state `cur`.

### 1d. Event Extraction — `extract_events()` (line 215)

**CLEAN.** Regime ON transitions at line 223: `state[i-1] == 0 and state[i] != 0`.
Causal comparison. Forward scanning for entry/exit is correct (real-time behavior).

### 1e. Timeframe Resampling — `build_timeframes()` (line 100)

**CLEAN.** Standard pandas `.resample()` with OHLC aggregation. Left-edge labeling
by default.

---

## 2. Train/Test Split Integrity

### 2a. Walk-Forward Structure (line 354)

**STRUCTURALLY CORRECT.**
- Training: `cum_sums[ti - 1]` — cumulative to previous year
- Testing: `sums[ti]` — only current test year
- No data from the test year enters training

### 2b. CRITICAL: Massive Selection Bias

The walk-forward searches:
- 5 days × 48 windows = **240 slots**
- Per slot: 13 entry_confirm × 13 exit_confirm × ~120 timed_exit = **~20,000 combos**
- Total: **~4.8 million parallel strategies**

With this many degrees of freedom, the Bonferroni correction would require p < 1e-8
for significance. No such correction is applied. The system picks the single best
training combo per slot and applies it to the test year.

### 2c. Low Test Threshold

Line 396: `valid_test = te_c >= 3` — only 3 trades required. A slot with 3 trades
has ~50% chance of appearing profitable by noise alone.

---

## 3. Data Leakage Audit

### 3a. Indicator Leakage — CLEAN

No cross-validation-aware fitting, no normalization across full dataset.

### 3b. Feature Engineering — CLEAN

Simple weighted MTF directional signals. No learned features.

---

## 4. Trade Execution Audit

### 4a. Bid/Ask Pricing — CORRECT

- Long entries use ask, exits use bid (lines 259, 267)
- Short entries use bid, exits use ask
- Spread correctly paid both ways

### 4b. Slippage — CORRECT

0.3 pips/side (0.6 round trip) applied in simulator on top of bid/ask.
Reasonable for majors, arguably light for exotic crosses.

### 4c. Position Blocking — CORRECT

Per-pair single position + max concurrent limit enforced.

---

## 5. Statistical Plausibility

| Metric | Value | Assessment |
|--------|-------|------------|
| Total trades (6.15 yrs) | 98,812 | ~16,000/yr — very high |
| Avg pips/trade | +5.998 | **Extremely high** |
| Win rate | 44.4% | Plausible for trend system |
| Profit factor | 2.296 | **Very high** |
| Max drawdown | 981 pips | Reasonable vs total |
| All 19 pairs profitable | Yes | **Suspicious** |
| All 7 years profitable | Yes | **Suspicious** |
| Return autocorrelation | 0.263 | Elevated (correlated positions) |

### Key Concerns

1. **+6 pips/trade is unrealistic.** Professional FX trend systems achieve
   +0.5 to +2 pips net. With ~2-4 pip total costs, this implies an 8-10 pip
   gross edge — would be among the most profitable systematic FX strategies
   ever.

2. **All 19 pairs profitable** is statistically implausible for a real edge.
   Even the best strategies have losing pairs.

3. **2026 partial year** (7 weeks) shows the best metrics (9.044 avg,
   49.8% WR, 3.664 PF) — common for overfitted systems before mean reversion.

---

## 6. Root Cause: Selection Bias

The code does not cheat. But the methodology inflates results:

1. **4.8M strategies tested** with no multiple-testing correction
2. **Slot-level optimization** with small training samples (15-50 trades)
3. **Survivorship in slot selection** — only OOS-profitable slots count
4. **No Bonferroni or FDR correction** for the number of hypotheses tested

Likely real-world performance: the underlying signal may have a small positive
edge (+0.5 to +1.5 pips/trade), but reported +6 pips/trade is inflated ~4-10x
by selection effects.

---

## 7. Verdict Summary

| Category | Finding |
|----------|---------|
| Lookahead bias | CLEAN |
| Train/test separation | CLEAN |
| Indicator leakage | CLEAN |
| Trade execution | CLEAN |
| Code integrity | CLEAN |
| Selection bias | **SEVERE** |
| Results plausibility | **IMPLAUSIBLE at reported magnitude** |

---

## 8. Recommendations

1. **Reduce parameter space:** Fix timed exit and confirmation bars; only
   optimize thresholds
2. **Apply statistical corrections:** Bonferroni, FDR, or White's Reality
   Check for multiple testing
3. **Increase minimum trade counts:** Require 50+ trades in training, 20+
   in test per slot
4. **Reduce slot granularity:** Use 4-hour windows instead of 30-minute
   (240 → 30 slots)
5. **Out-of-sample holdout:** Reserve 2025-2026 as a true holdout never
   touched during development
