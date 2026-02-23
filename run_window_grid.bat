@echo off
setlocal

set DATA=C:\Forex_Projects\5_year_5s_data
set SLOTS=C:\Forex_Projects\Forex_bot_original\Testing\claude_code_version\regime_v2_slots_20260222_202854.csv
set OUT=C:\Forex_Projects\Forex_bot_original\Testing\claude_code_version

echo ============================================
echo  REGIME ACTIVE WINDOW GRID SWEEP
echo  Windows: 45 / 60 / 90 / 120 / 180 / 240 min
echo  28 runs (14 baseline + 14 stress)
echo ============================================

echo ============================================
echo  ROUND 1: BASELINE (0.3 slip, 0 delay)
echo ============================================

echo [1/28] Baseline - Timed only (reference)
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --timed-only

echo [2/28] Baseline - Timed only + Friday exit (reference)
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --timed-only --friday-exit 20

echo [3/28] Baseline - Window 45
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --regime-exit-window 45

echo [4/28] Baseline - Window 45 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --regime-exit-window 45 --friday-exit 20

echo [5/28] Baseline - Window 60
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --regime-exit-window 60

echo [6/28] Baseline - Window 60 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --regime-exit-window 60 --friday-exit 20

echo [7/28] Baseline - Window 90
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --regime-exit-window 90

echo [8/28] Baseline - Window 90 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --regime-exit-window 90 --friday-exit 20

echo [9/28] Baseline - Window 120
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --regime-exit-window 120

echo [10/28] Baseline - Window 120 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --regime-exit-window 120 --friday-exit 20

echo [11/28] Baseline - Window 180
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --regime-exit-window 180

echo [12/28] Baseline - Window 180 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --regime-exit-window 180 --friday-exit 20

echo [13/28] Baseline - Window 240
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --regime-exit-window 240

echo [14/28] Baseline - Window 240 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --regime-exit-window 240 --friday-exit 20

echo ============================================
echo  ROUND 2: STRESS (3.0 slip, 10min delay)
echo ============================================

echo [15/28] Stress - Timed only (reference)
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --timed-only

echo [16/28] Stress - Timed only + Friday exit (reference)
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --timed-only --friday-exit 20

echo [17/28] Stress - Window 45
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --regime-exit-window 45

echo [18/28] Stress - Window 45 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --regime-exit-window 45 --friday-exit 20

echo [19/28] Stress - Window 60
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --regime-exit-window 60

echo [20/28] Stress - Window 60 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --regime-exit-window 60 --friday-exit 20

echo [21/28] Stress - Window 90
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --regime-exit-window 90

echo [22/28] Stress - Window 90 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --regime-exit-window 90 --friday-exit 20

echo [23/28] Stress - Window 120
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --regime-exit-window 120

echo [24/28] Stress - Window 120 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --regime-exit-window 120 --friday-exit 20

echo [25/28] Stress - Window 180
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --regime-exit-window 180

echo [26/28] Stress - Window 180 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --regime-exit-window 180 --friday-exit 20

echo [27/28] Stress - Window 240
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --regime-exit-window 240

echo [28/28] Stress - Window 240 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --regime-exit-window 240 --friday-exit 20

echo ============================================
echo  ALL 28 RUNS COMPLETE
echo ============================================
pause
