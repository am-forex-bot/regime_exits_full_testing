@echo off
setlocal

set DATA=C:\Forex_Projects\5_year_5s_data
set SLOTS=C:\Forex_Projects\Forex_bot_original\Testing\claude_code_version\regime_v2_slots_20260222_202854.csv
set OUT=C:\Forex_Projects\Forex_bot_original\Testing\claude_code_version

echo ============================================
echo  ROUND 1: BASELINE (0.3 slip, 0 delay)
echo ============================================

echo [1/32] Baseline - Timed only
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --timed-only

echo [2/32] Baseline - Timed only + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --timed-only --friday-exit 20

echo [3/32] Baseline - Min hold 15
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --min-hold 15

echo [4/32] Baseline - Min hold 15 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --min-hold 15 --friday-exit 20

echo [5/32] Baseline - Min hold 30
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --min-hold 30

echo [6/32] Baseline - Min hold 30 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --min-hold 30 --friday-exit 20

echo [7/32] Baseline - Min hold 60
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --min-hold 60

echo [8/32] Baseline - Min hold 60 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --min-hold 60 --friday-exit 20

echo [9/32] Baseline - Min hold 120
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --min-hold 120

echo [10/32] Baseline - Min hold 120 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --min-hold 120 --friday-exit 20

echo [11/32] Baseline - Window 10
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --regime-exit-window 10

echo [12/32] Baseline - Window 10 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --regime-exit-window 10 --friday-exit 20

echo [13/32] Baseline - Window 20
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --regime-exit-window 20

echo [14/32] Baseline - Window 20 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --regime-exit-window 20 --friday-exit 20

echo [15/32] Baseline - Window 30
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --regime-exit-window 30

echo [16/32] Baseline - Window 30 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --regime-exit-window 30 --friday-exit 20

echo ============================================
echo  ROUND 2: STRESS (3.0 slip, 10min delay)
echo ============================================

echo [17/32] Stress - Timed only
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --timed-only

echo [18/32] Stress - Timed only + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --timed-only --friday-exit 20

echo [19/32] Stress - Min hold 15
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --min-hold 15

echo [20/32] Stress - Min hold 15 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --min-hold 15 --friday-exit 20

echo [21/32] Stress - Min hold 30
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --min-hold 30

echo [22/32] Stress - Min hold 30 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --min-hold 30 --friday-exit 20

echo [23/32] Stress - Min hold 60
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --min-hold 60

echo [24/32] Stress - Min hold 60 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --min-hold 60 --friday-exit 20

echo [25/32] Stress - Min hold 120
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --min-hold 120

echo [26/32] Stress - Min hold 120 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --min-hold 120 --friday-exit 20

echo [27/32] Stress - Window 10
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --regime-exit-window 10

echo [28/32] Stress - Window 10 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --regime-exit-window 10 --friday-exit 20

echo [29/32] Stress - Window 20
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --regime-exit-window 20

echo [30/32] Stress - Window 20 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --regime-exit-window 20 --friday-exit 20

echo [31/32] Stress - Window 30
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --regime-exit-window 30

echo [32/32] Stress - Window 30 + Friday exit
python regime_simulator.py --data-dir "%DATA%" --slots-csv "%SLOTS%" --output-dir "%OUT%" --slippage 3.0 --entry-delay 2 --regime-exit-window 30 --friday-exit 20

echo ============================================
echo  ALL 32 RUNS COMPLETE
echo ============================================
pause
