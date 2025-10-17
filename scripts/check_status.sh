#!/bin/bash
# Quick status check for overnight forecast run

echo "═══════════════════════════════════════════════════════════"
echo "  OVERNIGHT FORECAST RUN - STATUS CHECK"
echo "═══════════════════════════════════════════════════════════"

# Check if job is running
PID_FILE="logs/forecast.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Job is RUNNING (PID: $PID)"
    else
        echo "⏹️  Job COMPLETED or STOPPED (PID: $PID)"
    fi
else
    echo "❓ No PID file found"
fi

echo ""

# Check progress
if [ -f "models/checkpoints/progress.json" ]; then
    COMPLETED=$(python3 -c "import json; d=json.load(open('models/checkpoints/progress.json')); print(len(d['completed']))")
    PERCENT=$(python3 -c "print(f'{$COMPLETED/646.92:.1f}')")
    echo "📊 Progress: $COMPLETED / 64,692 tasks ($PERCENT%)"
else
    echo "❓ No progress file found"
fi

echo ""

# Check results
if [ -f "models/results/training_results.parquet" ]; then
    echo "📈 Results Summary:"
    python3 -c "
import pandas as pd
df = pd.read_parquet('models/results/training_results.parquet')
print(f'  Total: {len(df):,}')
print(f'  Success: {(df[\"status\"] == \"success\").sum():,}')
print(f'  Failed: {(df[\"status\"] == \"failed\").sum():,}')
print(f'  Timeout: {(df[\"status\"] == \"timeout\").sum():,}')
print(f'  Success Rate: {(df[\"status\"] == \"success\").mean()*100:.1f}%')
"
else
    echo "❓ No results file found yet"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "📡 Monitor logs: tail -f logs/forecast_20251017_024353.log"
echo "🔍 Full details: cat OVERNIGHT_RUN_STATUS.md"
echo ""

