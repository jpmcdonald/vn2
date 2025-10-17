#!/usr/bin/env zsh
# Monitor forecast training progress

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROGRESS_FILE="models/checkpoints/progress.json"
PIDFILE="logs/forecast.pid"

# Check if process is running
if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Process running (PID: $PID)"
    else
        echo "❌ Process not running (PID file exists but process dead)"
        rm -f "$PIDFILE"
        exit 1
    fi
else
    echo "⚠️  No PID file found. Process may not be running."
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "FORECAST TRAINING PROGRESS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Parse progress file
if [ -f "$PROGRESS_FILE" ]; then
    # Check if jq is available
    if command -v jq &> /dev/null; then
        COMPLETED=$(jq '.completed | length' "$PROGRESS_FILE")
        FAILED=$(jq '.failed | length' "$PROGRESS_FILE")
        echo "📊 Tasks completed: $COMPLETED"
        echo "❌ Tasks failed: $FAILED"
        echo ""
        echo "Progress detail:"
        jq '.' "$PROGRESS_FILE"
    else
        echo "Progress file contents:"
        cat "$PROGRESS_FILE"
    fi
else
    echo "⚠️  No progress file found yet"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Show recent log
LATEST_LOG=$(ls -t logs/forecast_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "📝 Latest log (last 20 lines): $LATEST_LOG"
    echo ""
    tail -20 "$LATEST_LOG"
else
    echo "⚠️  No log files found"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Commands:"
echo "  Follow log:    tail -f $LATEST_LOG"
echo "  Kill process:  kill $PID"
echo "  Check status:  ps -p $PID"
echo ""

