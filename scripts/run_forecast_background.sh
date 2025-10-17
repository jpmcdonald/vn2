#!/usr/bin/env zsh
# Background launcher for forecast training with monitoring

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Create logs directory
mkdir -p logs

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/forecast_${TIMESTAMP}.log"
PIDFILE="logs/forecast.pid"
PROGRESS_FILE="models/checkpoints/progress.json"

# Mode: pilot or full
MODE="${1:-pilot}"
N_JOBS="${2:-11}"

if [ "$MODE" = "pilot" ]; then
    PILOT_FLAG="--pilot"
    echo "🧪 Starting PILOT forecast run..."
else
    PILOT_FLAG=""
    echo "🚀 Starting FULL forecast run..."
fi

# Activate environment and run in background
source activate.sh

echo "📝 Logging to: $LOGFILE"
echo "🔢 PID file: $PIDFILE"
echo "📊 Progress: $PROGRESS_FILE"
echo ""

# Launch in background with nohup
nohup ./go forecast --config configs/forecast.yaml $PILOT_FLAG --n-jobs $N_JOBS \
    > "$LOGFILE" 2>&1 &

# Save PID
PID=$!
echo $PID > "$PIDFILE"

echo "✅ Process launched!"
echo "   PID: $PID"
echo ""
echo "📡 Monitor with:"
echo "   tail -f $LOGFILE"
echo "   cat $PROGRESS_FILE | jq '.'"
echo "   ps -p $PID"
echo ""
echo "🛑 Stop with:"
echo "   kill $PID"
echo ""

