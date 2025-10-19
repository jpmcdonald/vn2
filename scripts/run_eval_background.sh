#!/bin/bash
# Run model evaluation in background with resource control and logging

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Activate virtual environment directly
if [ -d "./V2env" ]; then
    source ./V2env/bin/activate
    export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"
elif [ -f "./activate.sh" ]; then
    source ./activate.sh
else
    echo "❌ Error: No virtual environment found"
    exit 1
fi

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="${PROJECT_ROOT}/logs/eval_${TIMESTAMP}.log"
PIDFILE="${PROJECT_ROOT}/logs/eval.pid"

# Create logs directory
mkdir -p "${PROJECT_ROOT}/logs"

# Set BLAS threads to avoid oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "🚀 Starting model evaluation in background..."
echo "📝 Logging to: $LOGFILE"
echo "🔢 PID file: $PIDFILE"

# Run evaluation in background
nohup python -m vn2.analyze.model_eval "$@" > "$LOGFILE" 2>&1 &

# Save PID
echo $! > "$PIDFILE"

echo "✅ Process launched!"
echo "   PID: $(cat $PIDFILE)"
echo ""
echo "📡 Monitor with:"
echo "   tail -f $LOGFILE"
echo "   cat models/results/eval_progress.json | jq '.'"
echo "   ps -p $(cat $PIDFILE)"
echo ""
echo "🛑 Stop with:"
echo "   kill $(cat $PIDFILE)"
echo ""

