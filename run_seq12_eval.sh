#!/bin/bash
# Run 12-week sequential backtest evaluation
# Usage: ./run_seq12_eval.sh

set -e

echo "========================================"
echo "12-Week Sequential Backtest Evaluation"
echo "========================================"
echo ""
echo "Configuration:"
echo "  - Models: All available (15 models)"
echo "  - SKUs: All available (~585 SKUs)"
echo "  - Weeks: 12"
echo "  - Workers: 12"
echo "  - PMF grain: 500"
echo "  - Costs: cu=1.0, co=0.2"
echo ""

# Activate environment
source activate.sh

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG="seq12_${TIMESTAMP}"

echo "Run tag: $RUN_TAG"
echo ""
echo "Starting evaluation at $(date)..."
echo ""

# Run evaluation
python -m vn2.cli sequential-eval \
  --checkpoints models/checkpoints \
  --demand data/processed/demand_long.parquet \
  --state data/interim/state.parquet \
  --out-dir models/results \
  --run-tag "$RUN_TAG" \
  --n-jobs 12 \
  --cu 1.0 \
  --co 0.2 \
  --sip-grain 500 \
  --holdout 12

echo ""
echo "========================================"
echo "Evaluation complete at $(date)"
echo "========================================"
echo ""
echo "Output files:"
echo "  - models/results/sequential_results_${RUN_TAG}.parquet"
echo "  - models/results/model_totals_${RUN_TAG}.parquet"
echo "  - models/results/selector_map_${RUN_TAG}.parquet"
echo "  - models/results/leaderboard_${RUN_TAG}.md"
echo ""
echo "To view leaderboard:"
echo "  cat models/results/leaderboard_${RUN_TAG}.md"
echo ""

