#!/bin/bash
# Quick commands for training and evaluating SURD-aware SLURP Bootstrap

echo "============================================================"
echo "SURD-Aware SLURP Bootstrap Training & Evaluation"
echo "============================================================"
echo ""

# Activate environment
source activate.sh

# Option 1: Smoke test (1 SKU, ~30 seconds)
echo "Option 1: Smoke Test (1 SKU)"
echo "  python -m vn2.cli forecast --test --n-jobs 4"
echo ""

# Option 2: Full training (599 SKUs, ~2-4 hours)
echo "Option 2: Full Training (599 SKUs)"
echo "  python -m vn2.cli forecast --n-jobs 12 --resume"
echo ""

# Option 3: Check progress
echo "Option 3: Check Training Progress"
echo "  cat models/checkpoints/progress.json | jq '.completed | length'"
echo ""

# Option 4: Run SIP evaluation after training
echo "Option 4: Run SIP Evaluation (after training complete)"
echo "  python -m vn2.cli eval-models \\"
echo "    --use-sip-optimization \\"
echo "    --sip-grain 1000 \\"
echo "    --out-suffix v4 \\"
echo "    --holdout 8 \\"
echo "    --n-jobs 12 \\"
echo "    --batch-size 2000 \\"
echo "    --resume"
echo ""

# Option 5: View results
echo "Option 5: View Leaderboard (after evaluation)"
echo "  python -c \"import pandas as pd; df = pd.read_parquet('models/results/leaderboards_v4.parquet'); print(df.to_string())\""
echo ""

echo "============================================================"
echo "Current Configuration:"
echo "  Model: slurp_surd_stockout_aware"
echo "  Status: ENABLED in configs/forecast.yaml"
echo "  SURD Transforms: data/processed/surd_transforms.parquet (599 SKUs)"
echo "  Training Data: data/processed/demand_imputed_capped.parquet"
echo "============================================================"

