# SIP Evaluation Runbook

## Quick Start

### 1. Verify Prerequisites

```bash
# Check that state data exists
ls -lh data/interim/state.parquet

# Check that capped demand exists
ls -lh data/processed/demand_imputed_capped.parquet

# Check trained models
ls models/checkpoints/
# Should see: slurp_stockout_aware, slurp_bootstrap, seasonal_naive
```

### 2. Run SIP Evaluation (v4)

```bash
# Activate environment
source activate.sh

# Run with full resources (12 cores)
python -m vn2.cli eval-models \
  --use-sip-optimization \
  --sip-grain 1000 \
  --out-suffix v4 \
  --holdout 8 \
  --n-jobs 12 \
  --batch-size 2000 \
  --resume
```

**Expected output:**
```
📊 Evaluating forecast models...
   Holdout: 8 folds
   Workers: 12
   Batch size: 2000
   Simulations: 500
   SIP Optimization: ENABLED
   SIP grain: 1000
📊 Loading data...
📦 Loading initial state from data/interim/state.parquet...
🤖 Found 3 models: ['slurp_stockout_aware', 'slurp_bootstrap', 'seasonal_naive']
📋 Generating tasks (holdout=8 folds)...
✅ Total tasks: 14376
✅ Already completed: 0
⏳ To run: 14376
```

### 3. Monitor Progress

```bash
# Check progress file
cat models/results/eval_progress_v4.json | jq '.completed | length'

# Watch CPU usage
top -o cpu

# Check partial results
ls -lh models/results/eval_folds_v4_part-*.parquet
```

### 4. View Results

Once complete, leaderboard will auto-generate:

```bash
# View leaderboard
python -c "import pandas as pd; df = pd.read_parquet('models/results/leaderboards_v4.parquet'); print(df.to_string())"
```

**Expected columns:**
- `model_name`
- `sip_realized_cost_w2` (total cost, lower is better)
- `sip_service_level_w2` (average, higher is better)
- `sip_fill_rate_w2` (average, higher is better)
- `mae`, `pinball_cf_h1`, `coverage_90` (accuracy/density metrics)

## Resource Configurations

### Full Resources (Overnight)

```bash
python -m vn2.cli eval-models \
  --use-sip-optimization \
  --out-suffix v4 \
  --n-jobs 12 \
  --cpu-fraction 1.0 \
  --batch-size 2000 \
  --resume
```

### Half Resources (Daytime)

```bash
python -m vn2.cli eval-models \
  --use-sip-optimization \
  --out-suffix v4 \
  --n-jobs 6 \
  --cpu-fraction 0.5 \
  --batch-size 1000 \
  --resume
```

### Test Run (1 SKU)

```bash
# Not yet implemented - would need to filter tasks
# For now, run full evaluation and interrupt after first batch
```

## Troubleshooting

### Issue: State file not found

```bash
# Check if state.parquet exists
ls data/interim/state.parquet

# If missing, you may need to create it from raw data
# (Implementation TBD - for now, SIP will fall back to zero state)
```

### Issue: Out of memory

```bash
# Reduce batch size
--batch-size 500

# Reduce workers
--n-jobs 4
```

### Issue: Progress file corrupted

```bash
# Clear progress and restart
rm models/results/eval_progress_v4.json
# Re-run with --resume
```

### Issue: Results look wrong

```bash
# Run unit tests
python test_sip_opt.py

# Check for degenerate forecasts (all-zero quantiles)
# These are automatically skipped
```

## Expected Timing

- **Single task:** ~0.5-2 seconds
- **Batch of 2000:** ~30-60 minutes (12 cores)
- **Full evaluation:** ~2-4 hours (599 SKUs × 8 folds × 3 models = 14,376 tasks)

## Output Files

### Per-Fold Results
`models/results/eval_folds_v4.parquet`
- One row per (model, SKU, fold)
- All metrics: accuracy, density, SIP costs, service levels

### Aggregated Results
`models/results/eval_agg_v4.parquet`
- One row per (model, SKU)
- Costs summed across folds
- Accuracy/density averaged across folds

### Leaderboard
`models/results/leaderboards_v4.parquet`
- One row per model
- Ranked by total `sip_realized_cost_w2`
- Includes overall performance metrics

## Next Steps

After evaluation completes:

1. **Review leaderboard:** Identify champion model
2. **Analyze per-SKU:** Which SKUs benefit from stockout-aware forecasting?
3. **Compare with baseline:** How much cost reduction vs. v3?
4. **Prepare submission:** Use champion model for final forecast

## Commands Summary

```bash
# Full evaluation
python -m vn2.cli eval-models --use-sip-optimization --out-suffix v4 --n-jobs 12 --resume

# Monitor
watch -n 10 'cat models/results/eval_progress_v4.json | jq ".completed | length"'

# View results
python -c "import pandas as pd; df = pd.read_parquet('models/results/leaderboards_v4.parquet'); print(df[['model_name', 'sip_realized_cost_w2', 'sip_service_level_w2', 'mae']].to_string())"

# Aggregate only (if needed)
python -m vn2.cli eval-models --aggregate --out-suffix v4
```

