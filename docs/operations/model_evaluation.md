# Model Evaluation System

## Overview

Comprehensive evaluation system for the 12 trained forecast models using:
- Standard metrics (MAE, MAPE, MASE, Bias, pinball, CRPS, coverage/width)
- Density shape diagnostics (PI width, curvature, moments)
- Cost-based value metrics (expected cost under inventory policy)

## Features

### Parallelism & Resource Control
- Configurable CPU allocation via `--cpu-fraction` (0-1) or `--n-jobs`
- Process-based parallelism with `joblib.Parallel(backend='loky')`
- BLAS thread control to avoid oversubscription
- Batched execution to manage memory

### Checkpointing & Resume
- Progress tracked in `models/results/eval_progress.json`
- Per-batch atomic writes: `eval_folds_part-*.parquet`
- Safe interruption via SIGINT/SIGTERM
- Resume with `--resume` flag

### Background Execution
- Script: `scripts/run_eval_background.sh`
- Logs: `logs/eval_<timestamp>.log`
- PID file: `logs/eval.pid`
- Monitoring: `tail -f logs/eval_*.log`

## Usage

### Foreground (interactive)
```bash
cd /Users/jpmcdonald/Code/vn2
source activate.sh

# Half cores (daytime)
./go eval-models \
  --holdout 8 \
  --cpu-fraction 0.5 \
  --n-sims 500 \
  --batch-size 2000 \
  --resume

# Full cores (overnight)
./go eval-models \
  --holdout 8 \
  --cpu-fraction 1.0 \
  --n-sims 500 \
  --batch-size 2000 \
  --resume
```

### Background (unattended)
```bash
# Half cores today
./scripts/run_eval_background.sh \
  --holdout 8 \
  --cpu-fraction 0.5 \
  --n-sims 500 \
  --batch-size 2000 \
  --resume

# Full cores tonight
./scripts/run_eval_background.sh \
  --holdout 8 \
  --cpu-fraction 1.0 \
  --n-sims 500 \
  --batch-size 2000 \
  --resume

# Monitor
tail -f logs/eval_*.log
cat models/results/eval_progress.json | jq '.'
ps -p $(cat logs/eval.pid)

# Stop if needed
kill $(cat logs/eval.pid)
```

### Aggregation Only
```bash
./go eval-models --aggregate
```

## Evaluation Methodology

### Rolling-Origin Holdout
- 8 folds (configurable via `--holdout`)
- Each fold: train on history up to fold origin, test on h=1,2
- Reconstruct train/test splits using `prepare_train_test_split()`

### Order Decision Policy
At each fold origin:
1. Estimate μ and σ from h=1 forecast distribution (SIP sampling)
2. Compute base-stock level: S = μ×(L+R) + z×σ×√(L+R)
   - Critical fractile = shortage/(holding+shortage) = 0.8333
   - L=2 (lead weeks), R=1 (review weeks)
3. Order quantity: q = max(0, S − inventory_position)
4. Simulate weeks t+1, t+2 with zero recourse
5. Measure holding and shortage costs

### Metrics Computed

**Point forecast (from median):**
- MAE: Mean Absolute Error
- MAPE: Mean Absolute Percentage Error (ε=1)
- MASE: Mean Absolute Scaled Error (vs seasonal naive L=52)
- Bias: Mean error
- RMSE: Root Mean Squared Error

**Distribution quality:**
- Pinball loss: Average across all quantiles
- CRPS: Continuous Ranked Probability Score
- Coverage: Actual vs nominal at 80/90/95%
- Width: Average PI width at 80/90/95%

**Density shape (h=1,2):**
- Width at 80/90/95% coverage
- Curvature: ratio width(h=2)/width(h=1)
- Moments: mean, std, skewness, kurtosis (via SIP sampling)

**Cost/value:**
- Expected cost: Mean over simulations
- Cost std: Standard deviation
- Cost q05/q95: 5th and 95th percentiles
- Order quantity and base-stock level

### Aggregation & Ranking
- Aggregate per (SKU, model) across 8 folds: mean of metrics
- Primary ranking: expected_cost (lower is better)
- Tie-breakers: coverage closeness, narrower width, lower pinball/CRPS, MAE/MASE

## Outputs

### Per-Fold Results
- `models/results/eval_folds_part-*.parquet` - Batch outputs
- `models/results/eval_folds.parquet` - Consolidated

Columns: model_name, store, product, fold_idx, horizon, all metrics

### Aggregated Results
- `models/results/eval_agg.parquet` - Per (SKU, model) aggregates

Columns: store, product, model_name, mean of each metric, n_folds

### Leaderboards
- `models/results/leaderboards.parquet` - Overall rankings

Columns: model_name, overall mean of each metric, sorted by expected_cost

### Progress
- `models/results/eval_progress.json` - Checkpoint state

Format: `{"completed": ["model_store_product_fold", ...]}`

## Runtime Estimates

**Configuration:** 599 SKUs × 8 folds × 12 models = 57,504 tasks

**Phase 1 (base-stock, n_sims=500):**
- Half cores (6 workers): ~2-3 hours
- Full cores (12 workers): ~1.5-2 hours

**Memory:** ~2-4 GB with batch_size=2000

## Cost Function Details

**Costs (from `configs/base.yaml`):**
- Holding: €0.20 per unit per week
- Shortage: €1.00 per unit of lost sales

**Simulator:**
- Lead time: 2 weeks (order at end of W, receive start of W+3)
- Review period: 1 week
- State: on_hand, intransit_1, intransit_2
- Dynamics: receive → sell → update → order

**Initial state for evaluation:**
- Zero inventory (fair comparison across models)
- Can be customized via `data/interim/state.parquet` if needed

## Troubleshooting

### Out of Memory
- Reduce `--batch-size` (try 1000 or 500)
- Reduce `--n-sims` (try 200 or 100)
- Reduce `--n-jobs`

### Slow Progress
- Increase `--n-jobs` (up to total cores)
- Reduce `--n-sims` for faster iteration
- Check CPU usage: `top` or `htop`

### Interrupted Run
- Simply restart with `--resume` flag
- Progress is saved after each batch
- Completed tasks are skipped automatically

### Missing Checkpoints
- Verify models were trained: `ls models/checkpoints/`
- Check specific model: `ls models/checkpoints/<model_name>/`
- Re-run forecast training if needed

## Next Steps

After evaluation completes:
1. Review leaderboards: `cat models/results/leaderboards.parquet`
2. Analyze per-SKU winners
3. Compare cost vs accuracy trade-offs
4. Examine density shape diagnostics
5. Select top 2-3 models for production
6. Optional: Phase 2 refinement with MC grid optimization

