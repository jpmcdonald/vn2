# Model Evaluation System - Ready to Run

## Status: ✅ Implementation Complete

The comprehensive model evaluation system is now ready to run. All code has been implemented, tested, and committed.

## What Was Built

### Core Evaluator (`src/vn2/analyze/model_eval.py`)
- Evaluates all 12 trained models over 8-fold rolling-origin holdout
- Computes 30+ metrics per (SKU, model, fold):
  - **Point metrics**: MAE, MAPE, MASE, Bias, RMSE
  - **Distribution metrics**: Pinball loss, CRPS, coverage/width at 80/90/95%
  - **Shape diagnostics**: PI width, curvature, moments (mean, std, skew, kurtosis)
  - **Cost/value**: Expected cost, std, quantiles under base-stock policy

### Resource Management
- **CPU control**: `--cpu-fraction 0.5` (half cores) or `--cpu-fraction 1.0` (full cores)
- **Explicit cores**: `--n-jobs 6` (overrides fraction)
- **Memory management**: `--batch-size 2000` (adjustable)
- **Simulation samples**: `--n-sims 500` (configurable)

### Checkpointing & Resume
- Progress saved after each batch in `models/results/eval_progress.json`
- Atomic batch writes: `eval_folds_part-*.parquet`
- Safe interruption: Ctrl+C or `kill <PID>` → finishes current batch
- Resume: Just add `--resume` flag to continue

### Background Execution
- Script: `scripts/run_eval_background.sh`
- Logs: `logs/eval_<timestamp>.log`
- PID tracking: `logs/eval.pid`
- Easy monitoring and control

## How to Run

### Option 1: Foreground (Interactive)

```bash
cd /Users/jpmcdonald/Code/vn2
source activate.sh

# Start with half cores (recommended for daytime)
./go eval-models \
  --holdout 8 \
  --cpu-fraction 0.5 \
  --n-sims 500 \
  --batch-size 2000 \
  --resume
```

### Option 2: Background (Recommended)

```bash
cd /Users/jpmcdonald/Code/vn2

# Half cores today (leaves resources for other work)
./scripts/run_eval_background.sh \
  --holdout 8 \
  --cpu-fraction 0.5 \
  --n-sims 500 \
  --batch-size 2000 \
  --resume

# Monitor progress
tail -f logs/eval_*.log
cat models/results/eval_progress.json | jq '.'

# Check if still running
ps -p $(cat logs/eval.pid)

# Stop if needed (graceful)
kill $(cat logs/eval.pid)
```

### Option 3: Full Cores Tonight

```bash
# Before bed, restart with full resources
kill $(cat logs/eval.pid)  # Stop current run

./scripts/run_eval_background.sh \
  --holdout 8 \
  --cpu-fraction 1.0 \
  --n-sims 500 \
  --batch-size 2000 \
  --resume
```

## Expected Runtime

**Total work:** 599 SKUs × 8 folds × 12 models = **57,504 evaluations**

**Timing estimates:**
- **Half cores (6 workers)**: ~2-3 hours
- **Full cores (12 workers)**: ~1.5-2 hours

**Memory usage:** ~2-4 GB with default batch size

## Monitoring

### Watch the log
```bash
tail -f logs/eval_*.log
```

### Check progress
```bash
cat models/results/eval_progress.json | jq '. | {completed: (.completed | length)}'
```

### See partial results
```bash
ls -lh models/results/eval_folds_part-*.parquet
```

### Process status
```bash
ps -p $(cat logs/eval.pid) -o pid,etime,%cpu,%mem,command
```

## Outputs

When complete, you'll have:

1. **Per-fold results**: `models/results/eval_folds.parquet`
   - Every (SKU, model, fold) evaluation
   - All 30+ metrics
   - ~57K rows

2. **Aggregated results**: `models/results/eval_agg.parquet`
   - Per (SKU, model) averages across 8 folds
   - ~7,188 rows (599 SKUs × 12 models)

3. **Overall leaderboard**: `models/results/leaderboards.parquet`
   - Overall model rankings
   - Sorted by expected_cost
   - 12 rows (one per model)

## What Happens During Evaluation

For each (SKU, model, fold):

1. **Load checkpoint** from `models/checkpoints/<model>/<store>_<product>/fold_<k>.pkl`
2. **Reconstruct test data** for that fold using `prepare_train_test_split()`
3. **Compute point metrics** from median forecast
4. **Compute distribution metrics** (pinball, CRPS, coverage/width)
5. **Sample from density** to estimate shape (moments, curvature)
6. **Simulate ordering decision**:
   - Estimate μ, σ from h=1 forecast
   - Compute base-stock level S = μ×3 + z×σ×√3
   - Place order q = max(0, S - inventory_position)
   - Simulate 2 weeks with 500 demand scenarios
   - Measure holding and shortage costs
7. **Save metrics** to batch file

## Cost Function Details

**Inventory policy:** Base-stock (order-up-to)
- Critical fractile: 0.8333 (from holding=0.2, shortage=1.0)
- Lead + review: 3 weeks
- Safety stock: z × σ × √3 where z = Φ⁻¹(0.8333) ≈ 0.97

**Simulation:**
- Initial state: zero inventory (fair comparison)
- Horizon: 2 weeks (h=1, h=2)
- Recourse: zero (no orders after week 0)
- Samples: 500 demand scenarios per fold

**Costs:**
- Holding: €0.20 per unit per week
- Shortage: €1.00 per unit of lost sales

## Troubleshooting

### If it runs out of memory
```bash
# Reduce batch size and/or simulations
./scripts/run_eval_background.sh \
  --batch-size 1000 \
  --n-sims 200 \
  --cpu-fraction 0.5 \
  --resume
```

### If it's too slow
```bash
# Use more cores (if available)
./scripts/run_eval_background.sh \
  --cpu-fraction 1.0 \
  --resume
```

### If you need to stop and restart
```bash
# Stop (graceful - finishes current batch)
kill $(cat logs/eval.pid)

# Or force stop
kill -9 $(cat logs/eval.pid)

# Restart later - it will resume automatically
./scripts/run_eval_background.sh --resume ...
```

## After Completion

Once evaluation finishes, you can:

1. **View the leaderboard**:
```bash
python -c "import pandas as pd; df = pd.read_parquet('models/results/leaderboards.parquet'); print(df.sort_values('expected_cost')[['model_name', 'expected_cost', 'mae', 'coverage_90']].to_string())"
```

2. **Aggregate only** (if you want to recompute rankings):
```bash
./go eval-models --aggregate
```

3. **Analyze results** in detail (next step - we can build visualizations and reports)

## Ready to Start?

Just run:
```bash
cd /Users/jpmcdonald/Code/vn2
./scripts/run_eval_background.sh \
  --holdout 8 \
  --cpu-fraction 0.5 \
  --n-sims 500 \
  --batch-size 2000 \
  --resume
```

Then monitor with:
```bash
tail -f logs/eval_*.log
```

The system will:
- ✅ Use half your cores (leaving resources for other work)
- ✅ Save progress after every batch
- ✅ Be safely interruptible
- ✅ Resume automatically if restarted
- ✅ Complete in ~2-3 hours

**You're all set!** 🚀

