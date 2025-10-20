# SIP Evaluation: Launch Command

## ✅ System Ready

All prerequisites verified:
- ✅ Capped demand data (692K)
- ✅ Initial state data (8.0K)
- ✅ 3 trained models (21,564 total checkpoints)
- ✅ SIP module installed and importable
- ✅ 12 CPU cores available
- ✅ 980 GB disk space available

## Launch Command

```bash
source activate.sh

python -m vn2.cli eval-models \
  --use-sip-optimization \
  --sip-grain 1000 \
  --out-suffix v4 \
  --holdout 8 \
  --n-jobs 12 \
  --batch-size 2000 \
  --resume
```

## What Will Happen

1. **Load data:**
   - Demand: `data/processed/demand_imputed_capped.parquet`
   - State: `data/interim/state.parquet`
   - Master: `data/processed/master.parquet`

2. **Generate tasks:**
   - 599 SKUs × 8 folds × 3 models = **14,376 tasks**

3. **Parallel execution:**
   - 12 workers processing batches of 2,000 tasks
   - Each task: ~0.5-2 seconds
   - Estimated total time: **2-4 hours**

4. **Outputs:**
   - `models/results/eval_folds_v4.parquet` (per-fold results)
   - `models/results/eval_agg_v4.parquet` (per-SKU aggregated)
   - `models/results/leaderboards_v4.parquet` (overall ranking)
   - `models/results/eval_progress_v4.json` (checkpoint)

## Monitor Progress

### Terminal 1: Run evaluation
```bash
source activate.sh
python -m vn2.cli eval-models --use-sip-optimization --out-suffix v4 --n-jobs 12 --resume
```

### Terminal 2: Watch progress
```bash
# Count completed tasks
watch -n 10 'cat models/results/eval_progress_v4.json | jq ".completed | length"'

# Or check batch files
watch -n 10 'ls -lh models/results/eval_folds_v4_part-*.parquet | wc -l'
```

### Terminal 3: Monitor resources
```bash
top -o cpu
```

## Expected Output

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

============================================================
📦 Batch 1: tasks 1-2000 of 14376
============================================================
[Parallel(n_jobs=12)]: Using backend LokyBackend with 12 concurrent workers.
[Parallel(n_jobs=12)]: Done  26 tasks      | elapsed:   30.0s
[Parallel(n_jobs=12)]: Done 176 tasks      | elapsed:  2.5min
...
```

## When Complete

Leaderboard will auto-print:

```
============================================================
🏆 Overall Leaderboard (by SIP realized cost (week 2))
============================================================
              model_name  sip_realized_cost_w2  sip_service_level_w2  ...
 slurp_stockout_aware                  12345.6                  0.87  ...
       slurp_bootstrap                  13456.7                  0.85  ...
       seasonal_naive                   15678.9                  0.82  ...
```

## Interrupt and Resume

If you need to stop:
1. Press `Ctrl+C` (graceful shutdown after current batch)
2. Progress saved to `eval_progress_v4.json`
3. Resume with same command (--resume flag)

## View Results

```bash
# Leaderboard
python -c "import pandas as pd; df = pd.read_parquet('models/results/leaderboards_v4.parquet'); print(df.to_string())"

# Per-SKU details
python -c "import pandas as pd; df = pd.read_parquet('models/results/eval_agg_v4.parquet'); print(df.head(20).to_string())"

# Per-fold details
python -c "import pandas as pd; df = pd.read_parquet('models/results/eval_folds_v4.parquet'); print(df.head(20).to_string())"
```

## Troubleshooting

### High memory usage
```bash
# Reduce workers
--n-jobs 6

# Reduce batch size
--batch-size 1000
```

### Slow progress
```bash
# Check if tasks are actually running
ps aux | grep python

# Check CPU usage
top -o cpu
```

### Errors in output
```bash
# Check logs for specific errors
grep -i "error\|warning\|failed" models/results/eval_folds_v4_part-*.parquet
```

## Next Steps

1. **Review leaderboard** → Identify champion model
2. **Analyze per-SKU** → Where does stockout-aware help most?
3. **Compare with v3** → How much cost reduction?
4. **Prepare submission** → Use champion for final forecast

---

**Ready to launch!** 🚀

Run: `./check_sip_ready.sh` to verify again, then execute the launch command above.

