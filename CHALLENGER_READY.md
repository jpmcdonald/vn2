# Challenger Benchmark Implementation - READY

## ✅ Implementation Complete

### Data Preparation
- ✅ **Winsorized Data Created**: `data/processed/demand_imputed_winsor.parquet`
  - Transform-space winsorization (μ + 3σ per SKU using SURD transforms)
  - All 10,517 imputed values capped (100%)
  - Max reduced from 228B to 500 units
  - Preserves non-imputed observations exactly

### Models Enabled (9 total)

#### SLURP Family (3) - Train on Raw Data
1. **slurp_bootstrap** - Baseline conditional bootstrap
2. **slurp_surd** - SURD transforms, no stockout handling
3. **slurp_stockout_aware** - Censoring-aware, no SURD

Note: `slurp_surd_stockout_aware` already trained, disabled for this run

#### Challenger Models (6) - Train on Winsorized Data
4. **lightgbm_quantile** - Quantile regression (density)
5. **lightgbm_point** - MSE objective (point forecast)
6. **qrf** - Quantile Random Forest (density)
7. **ets** - Exponential Smoothing (density via PIs)
8. **linear_quantile** - Linear quantile regression (density)
9. **ngboost** - NGBoost LogNormal (density)

### GLM Models (Deferred)
- **glm_poisson** and **glm_negbin** implemented but disabled
- Require pipeline debugging (X_future handling)
- Can be enabled later if needed

## Training Configuration

### Data Strategy
```yaml
SLURP models:  data/processed/demand_long.parquet (raw, with in_stock flag)
Other models:  data/processed/demand_imputed_winsor.parquet (stable, capped)
```

### Compute Settings
```yaml
n_jobs: 11
timeout_per_fit: 90s
horizon: 2 weeks
holdout_weeks: 12
rolling_origins: 12
```

### Expected Training Time
- **SLURP models**: ~2-3 hours (3 models × 599 SKUs × 12 folds = 21,564 fits)
- **Challengers**: ~3-4 hours (6 models × 599 SKUs × 12 folds = 43,128 fits)
- **Total**: ~5-7 hours for all 9 models

## Launch Commands

### Full Training (All SKUs, All Models)
```bash
cd /Users/jpmcdonald/Code/vn2
source V2env/bin/activate
export PYTHONPATH=/Users/jpmcdonald/Code/vn2/src:$PYTHONPATH

# Clear old checkpoints for models being retrained
rm -f models/checkpoints/progress.json
rm -rf models/checkpoints/slurp_bootstrap
rm -rf models/checkpoints/slurp_surd
rm -rf models/checkpoints/slurp_stockout_aware
rm -rf models/checkpoints/lightgbm_quantile
rm -rf models/checkpoints/ets
rm -rf models/checkpoints/linear_quantile
rm -rf models/checkpoints/ngboost

# Launch training
nohup python -m vn2.cli forecast --n-jobs 11 > logs/train_challengers_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > logs/train.pid

# Monitor
tail -f logs/train_challengers_*.log
```

### Test Run (1 SKU)
```bash
python -m vn2.cli forecast --test --n-jobs 1
```

### Monitor Progress
```bash
# Check process
ps -p $(cat logs/train.pid)

# Watch progress
watch -n 10 'tail -20 logs/train_challengers_*.log'

# Count completed checkpoints
find models/checkpoints -name "*.pkl" | wc -l
```

## Evaluation Plan

After training completes, evaluate with SIP optimization:

```bash
# Evaluate all models
python -m vn2.cli eval-models \
  --holdout 12 \
  --n-sims 1000 \
  --cpu-fraction 1.0 \
  --n-jobs 11 \
  --batch-size 100 \
  --use-sip-optimization \
  --sip-grain 1000 \
  --out-suffix _v4_comp

# Results will be in:
# - models/results/eval_folds_v4_comp.parquet
# - models/results/eval_agg_v4_comp.parquet
# - models/results/eval_leaderboard_v4_comp.parquet
```

## What We'll Learn

1. **Jensen Effect**: Compare density-aware SIP vs point+service-level policies
2. **Stockout Awareness**: SLURP stockout-aware vs baseline on raw data
3. **SURD Value**: SLURP SURD vs non-SURD variants
4. **Challenger Parity**: Which standard models (LightGBM, ETS, NGBoost) compete with SLURP
5. **Point vs Density**: LightGBM Point (MSE) vs LightGBM Quantile

## Files Created

- `scripts/create_winsorized_data.py` - Winsorization script
- `src/vn2/forecast/models/lightgbm_point.py` - Point forecast baseline
- `src/vn2/forecast/models/qrf.py` - Quantile Random Forest
- `src/vn2/forecast/models/glm_count.py` - GLM Poisson/NegBin (deferred)
- `data/processed/demand_imputed_winsor.parquet` - Winsorized data

## Status: READY TO TRAIN 🚀

All components implemented, tested, and committed. Ready for full training run.

