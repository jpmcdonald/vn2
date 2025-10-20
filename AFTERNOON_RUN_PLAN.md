# Afternoon Run Plan: Stockout-Aware SLURP + Clean Data

## Status Summary

### Data Quality Issues Discovered
- **11.2% stockout rate** in historical data (10,517 / 94,043 observations)
- **Imputation created garbage**: 50% of imputed values >1,000 units (vs observed max 494)
- **Worst outliers**: 47 imputed values >1M units (max: 228 BILLION!)

### Data Cleaned
✅ Created `demand_imputed_capped.parquet`:
- Caps imputed values at `min(500, max(50, observed_max × 10))` per SKU
- Preserves stockout signal without absurd outliers
- **Max demand: 500 units**
- **99th percentile: 180 units**
- **Max 2-week consecutive: 1,000 units**

### Stockout-Aware SLURP Implemented
✅ New model: `slurp_stockout_aware`
- Treats stockouts as **censored observations** (preserves uncertainty)
- For stockout periods: samples from positive observed demand distribution
- Maintains stockout rate in bootstrap to capture uncertainty properly
- **Does NOT collapse uncertainty into a point estimate**

## Configuration for This Afternoon

### Models to Train (in forecast.yaml)
```yaml
slurp_stockout_aware:
  enabled: true
  n_neighbors: 50
  n_bootstrap: 1000
  stockout_aware: true

# All others disabled
```

### Data Source
Use `demand_imputed_capped.parquet` (clean data with reasonable imputed values)

### Training Command
```bash
cd /Users/jpmcdonald/Code/vn2
source activate.sh

# Train stockout-aware SLURP only
python -m vn2 forecast \
  --config configs/forecast.yaml \
  --n-jobs 12

# Expected runtime: ~30-45 minutes for 599 SKUs × 12 folds
```

### Checkpoint & Resume
- ✅ Checkpoints saved to `models/checkpoints/slurp_stockout_aware/`
- ✅ Progress tracked in `models/checkpoints/progress.json`
- ✅ Can resume with same command if interrupted

## Champion & Challenger Models

### 🏆 Champion: SLURP Stockout-Aware (NEW)
- **Pinball@CF**: TBD (expected best)
- **Calibration**: TBD
- **Coverage@90**: TBD (expected ~97%)
- **Handles stockout uncertainty properly**

### 🥈 Challenger #1: SLURP Bootstrap (existing)
- **Pinball@CF**: 20.38
- **Calibration**: 0.244
- **Coverage@90**: 96.8%
- **Already trained**, no retraining needed

### 🥉 Challenger #2: Seasonal Naive (existing)
- **Pinball@CF**: 33.25
- **Calibration**: 0.271
- **Coverage@90**: 95.9%
- **Simple, stable, interpretable**

## Evaluation Plan

After training completes, run evaluation with:
- ✅ Proper sequential newsvendor optimization (SIP-based)
- ✅ Actual initial state from `state.parquet`
- ✅ Integer optimization (Q ∈ {0..1000})
- ✅ Clean demand data (capped)
- ✅ Exclude week-1-only costs

### Evaluation Command (TO BE IMPLEMENTED)
```bash
# After SIP optimization is implemented
python -m vn2 eval-models \
  --holdout 8 \
  --demand-path data/processed/demand_imputed_capped.parquet \
  --use-sip-optimization \
  --sip-grain 1000 \
  --cpu-fraction 1.0 \
  --out-suffix v4
```

## SIP Grain Decision

**Chosen: 1000 units**
- Covers max 2-week demand (1,000)
- Covers 99%+ of all observations (99th percentile 2-week = 322)
- Brute-force search over 1000 candidates is fast
- PMF array: `pmf[0..1000]` for discrete probability mass

## What's Left to Implement

1. ✅ Stockout-aware SLURP - DONE
2. ✅ Clean capped data - DONE
3. ⏳ SIP-based sequential newsvendor optimization - TODO
4. ⏳ Use actual initial state - TODO
5. ⏳ Integer optimization over Q ∈ {0..1000} - TODO

## Timeline for This Afternoon

**Phase 1: Training** (~45 min)
- Train stockout-aware SLURP on 599 SKUs × 12 folds
- Checkpoint restart enabled

**Phase 2: Implement SIP Optimization** (~1-1.5 hours)
- Discrete PMF convolution for inventory dynamics
- Brute-force integer optimization
- Update evaluation pipeline

**Phase 3: Evaluation** (~30-45 min)
- Run 3 models: SLURP-stockout-aware, SLURP-original, Seasonal-Naive
- 599 SKUs × 3 models × 8 folds = 14,376 tasks
- With 12 workers: ~30-45 minutes

**Total: ~3 hours** - Should have results by evening for Monday submission!

## Expected Grain for SIP

**PMF resolution: 1001 points (0 to 1000 units)**
- Convert 13 quantiles → 1001-point discrete PMF via linear interpolation
- Sufficient for optimization given demand characteristics
- Can refine to finer quantiles (99 levels) in future if needed

## Next Steps When You Return

1. Kick off training: `python -m vn2 forecast --config configs/forecast.yaml --n-jobs 12`
2. Monitor: `tail -f logs/forecast_*.log`
3. When done, I'll implement SIP optimization
4. Run evaluation
5. Review results, select champion for Monday submission

