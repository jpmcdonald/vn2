# SLURP Ablation Study: Ready to Train

## Status: ✅ IMPLEMENTATION COMPLETE

All code changes are complete for the 2×2 factorial ablation study of SLURP variants.

## What Was Changed

### 1. CLI Update (`src/vn2/cli.py`)
- ✅ Changed data source: `demand_imputed.parquet` → `demand_imputed_capped.parquet`
- ✅ Added `make_slurp_surd()` factory (SURD transforms, no stockout handling)
- ✅ Registered `slurp_surd` in model selection logic

### 2. Config Update (`configs/forecast.yaml`)
- ✅ Added `slurp_surd` configuration
- ✅ Enabled 3 models for training: `slurp_bootstrap`, `slurp_surd`, `slurp_stockout_aware`
- ✅ Disabled `slurp_surd_stockout_aware` (already trained on clean data)

### 3. Checkpoint Management
- ✅ Backed up old checkpoints to `*_OLD` directories
- ✅ Cleared `slurp_bootstrap` and `slurp_stockout_aware` for re-training

## Model Matrix (4 SLURP Variants)

| Model | SURD | Stockout | Data | Status |
|-------|------|----------|------|--------|
| `slurp_bootstrap` | ❌ No | ❌ No | Capped | 🔄 Ready to train |
| `slurp_surd` | ✅ Yes | ❌ No | Capped | 🆕 Ready to train |
| `slurp_stockout_aware` | ❌ No | ✅ Yes | Capped | 🔄 Ready to train |
| `slurp_surd_stockout_aware` | ✅ Yes | ✅ Yes | Capped | ✅ Already trained (7,188 checkpoints) |

## Ablation Design

### Main Effects
1. **SURD effect alone**: `slurp_surd` - `slurp_bootstrap`
2. **Stockout effect alone**: `slurp_stockout_aware` - `slurp_bootstrap`

### Interaction
3. **Combined effect**: `slurp_surd_stockout_aware` - `slurp_bootstrap`
4. **Synergy test**: Is combined > (SURD alone + Stockout alone)?

### Research Questions
- Does SURD stabilize variance across SKUs?
- Does stockout-awareness improve tail coverage?
- Is there positive interaction (synergy)?
- Which component contributes more to cost reduction?

## Training Commands

### Smoke Test (1 SKU, ~30 seconds)
```bash
source activate.sh
python -m vn2.cli forecast --test --n-jobs 4
```

### Full Training (3 models, ~6-8 hours)
```bash
source activate.sh
python -m vn2.cli forecast --n-jobs 12 --resume
```

**Expected:**
- 3 models × 599 SKUs × 12 folds = 21,564 fits
- Time: ~6-8 hours on 12 cores
- Checkpoints: `models/checkpoints/slurp_{bootstrap,surd,stockout_aware}/`

### Monitor Progress
```bash
# Count completed tasks
cat models/checkpoints/progress.json | jq '.completed | length'

# Check checkpoint directories
ls -d models/checkpoints/slurp_*

# Watch log (if running in background)
tail -f logs/forecast_*.log
```

## Evaluation Commands

### Full Ablation with SIP Optimization
```bash
python -m vn2.cli eval-models \
  --use-sip-optimization \
  --sip-grain 1000 \
  --out-suffix v4_ablation \
  --holdout 8 \
  --n-jobs 12 \
  --batch-size 2000 \
  --resume
```

**Expected:**
- 5 models (4 SLURP + seasonal_naive) × 599 SKUs × 8 folds = 23,960 tasks
- Time: ~4-6 hours on 12 cores
- Outputs: `models/results/{eval_folds,eval_agg,leaderboards}_v4_ablation.parquet`

### View Results
```bash
python -c "import pandas as pd; df = pd.read_parquet('models/results/leaderboards_v4_ablation.parquet'); print(df.to_string())"
```

## Expected Leaderboard

```
Model                        sip_realized_cost_w2  coverage_90  crps  mase
seasonal_naive                        XXXX.XX         0.XX    X.XX  X.XX
slurp_bootstrap                       XXXX.XX         0.XX    X.XX  X.XX  ← Baseline
slurp_surd                            XXXX.XX         0.XX    X.XX  X.XX  ← +SURD
slurp_stockout_aware                  XXXX.XX         0.XX    X.XX  X.XX  ← +Stockout
slurp_surd_stockout_aware             XXXX.XX         0.XX    X.XX  X.XX  ← Champion?
```

## Data Quality

### Clean Capped Data
- **File:** `data/processed/demand_imputed_capped.parquet`
- **Max demand:** 500 units (realistic, no billions)
- **Capping rule:** `min(imputed_value, max(50, observed_max × 10, 500))`
- **Stockout signal:** Preserved (in_stock column)

### SURD Transforms
- **File:** `data/processed/surd_transforms.parquet`
- **SKUs:** 599
- **Transforms:** log (~70%), cbrt (~20%), sqrt (~5%), log1p (~3%), identity (~2%)

## Validation Checklist

### Pre-Training ✅
- [x] Capped data exists and is reasonable
- [x] SURD transforms loaded
- [x] All 4 SLURP models configured
- [x] Old checkpoints backed up
- [x] CLI loads capped data

### During Training
- [ ] Quantiles are reasonable (< 1000 units)
- [ ] No timeout errors
- [ ] Checkpoints being created
- [ ] Progress advancing steadily

### Post-Training
- [ ] All 4 models have 7,188 checkpoints
- [ ] Quantiles monotonic (q05 ≤ q50 ≤ q95)
- [ ] Max forecast < 1000 units
- [ ] Coverage metrics vary by model

### Post-Evaluation
- [ ] Clear cost differences between models
- [ ] SURD effect quantified
- [ ] Stockout effect quantified
- [ ] Interaction effect assessed
- [ ] Champion identified

## Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Smoke test | 2 min | Verify setup |
| Full training | 6-8 hours | 3 models in parallel |
| Evaluation | 4-6 hours | 5 models with SIP |
| Analysis | 1-2 hours | Ablation study |
| **Total** | **12-17 hours** | Can run overnight + next day |

## Rollback Plan

If issues arise:
```bash
# Restore old checkpoints
mv models/checkpoints/slurp_bootstrap_OLD models/checkpoints/slurp_bootstrap
mv models/checkpoints/slurp_stockout_aware_OLD models/checkpoints/slurp_stockout_aware

# Revert CLI to old data
# Edit src/vn2/cli.py: demand_imputed_capped.parquet → demand_imputed.parquet

# Continue with just slurp_surd_stockout_aware
# Edit configs/forecast.yaml: disable all except slurp_surd_stockout_aware
```

## Files Modified

### Code (1 file)
- `src/vn2/cli.py`: Data path, factory, registration

### Config (1 file)
- `configs/forecast.yaml`: Added slurp_surd, enabled 3 models

### Checkpoints (backed up)
- `models/checkpoints/slurp_bootstrap_OLD/`
- `models/checkpoints/slurp_stockout_aware_OLD/`

## Next Steps

1. **Run smoke test:**
   ```bash
   source activate.sh
   python -m vn2.cli forecast --test --n-jobs 4
   ```

2. **Verify output looks reasonable** (no extreme values)

3. **Launch full training:**
   ```bash
   python -m vn2.cli forecast --n-jobs 12 --resume
   ```

4. **Monitor progress** and let it run overnight

5. **After training, run evaluation** with SIP optimization

6. **Analyze ablation effects** and identify champion

---

**Ready to train 3 SLURP models on clean data!** 🚀

This ablation study will definitively answer:
- How much does SURD help?
- How much does stockout-awareness help?
- Is there synergy when combined?

