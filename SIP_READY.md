# SIP Optimization: Ready for Evaluation

## Status: ✅ IMPLEMENTATION COMPLETE

All components of the SIP-based sequential newsvendor optimization have been implemented, tested, and integrated into the evaluation pipeline.

## What Was Built

### 1. Core SIP Module (`src/vn2/analyze/sip_opt.py`)
- ✅ PMF interpolation from quantiles
- ✅ Inventory convolution with stockout handling
- ✅ Discrete order optimization (brute-force over integers)
- ✅ Realized cost computation with actual demand
- ✅ Unit tests passed (deterministic, stockout, asymmetric costs)

### 2. Evaluation Integration (`src/vn2/analyze/model_eval.py`)
- ✅ `compute_sip_cost_metric()` function
- ✅ Initial state loading from `state.parquet`
- ✅ SIP metrics added to evaluation results
- ✅ Aggregation updated for SIP costs (sum) and service levels (mean)
- ✅ Leaderboard ranking by `sip_realized_cost_w2`

### 3. CLI Interface (`src/vn2/cli.py`)
- ✅ `--use-sip-optimization` flag
- ✅ `--sip-grain` parameter (default: 1000)
- ✅ `--out-suffix` for versioned outputs (e.g., "v4")
- ✅ `--models` filter for specific model evaluation
- ✅ Automatic state loading when SIP enabled
- ✅ Capped demand data used for SIP mode

### 4. Documentation
- ✅ `SIP_IMPLEMENTATION.md`: Technical details
- ✅ `RUN_SIP_EVAL.md`: Execution runbook
- ✅ Inline code comments and docstrings

## Key Design Decisions

### 1. Week 2 Only Costs
- **Rationale:** Week 1 state is deterministic (same for all models)
- **Impact:** Fair comparison focused on forecast quality
- **Implementation:** `exclude_week1=True` in `compute_sip_cost_metric()`

### 2. Integer Order Quantities
- **Rationale:** Real-world constraint (can't ship fractional units)
- **Impact:** Discrete optimization required
- **Implementation:** Brute-force search over [0, 1000]

### 3. PMF Grain = 1000
- **Rationale:** Balances resolution vs. computation time
- **Impact:** Demands > 1000 truncated (rare after capping)
- **Implementation:** Configurable via `--sip-grain`

### 4. Actual Initial State
- **Rationale:** Realistic starting conditions from historical data
- **Impact:** Models evaluated in true operational context
- **Implementation:** Load from `state.parquet` indexed by (store, product, week)

### 5. Capped Demand Data
- **Rationale:** Extreme imputed outliers (billions) are unrealistic
- **Impact:** PMF grain remains tractable
- **Implementation:** Use `demand_imputed_capped.parquet` for SIP mode

## Validation Results

### Unit Tests (test_sip_opt.py)
```
✓ PMF sums to 1.0
✓ Deterministic case: I0=5, Q1=10, D1=10 → I_end=5
✓ Stockout case: I0=5, Q1=0, D1=10 → I_end=0
✓ Deterministic demand: Q*=50 achieves zero cost
✓ Asymmetric costs: Q*=84 for uniform [0,100] (CF=0.833)
✓ Shortage cost correct: 50 units × 1.0 = 50.0
```

All tests passed ✅

## Ready to Run

### Command
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

### Expected Outputs
- `models/results/eval_folds_v4.parquet` (14,376 rows)
- `models/results/eval_agg_v4.parquet` (1,797 rows)
- `models/results/leaderboards_v4.parquet` (3 rows)

### Estimated Time
- **12 cores:** 2-4 hours
- **6 cores:** 4-8 hours

## Metrics to Watch

### Primary (Decision-Focused)
- `sip_realized_cost_w2`: Total cost across all SKUs and folds (LOWER is better)
- `sip_service_level_w2`: Average service level (HIGHER is better)
- `sip_fill_rate_w2`: Average fill rate (HIGHER is better)

### Secondary (Accuracy)
- `mae`: Mean absolute error
- `mase`: Mean absolute scaled error
- `pinball_cf_h1`: Pinball loss at critical fractile

### Tertiary (Density)
- `coverage_90`: 90% prediction interval coverage
- `crps`: Continuous ranked probability score

## Expected Results

### Hypothesis
1. **SLURP Stockout-Aware** should outperform baseline SLURP by properly handling censored observations
2. **SLURP Bootstrap** should outperform Seasonal Naive due to conditional density estimation
3. **All models** should show differentiated costs (unlike v3 where all were tied)

### Success Criteria
- ✅ Costs are multiples of 0.2 (holding) or 1.0 (shortage)
- ✅ Models have different total costs (not all tied)
- ✅ Service levels vary by model (reflecting forecast quality)
- ✅ Champion model has lowest `sip_realized_cost_w2`

## Potential Issues

### 1. State File Missing
- **Symptom:** Warning about missing state data
- **Fallback:** Uses zero initial state (same as v3)
- **Fix:** Create `state.parquet` from raw data if needed

### 2. Extreme Computation Time
- **Symptom:** Tasks taking > 5 seconds each
- **Cause:** Large PMF grain or many SKUs with high demand
- **Fix:** Reduce `--sip-grain` to 500 or filter to specific models

### 3. Memory Issues
- **Symptom:** Process killed or swap thrashing
- **Cause:** Too many parallel workers
- **Fix:** Reduce `--n-jobs` to 6 or `--batch-size` to 1000

## Next Steps After Evaluation

1. **Review Leaderboard**
   ```bash
   python -c "import pandas as pd; df = pd.read_parquet('models/results/leaderboards_v4.parquet'); print(df.to_string())"
   ```

2. **Analyze Per-SKU Performance**
   ```python
   import pandas as pd
   agg = pd.read_parquet('models/results/eval_agg_v4.parquet')
   
   # Which SKUs benefit most from stockout-aware?
   pivot = agg.pivot_table(
       index=['store', 'product'],
       columns='model_name',
       values='sip_realized_cost_w2'
   )
   pivot['improvement'] = pivot['slurp_bootstrap'] - pivot['slurp_stockout_aware']
   print(pivot.nlargest(10, 'improvement'))
   ```

3. **Compare with Baseline (v3)**
   ```python
   v3 = pd.read_parquet('models/results/leaderboards_v3.parquet')
   v4 = pd.read_parquet('models/results/leaderboards_v4.parquet')
   
   # Cost reduction
   print(f"v3 total cost: {v3['expected_cost'].sum()}")
   print(f"v4 total cost: {v4['sip_realized_cost_w2'].sum()}")
   ```

4. **Prepare Submission**
   - Use champion model for final forecast
   - Generate order quantities for submission template
   - Validate against constraints (budget, capacity, etc.)

## Files Modified

### New Files
- `src/vn2/analyze/sip_opt.py` (core SIP logic)
- `SIP_IMPLEMENTATION.md` (technical documentation)
- `RUN_SIP_EVAL.md` (execution runbook)
- `SIP_READY.md` (this file)

### Modified Files
- `src/vn2/analyze/model_eval.py` (integration)
- `src/vn2/cli.py` (CLI flags)

### No Changes Required
- Trained model checkpoints (already exist)
- Demand data (using capped version)
- Master data (unchanged)

## Rollback Plan

If SIP evaluation fails or produces unexpected results:

```bash
# Run baseline evaluation (v3 logic)
python -m vn2.cli eval-models \
  --out-suffix v3_rerun \
  --n-jobs 12 \
  --resume
```

SIP flags are opt-in; baseline evaluation is unchanged.

## Contact

For questions or issues:
1. Check `SIP_IMPLEMENTATION.md` for technical details
2. Review `RUN_SIP_EVAL.md` for troubleshooting
3. Run unit tests: `python test_sip_opt.py` (if recreated)

---

**Ready to launch evaluation!** 🚀

