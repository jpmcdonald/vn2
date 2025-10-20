# Model Evaluation V3: Newsvendor-Optimized Analysis

## Overview

V3 evaluation implements **newsvendor-specific metrics** for inventory optimization with asymmetric costs (shortage=1.0, holding=0.2), fixing critical issues from v1/v2.

## Critical Fractile

For your cost ratio: **τ = 1.0 / (1.0 + 0.2) = 0.833 (83.3rd percentile)**

This is your **optimal service level** for the newsvendor problem.

## What Changed in V3

### 1. Newsvendor-Specific Metrics ✅

**Pinball at Critical Fractile:**
- `pinball_cf_h1`, `pinball_cf_h2`: Quantile forecast loss at τ=0.833
- **PRIMARY METRIC**: Directly measures decision quality for your ordering problem

**Calibration at Critical Fractile:**
- `hit_cf_h1`, `hit_cf_h2`: Did actual demand exceed forecast at τ=0.833?
- Target: ~0.833 (83.3% of observations should be below forecast)
- Tells you if your critical fractile is correctly calibrated

**Local Sharpness (Decision Zone):**
- `local_width_h1`, `local_width_h2`: Width from 80th to 90th percentile
- Narrower = more confident decisions around your critical fractile
- More important than overall PI width (80-95%)

**Quantile Gradient:**
- `quantile_gradient_h1`, `quantile_gradient_h2`: Steepness around critical fractile
- Measures sensitivity: how much does order quantity change per unit forecast error?

**Critical Fractile Asymmetry:**
- `cf_asymmetry_h1`, `cf_asymmetry_h2`: Ratio of upper/lower tails around τ=0.833
- ~1.0 = symmetric, >1.0 = right-skewed (longer upper tail)

**Asymmetric Loss:**
- `asymmetric_loss_h1`, `asymmetric_loss_h2`: Cost-weighted forecast error
- Penalizes underprediction 5× more than overprediction

### 2. Decision-Based Metrics ✅

**Service Metrics:**
- `service_level`: Fraction of periods without stockouts (0-1)
- `fill_rate`: Fraction of demand satisfied (0-1)

**Regret:**
- `regret_qty`: |ordered - optimal_order|
- `optimal_qty`: What you should have ordered (with perfect foresight)
- Measures opportunity cost of forecast uncertainty

**Cost Components:**
- `shortage_cost`: Total from stockouts
- `holding_cost`: Total from excess inventory
- `expected_cost`: Total = shortage + holding

### 3. Integer Math & Exact Costs ✅

**Integerization:**
- Order quantity: `int(ceil(order_qty))` - round UP (conservative for asymmetric costs)
- Demand: `int(round(demand))` - round to nearest integer

**Cost Verification:**
- All costs now exact multiples of 0.2
- Shortage: 1.0 × integer units
- Holding: 0.2 × integer units

### 4. Proper Aggregation ✅

**Per-SKU (across folds):**
- Costs: **SUM** (total cost across 8 folds)
- Accuracy metrics: **MEAN**

**Leaderboard (across SKUs):**
- Costs: **SUM** (total cost across all 599 SKUs)
- Accuracy metrics: **MEAN**
- **Rank by**: `expected_cost` (total) ← THIS IS THE WINNER

## Metric Interpretation Guide

### For Newsvendor Optimization (Your Use Case)

**Primary Rankings:**
1. **Total Expected Cost** (lower is better)
   - Sum across all SKUs and folds
   - The ultimate decision metric

2. **Pinball at Critical Fractile** (`pinball_cf_h1`, lower is better)
   - Best predictor of cost performance
   - Directly measures quantile forecast quality at τ=0.833

3. **Calibration at CF** (`hit_cf_h1`, target ~0.833)
   - Are you hitting your service level target?
   - Over-coverage (>0.833) wastes inventory
   - Under-coverage (<0.833) causes stockouts

**Secondary Rankings:**
4. **Local Width** (lower is better)
   - Sharper decisions around critical fractile

5. **Service Level** (higher is better, but trade-off with cost)
   - 100% service = infinite cost
   - Optimal ≠ 100%, depends on cost ratio

6. **Asymmetric Loss** (lower is better)
   - Cost-weighted accuracy

**Less Important (for your problem):**
- Point accuracy (MAE, RMSE): Useful but not decision-relevant
- Overall coverage/width: Critical fractile matters more
- CRPS: Good overall metric but not specific to your problem

## Per-SKU Model Selection

Use `model_selector.py` to pick the best model for each SKU:

```bash
# By pinball at critical fractile (recommended)
python -m vn2.analyze.model_selector \
  --eval-agg models/results/eval_agg_v3.parquet \
  --metric pinball_cf_h1 \
  --output models/results/best_models_by_pinball.parquet

# By total cost (alternative)
python -m vn2.analyze.model_selector \
  --eval-agg models/results/eval_agg_v3.parquet \
  --metric expected_cost \
  --output models/results/best_models_by_cost.parquet
```

This creates a mapping: (store, product) → best_model

## Expected Results

Based on v2 analysis:

**Overall Winner (by pinball & sharpness):**
- **SLURP Bootstrap**
  - Lowest pinball loss
  - Sharpest intervals
  - Well-calibrated (96.8% coverage, slight over-coverage is OK)
  - Full SKU coverage (599/599)

**Strong Alternatives:**
- **NGBoost**: Best on 23.7% of SKUs (CRPS), but under-covers (35%)
- **Croston TSB**: Best calibration (57.8% of SKUs), but wide intervals
- **KNN Profile**: Competitive point accuracy, but density outputs are buggy

**Per-SKU Breakdown:**
- SLURP wins 57% of SKUs on pinball
- SLURP wins 49% of SKUs on CRPS
- NGBoost wins 21% of SKUs on sharpness
- Croston TSB wins 58% of SKUs on calibration

→ **Ensemble or per-SKU selection likely best**

## V3 Runtime

- **12 workers (full cores)**
- **57,384 tasks** (599 SKUs × 12 models × 8 folds)
- **Expected: ~45-60 minutes**
- **Outputs**: `eval_folds_v3.parquet`, `eval_agg_v3.parquet`, `leaderboards_v3.parquet`

## Monitoring

```bash
# Watch progress
tail -f logs/eval_20251020_044913.log

# Check status
ps -p 24804

# View progress
cat models/results/eval_progress.json | jq '.completed | length'
```

## Post-Run Analysis

Once complete, analyze results:

```bash
# Load and examine
python -c "
import pandas as pd
leaderboard = pd.read_parquet('models/results/leaderboards_v3.parquet')
print(leaderboard[['model_name', 'expected_cost', 'pinball_cf_h1', 'hit_cf_h1', 'service_level']].to_string())
"

# Per-SKU winners
python -m vn2.analyze.model_selector \
  --eval-agg models/results/eval_agg_v3.parquet \
  --metric pinball_cf_h1

# Detailed per-SKU analysis
python -c "
import pandas as pd
agg = pd.read_parquet('models/results/eval_agg_v3.parquet')
# Filter to SLURP for one SKU
sample = agg[(agg['store']==0) & (agg['product']==126) & (agg['model_name']=='slurp_bootstrap')]
print(sample.T)
"
```

## Key Insights

1. **Density shape > point accuracy** for inventory decisions
2. **Critical fractile performance** is what matters, not overall accuracy
3. **Total cost** is the ultimate metric (sum, not average)
4. **Integer math** ensures costs are exact multiples of 0.2
5. **Per-SKU selection** likely beats any single model
6. **Slight over-ordering** (positive bias) is optimal with 5:1 shortage:holding ratio

## Files Created

- `src/vn2/analyze/model_eval.py`: Updated with newsvendor metrics
- `src/vn2/analyze/model_selector.py`: Per-SKU model selection tool
- `models/results/eval_folds_v3.parquet`: Per-fold results with all metrics
- `models/results/eval_agg_v3.parquet`: Per-SKU aggregated results
- `models/results/leaderboards_v3.parquet`: Overall rankings by total cost
- `logs/eval_20251020_044913.log`: Evaluation log

## Next Steps

1. ✅ Wait for v3 completion (~45-60 min)
2. Review leaderboards_v3.parquet
3. Run per-SKU model selector
4. Analyze pinball_cf_h1 and calibration by model
5. Implement ensemble or hybrid strategy
6. Deploy to production forecasting pipeline

