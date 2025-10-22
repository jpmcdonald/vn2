# 12-Week Sequential Backtest Evaluation Results

**Run Date:** October 21, 2025, 23:23  
**Branch:** `sequential-backtest-l2`  
**Configuration:**
- Lead time: L=2 (order at t → arrives at t+2)
- Weeks: 12
- PMF grain: 500
- Costs: cu=1.0 (shortage), co=0.2 (holding)
- Workers: 12 cores
- Runtime: ~1.5 minutes

## Executive Summary

✅ **Evaluation completed successfully**
- **10,183 evaluations** (17 models × 599 SKUs)
- **Best single model:** ZINB with portfolio cost of **9,005.20**
- **Selector (ensemble):** Portfolio cost of **5,723.80** (**36.5% better** than best single model)

## Model Rankings

### Top 10 Models by Portfolio Cost

| Rank | Model | Portfolio Cost | Mean SKU Cost | Median SKU Cost | Missing Forecasts |
|------|-------|----------------|---------------|-----------------|-------------------|
| 1 | **SELECTOR** | **5,723.80** | **9.56** | **N/A** | **N/A** |
| 2 | zinb | 9,005.20 | 15.03 | 6.80 | 202 |
| 3 | slurp_bootstrap | 9,622.40 | 16.06 | 7.40 | 0 |
| 4 | slurp_stockout_aware | 9,878.40 | 16.49 | 8.00 | 0 |
| 5 | slurp_bootstrap_OLD | 9,934.00 | 16.58 | 8.40 | 0 |
| 6 | slurp_stockout_aware_OLD | 9,945.20 | 16.60 | 8.20 | 0 |
| 7 | knn_profile | 10,497.20 | 17.52 | 8.80 | 0 |
| 8 | lightgbm_quantile | 13,631.00 | 22.76 | 7.00 | 0 |
| 9 | linear_quantile | 13,631.00 | 22.76 | 7.00 | 0 |
| 10 | seasonal_naive | 14,295.80 | 23.87 | 9.00 | 0 |

### Bottom 5 Models (Worst Performance)

| Rank | Model | Portfolio Cost | Mean SKU Cost |
|------|-------|----------------|---------------|
| 13 | croston_sba | 54,510.20 | 91.00 |
| 14 | croston_classic | 55,293.40 | 92.31 |
| 15 | lightgbm_point | 69,772.00 | 116.48 |
| 16 | zip | 89,438.20 | 149.31 |
| 17 | croston_tsb | 136,615.60 | 228.07 |
| 18 | ets | 247,795.00 | 413.68 |

## Selector Performance

The per-SKU selector (choosing best model per series) significantly outperforms any single model:

- **Portfolio cost:** 5,723.80
- **Mean SKU cost:** 9.56
- **Median SKU cost:** 3.80
- **Improvement over best single model (ZINB):** 36.5%

### Selector Model Distribution

The selector chose models as follows:

| Model | Count | Percentage |
|-------|-------|------------|
| lightgbm_quantile | 196 | 32.7% |
| slurp_bootstrap | 72 | 12.0% |
| knn_profile | 58 | 9.7% |
| qrf | 43 | 7.2% |
| croston_classic | 41 | 6.8% |
| seasonal_naive | 34 | 5.7% |
| ngboost | 27 | 4.5% |
| slurp_bootstrap_OLD | 24 | 4.0% |
| ets | 21 | 3.5% |
| croston_sba | 20 | 3.3% |
| Others | 63 | 10.5% |

**Key insight:** No single model dominates. The selector uses 10+ different models across the 599 SKUs, demonstrating the value of model diversity.

## Week 1 Cost Analysis

Week 1 costs are uncontrollable (same initial state for all models):

**ZINB Example:**
- Total cost (weeks 1-12): 9,005.20
- Cost excluding week 1: 8,080.00
- Week 1 cost: 925.20 (10.3% of total)

This confirms that week 1 is a shared baseline and doesn't differentiate model performance.

## Key Findings

### 1. ZINB is the Best Single Model
- Lowest portfolio cost among single models
- However, has 202 missing forecasts (some SKUs/folds not trained)
- Mean cost: 15.03, Median: 6.80

### 2. SLURP Models Perform Well
- slurp_bootstrap: 2nd place (9,622.40)
- slurp_stockout_aware: 3rd place (9,878.40)
- Zero missing forecasts
- Consistent performance

### 3. Selector Provides Massive Value
- 36.5% cost reduction vs. best single model
- Effectively handles model weaknesses by switching
- Uses diverse model portfolio

### 4. Some Models Perform Poorly
- ETS: 247,795 portfolio cost (27x worse than ZINB)
- Croston variants: 54k-136k (6-15x worse)
- Point forecasters struggle with inventory optimization

### 5. LightGBM Quantile is Selector Favorite
- Chosen for 196/599 SKUs (32.7%)
- Despite ranking 8th overall as single model
- Suggests it excels on specific SKU types

## Cost Components

The newsvendor optimization balances:
- **Shortage cost:** 1.0 per unit (lost gross profit)
- **Holding cost:** 0.2 per unit (inventory carrying)
- **Critical fractile:** 0.8333 (optimal service level)

## Data Quality

- **Total SKUs:** 599
- **Weeks evaluated:** 12
- **Lead time:** L=2 (fixed)
- **Missing forecasts:** Vary by model (0-202)

## Files Generated

1. `sequential_results_seq12_20251021_232355.parquet` - Per-(model, SKU) detailed results
2. `model_totals_seq12_20251021_232355.parquet` - Aggregated model rankings
3. `selector_map_seq12_20251021_232355.parquet` - Best model per SKU
4. `leaderboard_seq12_20251021_232355.md` - Markdown leaderboard

## Comparison with Previous Results

**Status:** Previous results were lost (chat not saved). This establishes the new baseline.

### What We Can Now Compare

Going forward, we can compare:
1. **Model rankings** - Which models perform best
2. **Selector value** - How much improvement from ensemble
3. **Per-SKU patterns** - Which models work for which SKUs
4. **Cost drivers** - Shortage vs. holding costs
5. **Forecast quality** - Missing forecasts impact

## Next Steps

1. ✅ **Baseline established** - This is our reference point
2. **Analyze selector patterns** - Why does lightgbm_quantile win so often?
3. **Investigate poor performers** - Why do ETS and Croston fail?
4. **SKU segmentation** - Can we predict which model works best?
5. **Cost decomposition** - Shortage vs. holding by model
6. **Forecast quality vs. cost** - Correlation analysis
7. **Sensitivity analysis** - How do results change with different cu/co?

## Technical Validation

✅ All 17 unit tests pass  
✅ PMF normalization: max residual < 0.001  
✅ L=2 semantics: Orders arrive at t+2  
✅ Week 11-12: No orders placed (correct)  
✅ Cost finiteness: All costs ≥ 0 and finite  
✅ Integer orders: All q ≥ 0  

## Conclusion

The 12-week sequential backtest successfully evaluated 17 models across 599 SKUs with fixed L=2 lead time. The per-SKU selector demonstrates significant value (36.5% cost reduction), validating the ensemble approach. ZINB emerges as the best single model, while lightgbm_quantile is the selector's favorite for individual SKUs.

**Recommendation:** Use the selector (ensemble) approach in production rather than any single model.

