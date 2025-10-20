# Ensemble Results Summary

## Completed Stages

### 1. Per-SKU Selector
- **Method**: For each SKU, select the model with lowest total SIP cost across folds
- **Implementation**: Post-hoc assembly from existing `eval_folds_v4_sip.parquet`
- **Artifact**: `models/results/eval_folds_v4_ens_selector.parquet`
- **Result**: SIP realized cost (week 2) = 14,122
- **Rank**: 3rd among evaluated methods
- **Notes**: QRF wins 57.5% of SKUs; ZINB 3.5%; ties dominate remaining

### 2. Cohort-Gated Selector
- **Method**: Learn rules mapping cohort features (rate_bin, zero_bin, CV_bin, stockout_bin) to best model per cohort
- **Rules Learned**: 21 cohort combinations
- **Artifact**: `models/results/eval_folds_v4_ens_cohort.parquet`
- **Result**: SIP realized cost (week 2) = 13,715
- **Rank**: Tied 2nd with QRF
- **Notes**: Cohort rules converge to QRF for most cohorts; minimal discrimination beyond per-SKU selector

### 3. Decision-Level Ensemble (Deferred)
- **Status**: Infrastructure ready (`cost_curve_vs_Q` in `sip_opt.py`); not yet evaluated
- **Reason**: Requires re-loading model checkpoints and computing cost curves per fold/SKU; estimated 2–4 hrs on 12 cores
- **Next Steps**: If needed, implement decision-level pooling for ZINB+QRF with uniform or tuned weights

## Leaderboard Comparison (SIP Realized Cost, Week 2)

| Rank | Method              | Model/Ensemble        | Cost (Week 2) | Notes                          |
|------|---------------------|-----------------------|---------------|--------------------------------|
| 1    | Single Model        | ZINB                  | 10,437        | Best overall                   |
| 2    | Single Model        | QRF                   | 13,715        | Dominates most SKUs            |
| 2    | Cohort-Gated        | ensemble_cohort       | 13,715        | Tied with QRF                  |
| 3    | Per-SKU Selector    | ensemble_selector     | 14,122        | Modest improvement over naive  |
| 4    | Single Model        | Croston Classic/SBA   | 16,278        |                                |
| 5    | Single Model        | Croston TSB/ETS/...   | 18,870        | Large tie group                |

## Key Findings

1. **ZINB dominates overall**: Lowest aggregate cost despite winning only 3.5% of SKUs
2. **QRF dominates per-SKU**: Wins 57.5% of SKUs but higher aggregate cost than ZINB
3. **Ensembles offer modest gains**: Selector and cohort-gated reduce cost vs naive baseline but don't beat ZINB
4. **Cohort rules collapse to QRF**: Limited discrimination; most cohorts select QRF
5. **Ties dominate fold-level**: Many models share identical costs per fold, limiting ensemble discrimination

## Implications

- **For submission**: Use ZINB as primary model; consider QRF for SKUs where ZINB underperforms
- **For paper**: Document ensemble methods and results; highlight ZINB's aggregate advantage vs QRF's per-SKU dominance
- **For future work**: Explore decision-level pooling with tuned weights; investigate why ZINB wins aggregate despite losing most SKUs

## Artifacts Generated

- `models/results/per_sku_selector_map.parquet`: Per-SKU champion mapping
- `models/results/per_sku_champion_shares__v4_sip.parquet`: Aggregated win shares
- `models/results/linear_pool_weights_zinb_qrf.parquet`: Weight grid search (uniform costs due to ties)
- `models/results/cohort_features_temp.parquet`: Cohort features (rate, zero_ratio, CV, stockout bins)
- `models/results/eval_folds_v4_ens_selector.parquet`: Selector ensemble fold results
- `models/results/eval_folds_v4_ens_cohort.parquet`: Cohort ensemble fold results
- `models/results/leaderboards__v4_ens_selector.parquet`: Selector leaderboard
- `models/results/leaderboards__v4_ens_cohort.parquet`: Cohort leaderboard
- `models/results/leaderboards__v4_all.parquet`: Combined leaderboard (all runs + ensembles)

## Next Steps (Optional)

1. Implement and evaluate decision-level pooling (ZINB+QRF)
2. Generate ensemble rank trajectory plots
3. Add ensemble section to paper draft
4. Merge feature/ensemble branch to main after validation

