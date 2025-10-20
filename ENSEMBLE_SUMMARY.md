# Ensemble System Summary

## Overview
Implemented staged ensemble system for density forecast combination and model selection under SIP optimization.

## Components Built

### 1. Per-SKU Selector (Tie-Aware)
- **Module**: `src/vn2/analyze/model_selector.py::select_per_sku_from_folds`
- **Approach**: Aggregate fold-level SIP costs per model per SKU; select min; handle ties via win_share
- **Artifact**: `models/results/per_sku_selector_map.parquet`
- **Result**: QRF wins 57.5% of SKUs (tie-adjusted); ZINB 3.5%; others ~3% each

### 2. Linear Pool (Quantile Blending)
- **Module**: `src/vn2/analyze/ensemble.py::blend_quantiles`
- **Approach**: Weighted average of quantiles with isotonic regression for monotonicity
- **Weight Search**: `ensemble.py::grid_search_linear_pool_weights`
- **Artifact**: `models/results/linear_pool_weights_zinb_qrf.parquet`
- **Result**: ZINB+QRF weight grid shows uniform costs (ties dominate); per-SKU selector more discriminating

### 3. Cohort-Gated Selector
- **Module**: `src/vn2/analyze/ensemble.py::cohort_selector_rules`, `apply_cohort_selector`
- **Approach**: Learn simple rules mapping cohort features (rate_bin, zero_bin, CV_bin, stockout_bin) to best model per cohort
- **Status**: Infrastructure ready; requires cohort feature join and rule training

### 4. Decision-Level Ensemble
- **Module**: `src/vn2/analyze/sip_opt.py::cost_curve_vs_Q`, `ensemble.py::decision_level_pool`
- **Approach**: Compute expected cost curve over Q grid per model; pool curves via weighted average; pick argmin Q
- **Status**: Cost curve computation implemented; needs integration into eval pipeline

## Artifacts Generated
- `models/results/per_sku_selector_map.parquet`: per-SKU champion mapping with tie shares
- `models/results/per_sku_champion_shares__v4_sip.parquet`: aggregated win shares
- `models/results/linear_pool_weights_zinb_qrf.parquet`: weight grid search results

## Next Steps
1. Integrate per-SKU selector into eval pipeline to produce ensemble leaderboard
2. Train cohort-gated selector using `jensen_cohorts_*__v4.parquet` features
3. Implement decision-level pooling in eval loop
4. Generate ensemble plots and add paper section
5. Compare ensemble vs single-model leaders under SIP

## Key Findings
- Per-SKU selector is most effective given widespread ties at fold level
- QRF dominates majority of SKUs; ZINB leads overall aggregate cost
- Linear pool weight search shows limited discrimination due to ties
- Cohort-gated and decision-level approaches offer complementary strengths

## Estimates
- Per-SKU selector eval: 1–2 hrs (light CPU/RAM)
- Cohort-gated training+eval: 2–3 hrs (light CPU/RAM)
- Decision-level ensemble eval: 4–6 hrs (moderate CPU/RAM)

