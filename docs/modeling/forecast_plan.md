## Forecasting plan (champion/challenger)

Goal: select a calibrated density forecaster per segment (ABC/XYZ), with challengers for comparison.

### Candidates

- Seasonal naive + bootstrapped residuals (PI)
- ETS with prediction intervals
- TBATS/Prophet with PI
- Quantile regression (linear + interactions)
- LightGBM quantile (P10..P90 or custom grid)
- NGBoost / distributional regression (e.g., LogNormal, Poisson)
- Intermittent demand: Croston/SBA/TSB (with stochastic PI)

### Data

- Train on `demand_imputed.parquet`
- Segment by ABC/XYZ; optionally train per segment

### Metrics

- Pinball loss (averaged over quantiles)
- Coverage and interval width (80/90/95%)
- CRPS (optional)
- Stability: rolling-window performance drift

### Protocol

1. Pilot on a slice (few stores/products) to sanity-check features and runtime
2. Train per candidate; log metrics per segment
3. Select champion per segment; keep top 2 challengers
4. Calibrate quantiles if systematic miscoverage (isotonic or conformal)

### Artifacts & resume

- Save per-model artifacts into `models/<model>/<segment>/...`
- Idempotent resume: skip if artifact exists with matching data hash/version


