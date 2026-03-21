# Residual Diagnostics: Integration Recommendation

## A. Autocorrelation of Residuals

- 0/599 SKUs (0.0%) show significant autocorrelation (Ljung-Box, p<0.05 at any tested lag)
- **Finding**: Residual autocorrelation is limited. Bootstrap approaches are likely adequate.

## B. Heteroskedasticity

- 345/599 SKUs (57.6%) show significant heteroskedasticity (Breusch-Pagan, p<0.05)
- Taylor's law on residuals: alpha = 0.656
- **Finding**: Residual variance scales with demand level. Constant-variance bootstrap will be miscalibrated for high/low demand SKUs.

## C. Tail Behaviour

- 542/599 SKUs (90.5%) have normally-distributed residuals (Jarque-Bera, p>=0.05)
- Mean excess kurtosis: -0.28 (0 = Normal, positive = heavy tails)
- Mean fraction beyond 3-sigma: 0.000 (Normal expectation: 0.003)
- **Finding**: Tail behaviour is near-Normal.

## D. Cross-Model Comparison

| Model | MAE | Bias | CRPS | RMSE/MAE ratio |
|---|---|---|---|---|
| slurp_surd | 1.597 | -0.688 | 0.954 | 1.18 |
| slurp_bootstrap | 1.597 | -0.687 | 0.926 | 1.18 |
| slurp_surd_stockout_aware | 1.604 | -0.575 | 0.972 | 1.18 |
| slurp_stockout_aware | 1.615 | -0.489 | 0.958 | 1.17 |
| deepar_surd | 1.734 | -1.220 | 1.321 | 1.16 |
| lightgbm_surd | 1.902 | 0.358 | 1.077 | 1.12 |
| lightgbm_quantile | 1.903 | 0.359 | 1.054 | 1.12 |
| deepar | 2.049 | -1.543 | 1.398 | 1.12 |
| seasonal_naive | 110.332 | 108.005 | 130.077 | 1.49 |

## Integration Decision

**Recommendation: Cautious — consider hybrid approach.** Residuals exhibit heteroskedastic residuals. SLURP bootstrap may work for most SKUs but consider native distributional outputs for the affected cohort.