# Sequential L=2 Leaderboard (H=12 epochs)

Per-SKU sequential planning with exact realized costs at decision-affected periods.

| model_name | portfolio_cost | n_skus | mean_sku_cost | p50_sku | avg_coverage |
| --- | --- | --- | --- | --- | --- |
| SELECTOR | 5593.00 | 599 | 9.34 | N/A | N/A |
| zinb | 8905.20 | 599 | 14.87 | 7.00 | 0.9719 |
| slurp_bootstrap | 9769.80 | 599 | 16.31 | 7.80 | 1.0000 |
| slurp_stockout_aware | 10049.40 | 599 | 16.78 | 8.60 | 1.0000 |
| slurp_bootstrap_OLD | 10133.80 | 599 | 16.92 | 8.60 | 1.0000 |
| slurp_stockout_aware_OLD | 10144.40 | 599 | 16.94 | 8.60 | 1.0000 |
| knn_profile | 10662.60 | 599 | 17.80 | 9.00 | 1.0000 |
| lightgbm_quantile | 14092.60 | 599 | 23.53 | 7.00 | 1.0000 |
| linear_quantile | 14092.60 | 599 | 23.53 | 7.00 | 1.0000 |
| seasonal_naive | 15429.00 | 599 | 25.76 | 9.60 | 1.0000 |
| ngboost | 22132.00 | 599 | 36.95 | 17.60 | 1.0000 |
| qrf | 26029.80 | 599 | 43.46 | 10.20 | 1.0000 |
| croston_sba | 60268.40 | 599 | 100.62 | 30.00 | 1.0000 |
| croston_classic | 61124.20 | 599 | 102.04 | 30.00 | 1.0000 |
| lightgbm_point | 77159.80 | 599 | 128.81 | 110.80 | 1.0000 |
| zip | 98550.00 | 599 | 164.52 | 10.00 | 1.0000 |
| croston_tsb | 151305.00 | 599 | 252.60 | 94.00 | 1.0000 |
| ets | 274324.20 | 599 | 457.97 | 227.20 | 1.0000 |