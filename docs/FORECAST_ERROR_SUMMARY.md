# Forecast Error Summary (One-Pager)

## Objective

Answer: **Is our forecast error that far off?** Compare our quantile forecasts to actuals and to a naive baseline (e.g. seasonal naive or the official benchmark’s point forecast) on holdout weeks.

---

## Metrics to use


| Metric           | Where                                                                                                                    | Purpose                                |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------------------------------- |
| **Pinball loss** | [src/vn2/forecast/evaluation.py](../src/vn2/forecast/evaluation.py) `average_pinball_loss`                               | Quantile accuracy                      |
| **CRPS**         | Same module `average_crps`                                                                                               | Full distribution accuracy             |
| **Coverage**     | Same module `coverage_metrics`                                                                                           | Calibration of intervals               |
| **Cost-based**   | Same module `cost_based_metric`; [src/vn2/analyze/model_eval.py](../src/vn2/analyze/model_eval.py) `compute_cost_metric` | Decision-relevant (holding + shortage) |
| **MAE / RMSE**   | Same module (median vs actuals)                                                                                          | Point accuracy vs actuals              |


---

## How to run

1. **Data**: Ensure `data/processed/demand_long.parquet` includes holdout weeks (build with [scripts/build_demand_long.py](../scripts/build_demand_long.py) when Week 1–8 Sales are present).
2. **Script**: `scripts/analyze_forecast_metrics.py --demand-parquet data/processed/demand_long.parquet` (and checkpoint/selector paths as needed).
3. **By model**: Run evaluation per model (or selector) on the same holdout weeks; record pinball, CRPS, cost-based metric.
4. **Baseline**: Compute the same metrics for (a) seasonal naive, (b) official benchmark’s point forecast (replicated from [Official_Benchmark.py](../notebooks/Official_Benchmark.py)) so we can see if we’re “that far off” in quantile space.

---

## One-page output (template)


| Model / baseline           | Pinball | CRPS  | Coverage (80%) | Cost-based (holdout) |
| -------------------------- | ------- | ----- | -------------- | -------------------- |
| SELECTOR                   | *run*   | *run* | *run*          | *run*                |
| zinb                       | *run*   | *run* | *run*          | *run*                |
| slurp_*                    | *run*   | *run* | *run*          | *run*                |
| Seasonal naive             | *run*   | *run* | *run*          | *run*                |
| Official benchmark (point) | N/A     | *run* | N/A            | *run*                |


**Conclusion**: *Fill after runs: whether our forecast error is the main driver of the gap to benchmark/winner, and which levers (calibration, SURD, stockout, selector) to prioritize.*

---

## References

- [src/vn2/forecast/evaluation.py](../src/vn2/forecast/evaluation.py) — `evaluate_forecast`, pinball, CRPS, coverage, cost
- [scripts/analyze_forecast_metrics.py](../scripts/analyze_forecast_metrics.py) — demand-parquet and checkpoint usage
- [WHY_NOT_BEAT_BENCHMARK.md](WHY_NOT_BEAT_BENCHMARK.md) — root cause and decomposition

