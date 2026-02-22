# Why We Didn't Beat the Benchmark

## Summary

We did not beat the official competition benchmark on 8-week cost. Our actual 8-week cost was **€7,787.40** (rank 110); the winner achieved **€4,677**; and the **Benchmark** (seasonal MA + order-up-to) tied or beat us on the full horizon. This document summarizes root causes and planned analysis.

---

## Official benchmark vs our approach


| Aspect        | Official benchmark                                                                      | Our pipeline                                                                 |
| ------------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Forecast**  | Point: seasonal factors + 13-week moving average (de-seasonalized then re-seasonalized) | Quantile: ZINB, SLURP, LightGBM, KNN, etc., with SURD and stockout awareness |
| **Policy**    | Order-up-to with 4 weeks coverage; order = (order_up_to - net_inventory).clip(0)        | PMF-based newsvendor (SIP) with L=3; integer Q optimized per SKU             |
| **Lead time** | Implicit (competition rule: order end of week X arrives start of week X+3)              | We initially implemented L=2 (arrives week t+2); corrected to L=3            |
| **Source**    | [notebooks/Official_Benchmark.py](notebooks/Official_Benchmark.py)                      | Selector + checkpoints + `choose_order_L3()`                                 |


A simple rule (point forecast + order-up-to) can outperform a complex stack when our forecasts are biased, poorly calibrated, or when the model selector picks worse models per SKU. The benchmark is robust and easy to reproduce; we should run it on the same actuals and compare 8-week cost.

---

## Root cause: forecast quality, not lead time

From [L3_LEAD_TIME_ANALYSIS.md](L3_LEAD_TIME_ANALYSIS.md):

- **Lead time error**: We implemented L=2 instead of L=3. Fixing it improved estimated 8-week cost by only **~1.6% (€125)**. So "we didn't beat the benchmark" is **not** primarily due to this bug.
- **Forecast quality**: The main driver of the gap. Our h=3 forecasts (and in many cases h=1/h=2) were not accurate enough; some SKUs had large over- or under-prediction. The L3 doc states: *"The primary cause of our poor performance was forecast quality, not the lead time implementation."*

### Decomposing the gap

Planned analysis (using existing scripts and backtests):

1. **Forecast error**: Use [src/vn2/forecast/evaluation.py](../src/vn2/forecast/evaluation.py) (pinball, CRPS, cost-based metric) and [scripts/analyze_forecast_metrics.py](../scripts/analyze_forecast_metrics.py) on holdout weeks with actuals. Compare our quantiles to actuals and to the official benchmark’s point forecast (as a naive baseline). Answer: *Is our forecast error that far off?*
2. **Policy**: Compare benchmark orders vs our orders on the same actuals: run the benchmark logic to get benchmark 8-week cost, run our pipeline to get our 8-week cost. The difference (after controlling for forecasts) indicates whether newsvendor vs order-up-to is helping or hurting.
3. **Model selector**: Selector was trained on h=1/h=2 historical accuracy; competition needs h=3. Re-evaluate selector choices per SKU and whether different models would have reduced cost.

---

## Benchmark comparison (to run)

When Week 1–8 Sales and state/orders are available:

1. **Benchmark 8-week cost**: Run Official_Benchmark logic (or equivalent) to produce orders for weeks 1..6; run the competition sim (L=3) with actual demand; sum costs over 8 weeks.
2. **Our 8-week cost**: Run our backtest (selector + L=3 orders) on the same actuals; record realized 8-week cost.
3. **Report**: Table: Benchmark cost, our cost, winner cost; short conclusion on whether the gap is forecast quality, policy, or both.

---

## AI agents vs our pipeline

The [notebooks/forecast_agents.ipynb](../notebooks/forecast_agents.ipynb) notebook runs a **multi-agent forecasting arena** (Claude, GPT, Gemini, Chronos, MLForecast, etc.). Agents produce **6-week point or quantile forecasts**; evaluation is **MASE/RMSE**, not inventory cost.

- **Same problem (retail sales), different objective**: Forecast accuracy (MASE/RMSE) vs inventory cost (holding + shortage).
- **Fair comparison**: To see if agent forecasts could have beaten us, we would (1) plug agent point or quantile forecasts into our SIP/newsvendor pipeline and compute 8-week cost, or (2) compare our backtest pinball/CRPS to agent MASE/RMSE on the same holdout. For inventory, **cost-based evaluation** and running agents through the same sim is the right comparison.
- **No implementation in this doc**: Only the evaluation plan; run one week or a subset of SKUs as an optional next step.

---

## References

- [L3_LEAD_TIME_ANALYSIS.md](L3_LEAD_TIME_ANALYSIS.md) — lead time bug and impact
- [backtesting_against_competition.md](backtesting_against_competition.md) — methodology and data
- [notebooks/Official_Benchmark.py](../notebooks/Official_Benchmark.py) — benchmark code
- [src/vn2/forecast/evaluation.py](../src/vn2/forecast/evaluation.py) — pinball, CRPS, cost-based metrics
- [src/vn2/analyze/model_eval.py](../src/vn2/analyze/model_eval.py) — cost-based model evaluation

