# Logic and Forecast Checklist

Use this checklist when validating the pipeline or preparing for backtests. No deep code audit—only “validate these areas.”

---

## Lead time and horizon

| Item | Status / notes |
|------|----------------|
| **L=3** | Competition rule: order at end of week X arrives at start of week X+3. We had L=2; fix in place. See [L3_LEAD_TIME_ANALYSIS.md](L3_LEAD_TIME_ANALYSIS.md). |
| **h=3 forecasts** | Competition needs demand for week t+3. We trained/selected on h=1, h=2; h=3 added via [scripts/regenerate_h3_forecasts.py](../scripts/regenerate_h3_forecasts.py) and `models/checkpoints_h3/`. |
| **First-arrival timing** | State propagation and SIP must use correct arrival week (order placed at end of t arrives start of t+3). |

---

## SURD and transforms

| Item | Status / notes |
|------|----------------|
| **SURD transform selection** | Docs mention possible “transform selection bugs.” Validate that per-SKU transform (e.g. log, none) is applied consistently in training and inference. |
| **Corrected SURD** | [scripts/regenerate_surd_transforms_corrected.py](../scripts/regenerate_surd_transforms_corrected.py) and `data/processed/surd_transforms.parquet`; use corrected transforms in backtest. |

---

## Stockout and demand

| Item | Status / notes |
|------|----------------|
| **Stockout censoring** | Demand during stockouts is censored (true demand ≥ observed sales). Don’t treat stockouts as true demand; use stockout-aware imputation or models (e.g. SLURP stockout-aware). |
| **Imputation** | [docs/STOCKOUT_IMPUTATION.md](STOCKOUT_IMPUTATION.md); ensure training targets and evaluation use consistent imputation. |

---

## Selector and evaluation

| Item | Status / notes |
|------|----------------|
| **Selector training** | Selector trained on historical (e.g. pinball or cost-based) with h=1/h=2; competition uses h=3. Consider re-training or re-evaluating selector with h=3. |
| **No future leakage** | Backtest and eval must only use data available at decision time (no future actuals in features or model choice). |

---

## Known issues (from docs)

- **Lead time**: L=2 vs L=3 — fixed; impact ~1.6%.
- **Transform selection bugs**: Referenced in [backtesting_against_competition.md](backtesting_against_competition.md); validate SURD usage.
- **Stockout censoring**: Model the censoring; see [L3_LEAD_TIME_ANALYSIS.md](L3_LEAD_TIME_ANALYSIS.md) and stockout imputation docs.

---

## Quick validation commands

```bash
# Check model/checkpoint coverage
python scripts/check_model_coverage.py --checkpoints-dir models/checkpoints --demand-path data/processed/demand_long.parquet

# Validate checkpoints (e.g. 12 folds)
python scripts/validate_checkpoints.py --checkpoints-dir models/checkpoints --expected-folds 12

# Full L=3 simulation (when sales files present)
python scripts/full_L3_simulation.py --initial-state "data/raw/Week 0 - 2024-04-08 - Initial State.csv" --sales-dir data/raw --weeks 1 2 3 4 5
```
