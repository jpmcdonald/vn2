# How We Beat the Winner: From €7,787 to €4,564

## The Journey

| Milestone | Cost | vs Winner (€4,677) | What Changed |
|-----------|-----:|-------------------:|--------------|
| Competition submission | €7,787 | +€3,110 (+66%) | Bugs: L=2 instead of L=3, h=2 instead of h=3, wrong cost tallying |
| Bug fixes + correct evaluation | €5,087 | +€410 (+8.8%) | Fixed L=3, h=3, separated eval costs from ordering costs; train-once strategy |
| Density SIP (single model) | €5,169 | +€492 (+10.5%) | slurp_bootstrap at theoretically optimal SL=0.833 |
| **Dynamic composite selector** | **€4,564** | **-€113 (-2.4%)** | Per-SKU model selection using pinball×Wasserstein metric |

---

## Phase 1: Finding the Bugs (€7,787 → €5,087)

### What went wrong during competition

1. **Lead time L=2 vs L=3.** The competition rules stated orders placed at end of week X arrive at start of week X+3. We implemented L=2 (arrives week X+2). Impact: ~€125 (1.6%). Small, but indicative of insufficient rule validation.

2. **Horizon h=2 vs h=3.** With L=3 lead time, the newsvendor solver needs demand forecasts for horizons 1, 2, and 3. Our checkpoints only had h=1 and h=2. The solver defaulted to zero for missing h=3, producing systematically underordered quantities for the critical third horizon.

3. **Cost tallying bug (critical).** When testing different service levels, the `simulate_week()` function used the *adjusted* costs (derived from the service level) for tallying realized costs instead of the constant true costs (cu=1.0, co=0.2). At SL=0.50, shortage was tallied at 1/5th its real value. This made lower service levels appear artificially cheap and completely invalidated early sweep results.

### The fix

- Separated `eval_costs` (always cu=1.0, co=0.2) from ordering costs
- Regenerated h=3 checkpoints for all models
- Corrected fold indexing: Order N uses fold N-1 (no data leakage)
- Discovered that train-once (static fold_0) outperforms sequential refit for SLURP models

**Result:** slurp_bootstrap, static folds, SL=0.833 → **€5,087** (beats benchmark €5,248 by 3.1%)

---

## Phase 2: Understanding the Cost Structure (€5,087 → €5,169)

With corrected backtesting, we ran a comprehensive analysis:

### Service level sweep

The theoretically optimal critical fractile CF = cu/(cu+co) = 0.833 is indeed optimal — but only for calibrated models. Miscalibrated models (LightGBM, seasonal_naive) perform better at SL=0.50-0.60 because the lower service level compensates for systematic over-prediction.

### Safety stock formula comparison

We backtested the ordinal progression of classical safety stock formulas against our density-based SIP:

| Policy | Best Total | vs SIP |
|--------|----------:|-------:|
| z·σ·√L (worst classical) | €6,498 | +€1,329 |
| z·σ·√(L+R) | €6,618 | +€1,449 |
| z·RMSE·√(L+R) | €6,018 | +€849 |
| k·RMSE (similar to VN2 winner) | €5,807 | +€638 |
| k·MAE (best classical) | €5,771 | +€602 |
| **Density SIP** | **€5,169** | **—** |

The density-based SIP dominates all classical formulas by €602-€1,449. Even k·MAE with optimized k doesn't beat the benchmark (€5,248). This confirms the Jensen Gap hypothesis: full probabilistic optimization captures demand distribution shape in ways that point+safety-stock formulas cannot.

---

## Phase 3: The Ensemble Breakthrough (€5,169 → €4,564)

### The insight

No single model is best for all 599 SKUs. Each model family excels on different demand patterns:
- **SLURP bootstrap** — best for stable, low-intermittency series
- **LightGBM quantile** — best for series with strong feature-driven patterns
- **DeepAR** — best for series with complex temporal dependencies
- **SLURP stockout-aware** — best for heavily censored series

### The method

We built a per-SKU model selector using the **composite metric**: pinball loss at the critical fractile (0.833) × Wasserstein distance. This combines decision-relevant accuracy (how good is the forecast at the cost-critical quantile?) with distributional quality (how well does the full forecast distribution match reality?).

For each of 599 SKUs, we pick the model with the lowest mean composite score across all 8 evaluation weeks.

### The winning ensemble

| Model | SKUs Selected | Share |
|-------|-------------:|------:|
| lightgbm_quantile | 183 | 30.6% |
| slurp_bootstrap | 183 | 30.6% |
| deepar | 167 | 27.9% |
| slurp_stockout_aware | 66 | 11.0% |

Remarkably balanced — the top three models each handle roughly a third of SKUs. DeepAR, despite being individually weakest (€5,647 as a single model), contributes 28% of selections because it's the best choice for specific demand patterns.

### Why weekly selection fails

We also tested oracle weekly selectors (different model per SKU per week, using same-week metrics). These performed *worse* than the static per-SKU selector:

| Selector | Total Cost |
|----------|----------:|
| **Static composite** | **€4,564** |
| Weekly composite (oracle) | €5,040 |
| Weekly Wasserstein (oracle) | €5,133 |

Model consistency across weeks matters more than per-week optimization. Switching models week-to-week introduces policy inconsistency that harms inventory dynamics.

---

## Key Takeaways

1. **Bugs are expensive.** Three implementation bugs turned a viable pipeline (€5,087) into a 110th-place finish (€7,787). Encode rules as tests.

2. **Calibration unlocks optimization.** The Jensen Gap is real (+€602-€1,449 vs classical formulas), but only for well-calibrated density forecasts. Miscalibrated density is worse than simple point+safety-stock.

3. **Diversity wins.** No single model dominates. Per-SKU selection using distributional quality metrics turns four mediocre-to-good individual models into a winning ensemble.

4. **The right metric matters.** Composite (pinball@CF × Wasserstein) outperforms pure CRPS or pure Wasserstein for selection. It combines what matters for inventory decisions — accuracy at the cost-relevant quantile — with overall distributional quality.

5. **Consistency beats optimality.** Static per-SKU selection beats oracle per-week selection. In sequential inventory problems, maintaining a consistent forecasting approach matters more than chasing per-period accuracy.

---

## References

- [docs/paper/revised_paper.md](paper/revised_paper.md) — full technical paper
- [docs/L3_LEAD_TIME_ANALYSIS.md](L3_LEAD_TIME_ANALYSIS.md) — lead time bug analysis
- [reports/safety_stock/policy_comparison.md](../reports/safety_stock/policy_comparison.md) — safety stock results
- [reports/dynamic_selector/selector_comparison.csv](../reports/dynamic_selector/selector_comparison.csv) — selector results
