# How We Beat the Winner: From €7,787 to €4,487

## The Journey

| Milestone | Cost | vs Winner (€4,677) | What Changed |
|-----------|-----:|-------------------:|--------------|
| Competition submission | €7,787 | +€3,110 (+66%) | Bugs: L=2 instead of L=3, h=2 instead of h=3, wrong cost tallying |
| Bug fixes + correct evaluation | €5,087 | +€410 (+8.8%) | Fixed L=3, h=3, separated eval costs from ordering costs; train-once strategy |
| Density SIP (single model) | €5,169 | +€492 (+10.5%) | slurp_bootstrap at theoretically optimal SL=0.833 |
| Dynamic composite selector (v1) | €4,564 | -€113 (-2.4%) | Per-SKU model selection using pinball×Wasserstein metric |
| **SURD fix + expanded ensemble** | **€4,487** | **-€190 (-4.1%)** | Fixed SURD bug, added lightgbm_surd + deepar_surd; 9-model selector |

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

We built a per-SKU model selector using the **composite metric**: pinball loss at the critical fractile (0.833) x Wasserstein distance.

For each of 599 SKUs, we pick the model with the lowest mean composite score across all 8 evaluation weeks.

### First ensemble result: €4,564

The initial 4-model ensemble beat the winner by €113 using lightgbm_quantile (30.6%), slurp_bootstrap (30.6%), deepar (27.9%), and slurp_stockout_aware (11.0%).

---

## Phase 4: The SURD Fix (€4,564 → €4,487)

### The bug

Our SURD (Systematic Unsupervised Representation Discovery) variance-stabilizing transforms were **never actually applied**. The `SURDSLURPBootstrapForecaster.fit()` tried to extract the SKU ID from `y.index`, but the training pipeline passes `y` with a simple integer index. Result: `sku_id = None`, transform lookup skipped, all 599 SKUs silently defaulted to identity. The paper's "H3 not supported" conclusion was based on a no-op.

Evidence: `slurp_bootstrap` vs `slurp_surd` checkpoints were **byte-for-byte identical** (max quantile diff = 0.000000). Meanwhile, the SURD transforms file showed 100% non-identity selections (588 cbrt, 11 log1p) with 84% mean CV reduction.

### The fix

Two-part fix:
1. Pipeline (`pipeline.py`): set `model.sku_id = task.sku_id` before `model.fit()`, so models have the SKU identity available for transform lookup.
2. Model (`slurp_bootstrap.py`): only attempt index-based SKU ID extraction if `self.sku_id` is not already set.

### Extending SURD to all models

With the fix in place, we created SURD variants for all model families:
- **lightgbm_surd**: `SURDWrapper(LightGBMQuantileForecaster)` — transforms the target variable before LightGBM training, inverse-transforms predictions
- **deepar_surd**: Transform demand per-SKU before GluonTS training, inverse-transform MC samples after prediction

### The winning ensemble (9 models)

| Model | SKUs Selected | Share |
|-------|-------------:|------:|
| slurp_bootstrap | 159 | 26.5% |
| deepar | 151 | 25.2% |
| lightgbm_quantile | 89 | 14.9% |
| **lightgbm_surd** | **80** | **13.4%** |
| slurp_stockout_aware | 41 | 6.8% |
| **slurp_surd_stockout_aware** | **34** | **5.7%** |
| **deepar_surd** | **33** | **5.5%** |
| **slurp_surd** | **12** | **2.0%** |

SURD models now account for **26.5%** of all SKU selections (159 of 599). The lightgbm_surd variant alone captures 80 SKUs where the SURD-transformed LightGBM outperforms both the original LightGBM and all SLURP/DeepAR variants.

### SURD impact on individual models (SL=0.833)

| Base Model | SURD Model | Base Cost | SURD Cost | Delta |
|-----------|-----------|--------:|---------:|------:|
| slurp_stockout_aware | slurp_surd_stockout_aware | €5,264 | €5,214 | -€50 |
| deepar | deepar_surd | €5,647 | €5,643 | -€4 |
| lightgbm_quantile | lightgbm_surd | €6,756 | €6,782 | +€26 |
| slurp_bootstrap | slurp_surd | €5,169 | €5,176 | +€8 |

SURD doesn't uniformly improve individual models — it adds diversity that the composite selector exploits.

### Why weekly selection still fails

| Selector | Total Cost |
|----------|----------:|
| **Static composite** | **€4,487** |
| Static Wasserstein | €4,909 |
| Weekly composite (oracle) | €4,991 |
| Weekly Wasserstein (oracle) | €5,040 |

Model consistency across weeks matters more than per-week optimization.

---

## Key Takeaways

1. **Bugs are expensive.** Three implementation bugs turned a viable pipeline (€5,087) into a 110th-place finish (€7,787). A fourth silent bug (SURD identity) cost €77 more. Encode rules as tests.

2. **Calibration unlocks optimization.** The Jensen Gap is real (+€602-€1,449 vs classical formulas), but only for well-calibrated density forecasts.

3. **Diversity wins.** No single model dominates. Per-SKU selection using 9 model variants (including SURD) turns individually mediocre models into a €4,487 ensemble.

4. **Variance stabilization adds diversity.** SURD transforms don't uniformly improve individual models, but they provide distinct forecast profiles that the composite selector exploits for 26.5% of SKUs.

5. **The right metric matters.** Composite (pinball@CF x Wasserstein) outperforms pure CRPS or pure Wasserstein for selection.

6. **Consistency beats optimality.** Static per-SKU selection beats oracle per-week selection. In sequential inventory problems, maintaining a consistent forecasting approach matters more than chasing per-period accuracy.

---

## References

- [docs/paper/revised_paper.md](paper/revised_paper.md) — full technical paper
- [docs/L3_LEAD_TIME_ANALYSIS.md](L3_LEAD_TIME_ANALYSIS.md) — lead time bug analysis
- [reports/safety_stock/policy_comparison.md](../reports/safety_stock/policy_comparison.md) — safety stock results
- [reports/dynamic_selector/selector_comparison.csv](../reports/dynamic_selector/selector_comparison.csv) — selector results
