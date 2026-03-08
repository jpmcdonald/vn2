# VN2 Inventory Planning Challenge: Lessons Learned

**Data science meetup — 15–20 min**

From 110th place to beating the winner: what we built, what broke, and what we learned about probabilistic inventory optimization.

---

## 1. Problem and Goal

- **Competition**: VN2 Inventory Planning (DataSource.AI). Minimize total cost over **8 weeks** (6 ordering rounds + 2 delivery-only weeks).
- **Setup**: 599 SKUs (Store × Product); **L=3** lead time (order end of week X → arrives start of week X+3); **asymmetric costs** (shortage €1/unit, holding €0.2/unit).
- **Critical fractile**: CF = cu/(cu+co) = 1.0/1.2 = 0.833 — the theoretically optimal quantile for the newsvendor problem.

---

## 2. What We Built

- **Quantile forecasting**: 5 model families (SLURP bootstrap, SLURP stockout-aware, LightGBM quantile, DeepAR, seasonal naive) producing full predictive distributions at 13 quantile levels.
- **SURD**: Variance-stabilizing transforms per series for better calibration.
- **Stockout awareness**: Censoring-aware models that preserve uncertainty from stockout periods.
- **SIP/newsvendor**: Quantiles → PMF → sequential newsvendor optimization with integer Q and L=3.
- **Dynamic model selector**: Per-SKU model chosen by composite metric (pinball@CF × Wasserstein distance).

---

## 3. Result: From 110th to 1st

| Milestone | Cost | vs Winner (€4,677) |
|-----------|-----:|-------------------:|
| Competition submission (bugged) | €7,787 | +66% |
| After bug fixes | €5,087 | +8.8% |
| First ensemble (4 models) | €4,564 | -2.4% |
| **SURD-expanded ensemble (9 models)** | **€4,487** | **-4.1%** |
| Official benchmark | €5,248 | +12.2% |
| Competition winner | €4,677 | — |

---

## 4. What Broke: Three Bugs That Cost Us €2,700

1. **Lead time L=2 vs L=3** (~€125): Misread the rules. Orders arrived one week early in our simulation.
2. **Horizon h=2 vs h=3** (~€500): Missing third-horizon forecasts meant the solver under-ordered for the critical lead-time window.
3. **Cost tallying bug** (~€2,000): Service-level experiments used adjusted costs for evaluation, making lower SLs appear 5× cheaper. Completely invalidated all sweep results.

**Lesson**: Encode competition rules as unit tests. Run end-to-end backtests on day one.

---

## 5. Why Formulas Fail: Safety Stock Comparison

We tested the ordinal progression of classical safety stock formulas against our density-based approach:

| Policy | Total Cost | vs Density SIP |
|--------|----------:|---------------:|
| z·σ·√L (worst) | €6,498 | +€1,329 |
| z·σ·√(L+R) | €6,618 | +€1,449 |
| z·RMSE·√(L+R) | €6,018 | +€849 |
| k·RMSE (≈ VN2 winner approach) | €5,807 | +€638 |
| k·MAE (best classical) | €5,771 | +€602 |
| **Density SIP (ours)** | **€5,169** | **—** |

The full probabilistic approach outperforms every classical formula by €602-€1,449. Even k·MAE with optimized k (€5,771) can't beat the benchmark (€5,248).

**Why**: Classical formulas assume demand follows a known distribution (usually normal). Our density SIP uses the *actual* predicted distribution shape, capturing asymmetry, intermittency, and fat tails that Gaussian assumptions miss.

---

## 6. The Synergy of Diversity: 9-Model Ensemble

No single model wins everywhere. After fixing the SURD bug (variance-stabilizing transforms were silently defaulting to identity), the winning selector draws from 9 model variants:

```
Model Selection Distribution (599 SKUs)
┌───────────────────────────────┬─────┬───────┐
│ slurp_bootstrap               │ 159 │ 26.5% │
│ deepar                        │ 151 │ 25.2% │
│ lightgbm_quantile             │  89 │ 14.9% │
│ lightgbm_surd            [S]  │  80 │ 13.4% │
│ slurp_stockout_aware          │  41 │  6.8% │
│ slurp_surd_stockout_aware [S] │  34 │  5.7% │
│ deepar_surd              [S]  │  33 │  5.5% │
│ slurp_surd               [S]  │  12 │  2.0% │
└───────────────────────────────┴─────┴───────┘
  [S] = SURD variant (26.5% of total selections)
```

SURD models account for **159 of 599** SKU selections. lightgbm_surd alone captures 80 SKUs where variance-stabilized LightGBM outperforms all other models. The composite selector achieves **€4,487** -- €77 better than the non-SURD ensemble (€4,564) and €190 better than the competition winner.

**Selection metric**: pinball(CF=0.833) x Wasserstein distance.

---

## 7. Density vs Point: The Jensen Gap

Jensen's inequality says: optimizing over the full distribution beats optimizing a point forecast, *if* the distribution is calibrated.

| Model Calibration | Jensen Gap | Outcome |
|-------------------|-----------|---------|
| Well-calibrated (SLURP) | +€815 | Density wins |
| Miscalibrated (LightGBM) | -€827 | Point wins |
| Catastrophically biased (seasonal naive) | -€1,026 | Point wins |

**Key insight**: Miscalibrated density forecasts are *worse* than point forecasts for decision-making. Calibration is the binding constraint, not optimization sophistication.

---

## 8. Consistency Beats Optimality

We tested two selection strategies:
- **Static**: same model per SKU across all weeks → **€4,487**
- **Oracle weekly**: best model per SKU per week (using actual outcomes) → €4,991

Even with perfect hindsight for weekly selection, the static selector wins. Switching models week-to-week breaks inventory dynamics -- inconsistent ordering policies create whiplash in stock levels.

---

## 9. Stockouts: Censoring Matters

Demand during stockouts is **censored** — true demand ≥ observed sales. Models that ignore censoring underestimate demand for frequently-out-of-stock SKUs.

- slurp_stockout_aware wins at conservative SLs (0.20-0.70): saves €34-€220
- slurp_bootstrap wins at SL=0.833: tighter bands avoid over-ordering
- **Lesson**: Censoring awareness helps most when you're under-ordering; it can slightly hurt when you're already ordering aggressively.

---

## 10. What We'd Do Differently

1. **Test end-to-end on day one** — a single 1-week simulation would have caught all three bugs
2. **Benchmark early** — run the official benchmark against our pipeline after Week 1 actuals arrive
3. **Train multiple model families from the start** — no single model dominates
4. **Use distributional metrics for selection** — CRPS and Wasserstein, not just pinball at one quantile
5. **Respect the Jensen Gap's conditions** — calibrate before optimizing; miscalibrated density is dangerous

---

## Summary Slide

```
┌────────────────────────────────────────────┐
│   VN2 Inventory Planning: Key Numbers      │
│                                            │
│   Competition submission:     €7,787       │
│   After fixing 4 bugs:       €5,087       │
│   Best classical formula:    €5,771       │
│   Official benchmark:        €5,248       │
│   Competition winner:        €4,677       │
│   Our post-comp ensemble:    €4,487       │
│                                            │
│   Density SIP vs best classical: -€602     │
│   Ensemble vs single model:     -€682     │
│   Ensemble vs winner:           -€190     │
│   SURD bug fix contribution:     -€77     │
└────────────────────────────────────────────┘
```

---

## References

- [HOW_WE_BEAT_THE_WINNER.md](HOW_WE_BEAT_THE_WINNER.md) — full journey narrative
- [paper/revised_paper.md](paper/revised_paper.md) — technical paper with all evidence
- [L3_LEAD_TIME_ANALYSIS.md](L3_LEAD_TIME_ANALYSIS.md) — lead time bug analysis
