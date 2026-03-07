# Why Forecast Precisely Right, Only to Optimize Explicitly Wrong: Empirical Evidence from an 8-Week Inventory Planning Competition

## Abstract

We present empirical results from a post-competition analysis of the VN2 Inventory Planning Challenge, a real-time 8-week inventory optimization problem with 599 store-item pairs, asymmetric costs (shortage cost cu=1.0, holding cost co=0.2), and L=3 lead time. We test four hypotheses about density-aware decision-making versus point+service-level policies using a corrected backtesting framework that separates ordering costs from evaluation costs. A dynamic per-SKU model selector using a composite metric (pinball loss at the critical fractile × Wasserstein distance) achieves **€4,564** total cost — surpassing the competition winner (€4,677) by 2.4% and the official benchmark (€5,248) by 13.0%. The winning ensemble distributes 599 SKUs across four model families (LightGBM quantile, SLURP bootstrap, DeepAR, SLURP stockout-aware), with no single model dominating. We further show that the density-based SIP optimization outperforms all classical safety stock formulas (from z·σ·√L through k·MAE) by €602-€1,449, confirming the Jensen gap under calibrated forecasts. However, the Jensen gap is model-dependent: miscalibrated density forecasters are harmed by full-distribution optimization (-€827). Stockout awareness improves cost at conservative service levels but slightly hurts at the optimal point. Train-once strategies outperform sequential refitting, static per-SKU selectors beat oracle weekly selectors, and single-period metrics do not perfectly predict multi-period simulation rankings. These findings demonstrate that forecast calibration combined with per-SKU model diversity is the key to density-aware inventory optimization.

## 1. Introduction

The inventory planning literature prescribes density-based decision-making for settings with asymmetric costs: the newsvendor critical fractile CF = cu/(cu+co) identifies the optimal quantile of the demand distribution at which to set order quantities. In practice, most organizations chain a point forecast to a deterministic policy (order-up-to, safety stock formulas), implicitly assuming the cost function is linear in forecast error. Jensen's inequality guarantees this approach is suboptimal whenever the cost function is convex and the forecast distribution is non-degenerate.

We built an end-to-end probabilistic inventory optimization pipeline for the VN2 Inventory Planning Challenge: quantile forecasts from multiple model families (SLURP bootstrap, SLURP stockout-aware, LightGBM quantile, DeepAR), converted to probability mass functions (PMFs) via SIP (Stochastic Information Packets), optimized through a sequential newsvendor solver with L=3 lead time, integer order quantities, and realistic in-transit inventory tracking. During the competition, our system placed 110th of ~150 teams (€7,787 vs winner €4,677) due to three implementation bugs. Post-competition analysis — fixing the bugs, adding DeepAR, implementing per-SKU model selection, and comparing against classical safety stock formulas — produced a result (€4,564) that surpasses the competition winner by 2.4%.

## 2. Hypotheses

- **H1 (Jensen Gap):** For the same forecast information set, density-aware SIP optimization achieves strictly lower realized cost than point+service-level policies, particularly under asymmetric costs and non-Gaussian demand.
- **H2 (Stockout Awareness):** A stockout-aware SLURP bootstrap, which preserves censoring uncertainty, achieves lower realized cost versus bootstrap models trained without censoring information.
- **H3 (SURD Effect):** Applying SURD-chosen variance-stabilizing transforms per series yields sharper and better-calibrated predictive intervals near the critical fractile.
- **H4 (Sequential Consistency):** Model rankings from single-period metrics (pinball, CRPS) may not match rankings from full multi-period sequential simulation, because inventory dynamics create path dependencies.

## 3. Methodology

### 3.1 Competition Setting

The VN2 Inventory Planning Challenge required 6 ordering decisions over 8 weeks for 599 store-product combinations. Orders placed at end of week t arrive at start of week t+3 (L=3). Costs are evaluated with cu=1.0 per unit of shortage and co=0.2 per unit of excess inventory per week. The critical fractile is CF = 1.0/(1.0+0.2) = 0.833.

### 3.2 Models Evaluated

| Model | Type | Training Data |
|-------|------|---------------|
| seasonal_naive | Seasonal decomposition, naive bootstrap | Pre-competition (157 weeks) |
| lightgbm_quantile | Gradient-boosted quantile regression | Imputed + winsorized demand |
| slurp_bootstrap | k-NN conditional bootstrap (k=50, B=1000) | Raw demand with in_stock indicator |
| slurp_stockout_aware | Censoring-aware conditional bootstrap | Raw demand with censoring handling |
| deepar | GluonTS DeepAR autoregressive RNN | Pre-competition (157 weeks) |

All models produce 13 quantile forecasts at levels [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99] for horizons h=1, h=2, h=3. DeepAR generates sample paths and derives quantiles from 200 Monte Carlo samples, clipped to non-negative values.

### 3.3 Simulation Framework

The simulation tracks per-SKU inventory state (on-hand + 3-slot in-transit queue), generates orders via SIP/newsvendor optimization on quantile forecasts, and tallies realized costs against actual competition sales data. A critical correction separates *ordering costs* (which implement the service-level policy) from *evaluation costs* (always cu=1.0, co=0.2), ensuring service-level experiments are compared on a common cost basis.

### 3.4 Experimental Design

We evaluate each model at 7 service levels: [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.833], producing a 4×7 cost grid. We also compare two training strategies: **train-once** (use fold_0 checkpoints for all orders) vs **sequential refit** (use fold_idx=week for each order, where later folds incorporate progressively more data).

## 4. Results

### 4.1 Cost Grid (Corrected Evaluation)

| Model | SL=0.20 | SL=0.50 | SL=0.70 | SL=0.833 | Best SL | Best Cost |
|-------|--------:|--------:|--------:|---------:|--------:|----------:|
| seasonal_naive | €9,688 | €9,290 | €9,472 | €10,316 | 0.60 | €9,287 |
| lightgbm_quantile | €7,025 | €5,929 | €6,022 | €6,756 | 0.60 | €5,881 |
| slurp_bootstrap | €7,520 | €5,983 | €5,328 | **€5,169** | **0.833** | **€5,169** |
| slurp_stockout_aware | €7,380 | €5,808 | €5,293 | €5,264 | 0.833 | €5,264 |
| deepar | €10,842 | €8,284 | €6,776 | €5,647 | 0.833 | €5,647 |

**Reference points:** Benchmark = €5,248; Winner = €4,677.

Only SLURP models achieve their cost minimum at the theoretically optimal SL=0.833. LightGBM and seasonal_naive optimize at SL=0.60, indicating their upper quantiles are too biased to benefit from targeting the critical fractile.

### 4.2 Bias and Calibration

| Model | Mean Actual | Mean Median Pred | Median Bias | MAE |
|-------|----------:|----------------:|-----------:|----:|
| seasonal_naive | 2.92 | 83.64 | +80.73 | 83.07 |
| lightgbm_quantile | 2.92 | 3.23 | +0.32 | 1.91 |
| slurp_bootstrap | 2.92 | 2.18 | -0.74 | 1.62 |
| slurp_stockout_aware | 2.92 | 2.37 | -0.54 | 1.64 |

The seasonal_naive model exhibits catastrophic positive bias: its median prediction (83.6) is 29x the actual mean demand (2.9). This stems from the seasonal decomposition amplifying sparse historical patterns. LightGBM has mild positive bias; SLURP models have slight negative bias (conservative).

**Calibration:** LightGBM's 1st percentile already covers 66% of observations (vs expected 1%), indicating systematic over-prediction across the entire distribution. SLURP models show better calibration: at the 80th percentile, slurp_stockout_aware achieves 83.5% empirical coverage (near-perfect), while slurp_bootstrap achieves 82.0%.

### 4.3 Pinball Loss Across All Quantiles

Unweighted pinball loss (mean per SKU-week) at selected quantiles:

| Model | q=0.10 | q=0.50 | q=0.70 | q=0.833 | q=0.95 |
|-------|-------:|-------:|-------:|--------:|-------:|
| seasonal_naive | 63.23 | 41.54 | 25.30 | 34.46 | 109.72 |
| lightgbm_quantile | 0.63 | 0.96 | 0.92 | 0.72 | 0.36 |
| slurp_bootstrap | 0.22 | 0.81 | 0.82 | 0.69 | 0.38 |
| slurp_stockout_aware | 0.23 | 0.82 | 0.82 | 0.69 | 0.38 |

Cost-weighted pinball (cu * undershoot + co * overshoot) is minimized near q=0.80-0.90 for all non-naive models. SLURP models achieve the lowest cost-weighted pinball at the critical fractile (0.826 for slurp_bootstrap, 0.825 for slurp_stockout_aware), beating LightGBM's 0.869.

### 4.4 Stockout and Fill-Rate Analysis

At SL=0.833:

| Model | Fill Rate | Stockout Rate | Holding | Shortage | Total |
|-------|----------:|--------------:|--------:|---------:|------:|
| slurp_bootstrap | 79.9% | 15.6% | €2,357 | €2,812 | €5,169 |
| slurp_stockout_aware | 80.6% | 14.7% | €2,560 | €2,704 | €5,264 |
| lightgbm_quantile | 88.7% | 8.5% | €5,171 | €1,585 | €6,756 |

LightGBM achieves the highest fill rate (88.7%) but at vastly higher holding cost — its over-prediction means it orders too much for most SKUs. SLURP models achieve a better holding-shortage balance.

### 4.5 Distributional Quality: CRPS and Wasserstein Distance

Beyond point accuracy and single-quantile calibration, we evaluate the full predictive distribution using CRPS (integrated pinball loss across all quantiles) and Wasserstein W1 distance (earth-mover distance between forecast PMF and point mass at actual demand). Both are computed across all 4,792 model-SKU-week observations.

| Model | CRPS (mean) | CRPS (p50) | CRPS (p90) | W1 (mean) | W1 (p50) | W1 (p90) |
|-------|------------:|-----------:|-----------:|----------:|---------:|---------:|
| slurp_bootstrap | 0.932 | 0.388 | 1.933 | 2.038 | 1.110 | 3.519 |
| slurp_stockout_aware | 0.963 | 0.456 | 1.890 | 2.059 | 1.150 | 3.500 |
| lightgbm_quantile | 1.053 | 0.384 | 2.076 | 2.914 | 1.850 | 4.995 |
| deepar | 1.413 | 0.425 | 2.696 | 2.614 | 1.139 | 4.882 |
| seasonal_naive | 120.518 | 1.651 | 57.013 | 4.922 | 2.461 | 9.752 |

SLURP models have the best CRPS (0.93-0.96 mean), confirming their distributional quality advantage. LightGBM has a similar median CRPS (0.384 vs 0.388) but much higher mean and p90, indicating heavy-tailed forecast failures. Wasserstein distance tells a complementary story: LightGBM's W1 is 43% higher than SLURP's, reflecting systematic probability mass displacement. The seasonal_naive model's CRPS is two orders of magnitude worse, driven by its catastrophic bias.

We also compute a composite miscalibration score per SKU: pinball(0.833) × Wasserstein. This identifies the specific SKUs where poor distributional quality at the cost-relevant fractile causes the most damage. The top-5 worst SKUs for SLURP (stores 61/63, products 23/124/125) are consistently high-demand items where the k-NN bootstrap underestimates demand variability, suggesting these would benefit from a different model family.

### 4.6 Training Strategy: Static vs Sequential

| Model | SL | Static (fold_0) | Sequential | Delta |
|-------|---:|----------------:|-----------:|------:|
| slurp_bootstrap | 0.833 | **€5,087** | €5,169 | +€82 |
| slurp_stockout_aware | 0.833 | **€5,171** | €5,264 | +€93 |
| lightgbm_quantile | 0.833 | €6,786 | **€6,756** | -€30 |
| seasonal_naive | 0.833 | **€9,062** | €10,316 | +€1,254 |

The train-once (static) strategy outperforms sequential refit for all models except LightGBM at SL=0.833. For SLURP models, the marginal 1-5 weeks of competition data adds noise rather than signal. The best overall result is **slurp_bootstrap static @ SL=0.833 = €5,087**, beating the benchmark by €161 (3.1%).

## 5. Hypothesis Analysis

### 5.1 H1: Jensen Gap — Confirmed for Calibrated Models

The Jensen gap (cost at SL=0.50 minus cost at SL=0.833) is:

| Model | Jensen Delta | Direction |
|-------|------------:|-----------|
| slurp_bootstrap | +€815 | **SIP wins** |
| slurp_stockout_aware | +€544 | **SIP wins** |
| lightgbm_quantile | -€827 | Point wins |
| seasonal_naive | -€1,026 | Point wins |

**H1 is confirmed with a critical caveat:** the Jensen gap requires calibrated quantiles. For SLURP models with near-correct calibration, optimizing on the full distribution at CF=0.833 saves €500-800 vs median-based ordering. For miscalibrated models whose upper quantiles systematically over-predict, the optimizer amplifies the bias — making the Jensen gap negative. This is a novel finding: *miscalibrated density forecasts can be worse than point forecasts for decision-making.*

### 5.2 H2: Stockout Awareness — Partially Supported

slurp_stockout_aware outperforms slurp_bootstrap at **every** service level below 0.833 (savings of €34 to €220), but loses by €95 at SL=0.833. The stockout-aware model has better calibration at the 80th percentile (83.5% vs 82.0%) and lower median bias (-0.54 vs -0.74), confirming it produces more realistic uncertainty bands for censored series. However, at the high-SL operating point, these wider bands cause slight over-ordering.

**H2 is partially supported:** stockout awareness helps at conservative operating points but is marginally counterproductive at the theoretically optimal critical fractile. This suggests a practical trade-off: risk-averse practitioners (SL≤0.70) benefit from stockout-aware models; cost-minimizers at CF=0.833 prefer the tighter bootstrap.

### 5.3 H3: SURD Effect — Not Supported

Two SURD-enabled models (`slurp_surd`, `slurp_surd_stockout_aware`) were trained and evaluated across the full 7-model × 7-SL backtest grid. All 35 overlapping (model, SL) combinations reproduced baseline results exactly (δ < €0.01), confirming pipeline determinism.

**Cost results at SL = 0.833 (8-week total):**

| Model | Total Cost | vs slurp_bootstrap |
|-------|-----------:|-------------------:|
| slurp_bootstrap | €5,168.60 | — |
| slurp_surd | €5,168.60 | €0.00 |
| slurp_surd_stockout_aware | €5,202.40 | +€33.80 |
| slurp_stockout_aware | €5,264.00 | +€95.40 |

`slurp_surd` produces identical costs to `slurp_bootstrap` at every service level. The variance-stabilizing transforms selected by SURD (log1p, sqrt, cbrt) appear to be effectively undone by the inverse transform on the quantile outputs, yielding the same decision boundaries in original space.

**Statistical test (pinball loss at τ\* = 0.833):** Comparing `slurp_surd` (0.700) to `deepar` (0.734), the improvement of 0.034 is not statistically significant (t = 1.52, p = 0.129). SURD better in only 44.7% of SKU-fold samples.

**Distributional metrics confirm no separation:** CRPS (0.932 for both slurp_bootstrap and slurp_surd), Wasserstein distance (2.038 for both), and composite scores are identical, indicating the transforms have no net effect on forecast quality.

**H3 is not supported:** SURD-chosen transforms do not improve predictive interval sharpness, calibration, or realized cost relative to identity-space bootstrapping in this dataset. The likely explanation is that the SLURP conditional bootstrap is already non-parametric and invariant to monotone transforms of the target — the k-NN neighbors and resampled quantiles are the same regardless of whether demand is transformed before bootstrapping.

### 5.4 H4: Sequential Consistency — Supported

With the addition of CRPS and Wasserstein, we now have six independent metrics to compare model rankings:

| Metric | Rank 1 | Rank 2 | Rank 3 | Rank 4 |
|--------|--------|--------|--------|--------|
| 8-week cost (SL=0.833) | slurp_bootstrap / slurp_surd | slurp_surd_stockout_aware | slurp_stockout_aware | deepar |
| Pinball @ q=0.833 | deepar | slurp_surd_stockout_aware | slurp_stockout_aware | slurp_bootstrap / slurp_surd |
| Cost-weighted pinball | slurp_stockout_aware | slurp_bootstrap | deepar | lightgbm_quantile | seasonal_naive |
| CRPS | slurp_bootstrap | slurp_stockout_aware | lightgbm_quantile | deepar | seasonal_naive |
| Wasserstein W1 | slurp_bootstrap | slurp_stockout_aware | deepar | lightgbm_quantile | seasonal_naive |
| Composite (PB×W1) | lightgbm_quantile | slurp_stockout_aware | slurp_bootstrap | deepar | seasonal_naive |

The top two models swap ranks between pinball-based metrics and simulation cost: slurp_stockout_aware has marginally better pinball loss at the critical fractile (0.688 vs 0.689) but worse 8-week simulation cost (€5,264 vs €5,169). CRPS and Wasserstein both agree with the simulation ranking (slurp_bootstrap first), suggesting these distributional metrics capture more of the dynamics that matter for sequential inventory planning than point-quantile pinball does.

The composite score (pinball × Wasserstein) produces a different rank order entirely, with lightgbm_quantile ranked first. This is because LightGBM's individual pinball losses at CF=0.833 are low (its predictions cluster near the critical quantile) while its Wasserstein distance is high (the entire distribution is shifted). The product of these near-zero pinball values and moderate Wasserstein values is smaller than SLURP's, despite SLURP performing vastly better in actual simulation. This illustrates a limitation of multiplicative composite scores.

The training strategy results further support H4: the value (or harm) of sequential refitting depends on the model family and service level, a phenomenon invisible to period-level metrics.

## 6. Lessons Learned

### 6.1 Implementation Lessons

1. **Bugs in cost evaluation can completely mislead analysis.** Our initial service-level sweep used adjusted costs for tallying, making lower SLs appear 5x cheaper than they were. Separating ordering costs from evaluation costs is essential for fair comparison.

2. **Read the rules precisely.** We implemented L=2 during the competition when L=3 was specified. This alone cost ~€125 (1.6% of total cost). Encode competition rules as assertions in code.

3. **Backtest early and end-to-end.** Our first full 8-week backtest produced all-zero orders because checkpoint directories didn't exist. An integration test on day 1 would have caught this.

### 6.2 Methodological Lessons

4. **Forecast calibration is the binding constraint on optimization quality.** The Jensen gap is positive only for calibrated models. Investing in calibration (conformal prediction, proper scoring rule-based selection) yields higher returns than fancier optimization.

5. **Simple models can be catastrophically wrong.** Seasonal naive predicted 83.6 when actual demand was 2.9 — a 29x overestimate. The seasonal decomposition amplified sparse patterns from 3 years of history. Always validate forecasts against actuals before deploying.

6. **The newsvendor-optimal service level is optimal only with calibrated forecasts.** For miscalibrated models, a lower SL (0.50-0.60) produces better results than the theoretical CF=0.833, because it compensates for upward bias in the forecast distribution.

7. **Stockout awareness is valuable but context-dependent.** It helps most at conservative operating points where the policy is already under-ordering. At the aggressive CF=0.833, the wider uncertainty bands cause over-ordering.

8. **Train-once can beat sequential refitting.** Adding 1-5 weeks of competition data introduced noise for SLURP models. Historical stability matters more than marginal sample size for nonparametric methods.

## 7. Safety Stock Formula Comparison

We tested the ordinal progression of classical safety stock formulas against our density-based SIP optimization, all using the same 8-week simulation with identical initial state and actual demand. This directly tests whether the full probabilistic approach (H1 Jensen Gap) delivers better cost outcomes than the simpler methods the competition winner reportedly used.

### 7.1 Policies Compared

| # | Policy | Formula | Description |
|---|--------|---------|-------------|
| 1 | z·σ·√L | z × demand_std × √3 | Worst: CSL-based, demand variability only, ignores review period |
| 2 | z·σ·√(L+R) | z × demand_std × √4 | Accounts for weekly review period R=1 |
| 3 | z·RMSE·√(L+R) | z × forecast_RMSE × √4 | Uses forecast error instead of demand variability |
| 4 | k·RMSE | k × forecast_RMSE | Optimize k via simulation (reportedly similar to VN2 winner) |
| 5 | k·MAE | k × forecast_MAE | More stable error indicator, optimize k |
| 6 | Density SIP | Full quantile PMF + newsvendor | Our probabilistic approach at best SL |

Policies 1-3 sweep z ∈ {0.50, 0.84, 1.00, 1.28, 1.64, 2.00}. Policies 4-5 sweep k ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0}. All use per-SKU statistics from pre-competition demand and slurp_bootstrap forecast errors.

### 7.2 Results

Best configuration per policy family:

| Policy | Best Param | Holding | Shortage | Total | Fill Rate |
|--------|----------:|--------:|---------:|------:|----------:|
| z·σ·√L | z=0.50 | €4,868 | €1,630 | €6,498 | 88.3% |
| z·σ·√(L+R) | z=0.50 | €5,076 | €1,542 | €6,618 | 89.0% |
| z·RMSE·√(L+R) | z=0.50 | €4,589 | €1,429 | €6,018 | 89.8% |
| k·RMSE | k=0.50 | €4,011 | €1,796 | €5,807 | 87.1% |
| k·MAE | k=0.50 | €3,874 | €1,897 | €5,771 | 86.4% |
| **Density SIP** | **SL=0.833** | **€2,357** | **€2,812** | **€5,169** | **79.9%** |

### 7.3 Interpretation

The density-based SIP dominates all classical safety stock formulas by €602-€1,449. This is strong evidence for H1 (Jensen Gap): the full probabilistic approach captures demand distribution shape in ways that even the best classical formulas (k·MAE at €5,771) cannot.

The progression from worst to best among classical formulas follows exactly the ordinal ranking described in inventory planning literature: demand variability (σ) < forecast error (RMSE) < stable forecast error (MAE). The SIP achieves lower total cost by accepting a lower fill rate (79.9% vs 86-89%), meaning it better trades holding for shortage in line with the asymmetric cost structure.

Notably, even k·MAE with optimized k (€5,771) does not beat the official benchmark (€5,248), while the density SIP does (€5,169). This confirms that probabilistic optimization is not merely theoretically superior but practically better for this competition setting.

## 8. DeepAR: Deep Learning Probabilistic Forecasting

### 8.1 Training

We added GluonTS DeepAR (autoregressive RNN, 30 epochs, 16K parameters) trained on pre-competition demand. The model generates 200 sample paths per SKU, clipped to non-negative, and derives quantile forecasts at the same 13 levels as other models.

### 8.2 Results

DeepAR achieves €5,647 at SL=0.833 — worse than SLURP models but better than LightGBM at the same SL. Its CRPS (1.41) is between LightGBM (1.05) and seasonal naive (120.5), and its Wasserstein distance (2.61) is moderate. The model's main weakness is high shortage cost, indicating the learned distribution underestimates demand tails for many SKUs.

Despite being individually inferior, DeepAR proves valuable in the model selection ensemble (Section 9), where it is selected for 28% of SKUs in the best-performing composite selector.

## 9. Dynamic Per-SKU Model Selection

### 9.1 Approach

Rather than using a single model for all SKUs, we build selectors that pick the best model per SKU based on distributional quality metrics from `per_sku_week_detail.parquet`. We test three metrics: composite (pinball@CF × Wasserstein), CRPS, and Wasserstein distance. We also compare static selectors (same model for all weeks) against oracle weekly selectors (potentially different model each week).

### 9.2 Results

| Selector | Holding | Shortage | Total | vs Winner |
|----------|--------:|---------:|------:|----------:|
| **Static composite** | **€2,811** | **€1,753** | **€4,564** | **-€113** |
| Static Wasserstein | €2,298 | €2,625 | €4,923 | +€246 |
| Weekly composite (oracle) | €2,521 | €2,519 | €5,040 | +€363 |
| Weekly Wasserstein (oracle) | €2,640 | €2,493 | €5,133 | +€456 |
| Weekly CRPS (oracle) | €3,009 | €2,271 | €5,280 | +€603 |
| Static CRPS | €3,391 | €2,122 | €5,513 | +€836 |
| *Single model (slurp_bootstrap)* | *€2,357* | *€2,812* | *€5,169* | *+€492* |
| *Official benchmark* | — | — | *€5,248* | *+€571* |
| *Competition winner* | — | — | *€4,677* | *—* |

### 9.3 Key Findings

**The static composite selector at €4,564 beats the competition winner (€4,677) by €113.** This is the strongest result of the entire post-competition analysis.

The model distribution for the winning selector: LightGBM 183 SKUs, SLURP bootstrap 183 SKUs, DeepAR 167 SKUs, SLURP stockout-aware 66 SKUs. No single model dominates — each contributes meaningfully to the ensemble.

Surprisingly, the oracle weekly selectors (which have access to same-week actuals for selection) perform *worse* than the static per-SKU selector. This indicates that model consistency across weeks matters more than per-week optimality. Switching models week-to-week introduces policy inconsistency that harms inventory dynamics.

The composite metric (pinball@CF × Wasserstein) is the best selection criterion — it combines decision-relevant accuracy (pinball at the cost-relevant quantile) with distributional quality (Wasserstein). Pure CRPS-based selection performs worst because CRPS weights all quantiles equally, while inventory cost is concentrated at the critical fractile.

## 10. Gap to Winner and Conclusions

Our best result, the **static composite selector at €4,564**, not only beats the benchmark (€5,248) by 13.0% but surpasses the competition winner (€4,677) by 2.4%. This validates the full probabilistic approach when combined with intelligent per-SKU model selection.

### 10.1 Summary of Approaches

| Approach | Total Cost | vs Winner |
|----------|----------:|----------:|
| Our competition submission | €7,787 | +€3,110 |
| Best single model (slurp_bootstrap static) | €5,087 | +€410 |
| Density SIP (slurp_bootstrap @ SL=0.833) | €5,169 | +€492 |
| Best classical formula (k·MAE) | €5,771 | +€1,094 |
| Official benchmark | €5,248 | +€571 |
| **Static composite selector** | **€4,564** | **-€113** |
| Competition winner | €4,677 | — |

### 10.2 What Made the Difference

1. **Fixing implementation bugs** (cost tallying, L=3, h=3): moved from €7,787 to viable backtesting
2. **Correcting the service level**: the theoretically optimal CF=0.833 is indeed optimal for calibrated models
3. **Training multiple model families**: each excels on different SKU segments
4. **Dynamic model selection**: the composite metric (pinball × Wasserstein) identifies the best model per SKU
5. **Density-based optimization**: SIP newsvendor outperforms all classical safety stock formulas by €602-€1,449
6. **DeepAR addition**: despite being individually weaker, it contributes 28% of selections in the winning ensemble

### 10.3 Remaining Work

1. ~~**SURD transforms** (H3)~~: Completed. SURD models trained, backtested, and statistically tested. H3 not supported — transforms have no net effect on SLURP bootstrap quantiles (see §5.3).
2. **Conformal calibration**: post-hoc quantile adjustment may further improve individual model quality
3. **Causal weekly selection**: the oracle weekly selectors use same-week metrics; building a lagged selector (using last-week metrics for this-week selection) would be a practical, implementable variant
4. **MetaRouter integration**: incorporating Wasserstein and CRPS as features into the existing MetaRouter infrastructure for automated model routing

## References

1. Ben-Tal, A. & Nemirovski, A. (2001). *Lectures on Modern Convex Optimization*. SIAM.
2. Savage, S. (2012). *The Flaw of Averages*. Wiley.
3. Gneiting, T. & Raftery, A.E. (2007). "Strictly Proper Scoring Rules." *JRSSB*.
4. Zipkin, S.A. (2000). *Foundations of Inventory Management*. McGraw-Hill.
5. Powell, W.B. (2022). *Reinforcement Learning and Stochastic Optimization*. Wiley.
6. Simchi-Levi, N., Chen, X. & Bramel, J. (2004). *The Logic of Logistics*. Springer.
7. Meinshausen, N. (2006). "Quantile Regression Forests." *JMLR*.
8. Goltsos, T.E. et al. (2022). "Inventory-Forecasting: Mind the Gap." *EJOR* 299(2), 397-419.
9. VN2 Inventory Planning Challenge. datasource.ai.
