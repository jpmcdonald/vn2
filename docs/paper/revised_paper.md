# Why Forecast Precisely Right, Only to Optimize Explicitly Wrong: Empirical Evidence from an 8-Week Inventory Planning Competition

## Abstract

We present empirical results from a post-competition analysis of the VN2 Inventory Planning Challenge, a real-time 8-week inventory optimization problem with 599 store-item pairs, asymmetric costs (shortage cost cu=1.0, holding cost co=0.2), and L=3 lead time. We test four hypotheses about density-aware decision-making versus point+service-level policies using a corrected backtesting framework that separates ordering costs from evaluation costs. Our SLURP bootstrap model at the newsvendor-optimal service level (0.833) achieves €5,087 total cost, beating the official benchmark (€5,248) by 3.1%. We find that the Jensen gap is model-dependent: well-calibrated density forecasters benefit from full-distribution optimization (+€815), while miscalibrated models are harmed by it (-€827). Stockout awareness improves cost at conservative service levels but slightly hurts at the optimal point. Train-once strategies outperform sequential refitting, and single-period metrics do not perfectly predict multi-period simulation rankings. These findings demonstrate that forecast calibration is the binding constraint on density-aware inventory optimization.

## 1. Introduction

The inventory planning literature prescribes density-based decision-making for settings with asymmetric costs: the newsvendor critical fractile CF = cu/(cu+co) identifies the optimal quantile of the demand distribution at which to set order quantities. In practice, most organizations chain a point forecast to a deterministic policy (order-up-to, safety stock formulas), implicitly assuming the cost function is linear in forecast error. Jensen's inequality guarantees this approach is suboptimal whenever the cost function is convex and the forecast distribution is non-degenerate.

We built an end-to-end probabilistic inventory optimization pipeline for the VN2 Inventory Planning Challenge: quantile forecasts from multiple model families, converted to probability mass functions (PMFs) via SIP (Stochastic Information Packets), optimized through a sequential newsvendor solver with L=3 lead time, integer order quantities, and realistic in-transit inventory tracking. During the competition, our system placed 110th of ~150 teams (€7,787 vs winner €4,677). Post-competition analysis revealed a critical implementation bug and produced substantially improved results that inform our four hypotheses.

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

All models produce 13 quantile forecasts at levels [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99] for horizons h=1, h=2, h=3.

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

### 4.5 Training Strategy: Static vs Sequential

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

### 5.3 H3: SURD Effect — Deferred

The SURD-enabled models were not trained in this round. Indirect evidence from computed SURD transforms shows meaningful variance reduction for ~40% of series. The 2×2 ablation (SURD on/off × Stockout on/off) is planned for the next training round.

### 5.4 H4: Sequential Consistency — Supported

| Metric | Rank 1 | Rank 2 |
|--------|--------|--------|
| 8-week cost (SL=0.833) | slurp_bootstrap | slurp_stockout_aware |
| Pinball @ q=0.833 | slurp_stockout_aware | slurp_bootstrap |
| Cost-weighted pinball | slurp_stockout_aware | slurp_bootstrap |

The top two models swap ranks between single-period metrics and multi-period simulation cost. slurp_stockout_aware has marginally better pinball loss at the critical fractile (0.688 vs 0.689) but worse 8-week simulation cost (€5,264 vs €5,169). This rank reversal demonstrates that inventory dynamics — carry-forward of excess inventory, lead-time interactions, and path-dependent compounding — create effects that single-period metrics cannot capture. H4 is supported.

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

## 7. Gap to Winner and Future Work

Our best result (€5,087) beats the benchmark (€5,248) by 3.1% but remains €410 above the winner (€4,677). The remaining gap is likely attributable to:

1. **Forecast quality at the critical fractile.** Our pinball loss at q=0.833 is 0.69; reducing this by ~15% through better calibration or model ensembling could close the gap.
2. **SKU-level policy adaptation.** Using a single service level for all 599 SKUs ignores heterogeneity in demand patterns. A per-SKU or per-segment selector could route low-demand SKUs to order-0 policies and high-demand SKUs to aggressive ordering.
3. **SURD transforms.** Enabling variance-stabilizing transforms may improve calibration at the critical fractile for heteroskedastic series.
4. **Conformal calibration.** Post-hoc calibration of quantile forecasts using historical actuals could correct the systematic biases observed in all models.

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
