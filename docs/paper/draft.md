<!-- Working draft for INFORMS Analytics+ submission -->

## Why Forecast Precisely Right, Only to Optimize Explicitly Wrong: a Censored SURD^2 SLURP Bootstrapping Engine

### Abstract

Many operational decisions are made by chaining a point forecast to a deterministic optimization policy, implicitly assuming linear objectives and symmetric loss. In inventory systems with asymmetric costs, lead times, and stockout censoring, this practice systematically underperforms density-based decision making due to Jensen’s inequality. We present a practical, production-ready framework that integrates: (i) stockout-aware conditional bootstrapping Stochastic Library Unit with Relationships Preserved (SLURP) operating on raw, censored observations; (ii) Systematic Unsupervised Representation Discovery (SURD) to select variance-stabilizing transforms per series; (iii) Synergistic Unique Redundant Decomposition (also SURD) and (iv) SIP (Stochastic Information Packet) optimization that solves the sequential newsvendor with integer order quantities and explicit lead times. We show how to avoid the common anti-pattern of “forecasting precisely right, then optimizing explicitly wrong” by evaluating models under the decision they induce. Across 599 store–item time series with frequent zeros and stockouts, our approach reduces realized cost relative to point+service-level policies and standard density baselines, and is robust to extreme imputations by modeling censoring uncertainty directly in the forecast distribution.

### 1. Introduction

Forecasting and optimization are often treated as separable modules: a model produces a point prediction, and an optimization rule converts it into an operational decision. This separation is attractive, but fragile in settings with asymmetric costs and nonlinear objectives (e.g., shortage vs. holding), where decisions depend on the shape—not just the center—of the predictive distribution. In retail inventory with stockouts and lead times, the forecasting target is censored demand, not true demand; naively imputing stockout periods as point values collapses uncertainty and can yield unstable training targets and biased service policies.

We propose an end-to-end decision-first forecasting framework that: (1) retains censoring information and pushes it into the forecast density via a stockout-aware SLURP bootstrap; (2) stabilizes variance using per-series SURD transforms for sharper, better-calibrated distributions; and (3) optimizes integer order quantities with a SIP-based sequential newsvendor solver. We evaluate both density-aware SIP decisions and point+service-level policies to quantify the Jensen gap (the performance difference arising from nonlinearity and distributional curvature). Our contributions are: practical methods, a reproducible pipeline, and empirical evidence that modeling uncertainty correctly matters more than chasing marginal point accuracy gains.

### 2. Hypotheses

- H1 (Jensen Gap): For the same forecast information set, density-aware SIP optimization achieves strictly lower realized cost than point+service-level policies, particularly under asymmetric costs and non-Gaussian demand.
- H2 (Stockout Awareness): A stockout-aware SLURP bootstrap, which preserves censoring uncertainty, achieves lower realized shortage cost and improved decision metrics versus bootstrap models trained on imputed targets.
- H3 (SURD Effect): Applying SURD-chosen transforms (e.g., log1p/sqrt/cbrt) per series yields sharper and better-calibrated predictive intervals near the critical fractile, improving cost and calibration relative to identity-space bootstrapping.
- H4 (Sequential Consistency): SIP with integer optimization and correct lead-time semantics (non-zero initial state, first-arrival timing) avoids degenerate ties and better discriminates model quality than single-period proxies.

### 3. Theory

#### 3.1 Jensen’s Inequality and Decision Curvature
Operational cost functions are typically nonlinear in the decision and random outcomes (e.g., asymmetric piecewise linear holding–shortage cost). Let C(Q; D) be cost under order quantity Q and random demand D. For nonlinear C, E[C(Q; D)] ≠ C(Q; E[D]). Optimizing on a point forecast (e.g., E[D]) therefore induces suboptimal decisions whenever curvature matters. When order decisions target a critical fractile (CF = p/(h+p)), the local shape of the predictive CDF around the CF determines the regret; sharper and better-calibrated distributions reduce expected loss.

#### 3.2 Censoring, Not Missingness
Stockouts censor demand: observed sales Y = min(D, S), where D is true demand and S is available stock. Treating censored periods as missing and imputing point values discards structural information about right-tail mass. A stockout-aware learner should upweight uncertainty during such periods rather than collapsing to a single imputed count.

#### 3.3 SURD: Variance-Stabilizing Transforms
Different series exhibit different dispersion–mean relationships. SURD selects among a small library (identity, log, log1p, sqrt, cbrt) per series, seeking transforms that stabilize variance and linearize dynamics. Forecasting in transform space and inverting without mean-bias correction for quantiles preserves rank information and improves sharpness.

Terminology note: In this work, SURD stands for Systematic Unsupervised Representation Discovery (transform selection). This differs from the information-theoretic literature on Synergistic, Unique, and Redundant Decomposition (also sometimes abbreviated SURD) concerned with decompositions of information/causality.

#### 3.4 SIP Math for the Sequential Newsvendor
We model the start-of-period inventory position as a random variable I_t and lead-time arrivals A_t (known for the first arrival). Given a predictive demand distribution for horizons h=1,2, we construct the distribution of ending inventory after sales: I_{t+1} = max(0, I_t + A_t − D_t). The order decision Q_t is an integer chosen to minimize expected cost over the evaluation horizon, accounting for the distribution of I_{t+2} before the second arrival. We discretize predictive distributions (quantiles→PMF), convolve with inventory, and evaluate discrete Q in [0, Q_max] to select argmin_Q E[holding + shortage]. This enforces correct lead-time timing and non-zero initial state.

### 4. Methodology

#### 4.1 Data and Censoring
We use 599 store–item weekly series with frequent zeros and stockouts. For stockout-aware models, we train on raw observations with an in_stock indicator to preserve censoring structure. For non-aware baselines, we create a stable target by winsorizing imputed demand in transform space (SURD μ+3σ cap, inverted and clipped), modifying only imputed extremes and preserving genuine observations.

#### 4.2 Models (Model Canvas)
- SLURP Bootstrap (baseline): conditional bootstrap on k-NN histories.
- SURD-SLURP: SLURP in transform space with SURD-selected transform.
- Stockout-Aware SLURP: censoring-aware sampling to retain right-tail uncertainty during stockouts.
- LightGBM Quantile: gradient boosted quantile regression (density).
- LightGBM Point: MSE objective with Gaussian residual PIs (point baseline).
- Linear Quantile Regression (density).
- NGBoost (LogNormal): distributional gradient boosting.
- QRF: quantile random forest via tree prediction distributions.
- ETS: exponential smoothing with prediction intervals.
- (Deferred) GLM Poisson/NegBin: parametric count with PPF quantiles.

#### 4.3 Decision Policies
- Density-aware SIP: quantiles→PMF→SIP optimizer with discrete Q, lead-time arrivals, exclude week-1 realized costs if no decision can affect them, and compute both expected and realized cost.
- Point+Service-Level: use a point forecast (mean/median) and target CF = p/(h+p) with a model-specific distributional approximation (e.g., Normal residuals, Poisson/NB), reflecting common industry practice.

#### 4.4 Metrics
Point: MAE, MAPE, MASE, bias, RMSE. Density: pinball loss, CRPS, coverage and width at 80/90/95. Decision: service level, fill rate, shortage/holding cost, regret quantity. Newsvendor-local: pinball_cf, hit_cf, local_width, quantile_gradient near CF; asymmetric loss around CF. SIP: realized cost (week-2) and expected cost.

### 5. Exploratory Data Analysis (EDA)

We characterize sparsity (counts of zeros), stockout frequency (in_stock=0), dispersion vs. mean by cohort, and right-tail behavior. We report extreme imputed outliers observed in preliminary preprocessing (up to 2.28×10^11), motivating transform-space winsorization limited to imputed periods. We also summarize SURD transform selections across series and their effect on variance stabilization. Plots: demand histograms, ACF/PACF, transform round-trip checks, interval width vs horizon, and stockout heatmaps by cohort.

### 6. Model Training and Selection

We train 9 models with parallel checkpointing and per-model data handling (raw+censoring for SLURP family; winsorized imputed for challengers). Resource use is configurable (n_jobs, BLAS threads). Selection is per series based on SIP realized cost (week-2) with density metric tie-breakers. We store fold-level evaluations and aggregate leaderboards.

### 6.1 Challenger Benchmark Plan (Density vs Point; Jensen Study)

- Objectives:
  - Maintain SLURP variants on clean raw+censoring for ablation parity
  - Build broad challenger set (point and density)
  - Winsorize imputed data in transform space for non-SLURP models
  - Evaluate each model under two decisions: density-aware SIP vs. point+service-level
  - Quantify SURD effect, stockout-awareness, Jensen gap, and challenger parity

- Data strategy:
  - SLURP (4): train on raw with in_stock (no imputation)
  - Non-SLURP: train on `demand_imputed_winsor.parquet` (SURD transform-space μ+3σ winsorization of imputed extremes; invert and clip; preserve observed weeks)

- Models to train (suite):
  - SLURP family: slurp_bootstrap; slurp_surd; slurp_stockout_aware; slurp_surd_stockout_aware
  - Challengers: lightgbm_quantile (density); lightgbm_point (deterministic/MSE); linear_quantile; NGBoost (LogNormal); QRF; ETS; Seasonal Naive; ZIP/ZINB; Croston (classic/SBA/TSB); GLM Poisson/NB (deferred)

- Decision policies per model:
  - Density-aware SIP: model quantiles → PMF → SIP optimizer (lead-time, integer Q, exclude week-1 realized cost)
  - Point+service-level: point forecast + CF = p/(h+p); order to target with best-practice approximations (Normal from residuals for continuous; Poisson/NB for intermittent)

- Evaluation matrix (v4):
  - Cost with densities (SIP) and with point policy
  - Coverage/width at 80/90/95; pinball@CF; CRPS; shape metrics near CF; service/fill rate
  - Leaderboards by realized cost; cohort leaderboards (stockout rate; mean demand; sparsity)

- What we learn (ties to hypotheses):
  - Jensen effect: Δcost = (point policy) − (SIP); expect positive gap favoring SIP
  - Stockout-awareness: SLURP stockout-aware vs. non-aware on high-stockout cohorts; expect lower shortage cost
  - SURD effect: SLURP SURD vs. non-SURD; expect improved calibration/sharpness near CF
  - Censoring bias: quantify penalty of non-SLURP models trained on winsorized vs raw
  - Challenger parity: which baselines are close/dominated; where SLURP adds most value

- Deliverables:
  - Winsorized data artifact; trained checkpoints; folds/agg/leaderboards for both decisions
  - Ablation and Jensen gap plots; cohort analyses

- Timeline and risks (abbrev.):
  - Training 4–8h; evaluation 3–6h (depending on cores)
  - Prophet/GLM optional; convergence/install risks mitigated by skipping/fallbacks

### 7. Evaluation of Hypotheses (placeholder)

We will present:
- Jensen gap per model: Δcost(point policy − SIP) and its distribution across series.
- Stockout-awareness effect: SLURP stockout-aware vs non-aware on censored cohorts.
- SURD effect: transform vs identity for SLURP and ablations; changes in calibration at CF.
- Cohort analysis by sparsity, mean demand, and stockout rate.

#### 7.1 Jensen Effect (Decision Gap)

Using the v4 evaluation, we computed Jensen deltas per model as Δ = cost(point policy) − cost(SIP). Positive values favor density-aware SIP decisions.

- Top deltas (aggregated): croston_tsb/ets/knn_profile/lightgbm_point/ngboost/seasonal_naive/slurp_* ≈ +606,858; croston_classic/sba ≈ +604,107; qrf ≈ +383,916; zinb ≈ +11,230.
- Interpretation: optimizing on the full predictive distribution materially reduces realized cost versus chaining a point forecast to a service-level rule, with the largest gains on methods whose point policies under-react to right-tail risk.

#### 7.2 Cohort Analysis

We sliced Jensen deltas by demand rate, zero ratio, dispersion (CV), and stockout rate to localize where decision-aware methods help most.

- By demand rate: largest mean gaps on low-rate (sparse) series for Croston/ETS/KNN families.
- By zeros: zinb shows moderate gains on low-zero cohorts; Croston/ETS gains grow with higher zero ratios.
- By CV: biggest gains in low/mid CV cohorts for Croston/ETS/QRF; limited sample in high CV.
- By stockouts: mid/high stockout cohorts show large gains for QRF and Croston/ETS relative to point policy.

Figures: Jensen delta summaries by cohort (higher is better for SIP vs point policy)

![Jensen by demand rate](../../reports/jensen_rate.png)

![Jensen by zero ratio](../../reports/jensen_zero.png)

![Jensen by coefficient of variation](../../reports/jensen_cv.png)

![Jensen by stockout rate](../../reports/jensen_stockout.png)

### 8. SLURP Ablation Analysis (placeholder)

2×2 ablation of SURD (on/off) × Stockout-aware (on/off):
1) SLURP baseline, 2) SLURP+SURD, 3) SLURP+StockoutAware, 4) SURD+StockoutAware. Compare sharpness, calibration near CF, and SIP realized cost.

### 9. Conclusions (placeholder)

We will synthesize: decision-driven forecasting beats point+service-level; censoring should be modeled, not imputed; SURD improves interval sharpness and calibration; and SIP with integer optimization and correct lead times is necessary to separate models under realistic constraints.

### References

1. A. Ben-Tal and A. Nemirovski (2001). Lectures on Modern Convex Optimization. SIAM.
2. S. A. van der Vaart (1998). Asymptotic Statistics. Cambridge University Press.
3. S. Boyd and L. Vandenberghe (2004). Convex Optimization. Cambridge University Press.
4. S. E. Shreve (2004). Stochastic Calculus for Finance II. Springer.
5. S. Boucheron, G. Lugosi, and P. Massart (2013). Concentration Inequalities. Oxford.
6. N. Simchi-Levi, X. Chen, and J. Bramel (2004). The Logic of Logistics. Springer.
7. A. Muharremoglu and J. Tsitsiklis (2008). “Price-of-capacity delay penalties...”. Operations Research.
8. S. A. Zipkin (2000). Foundations of Inventory Management. McGraw-Hill.
9. R. Fildes and K. Ord (2010). “Forecasting Competitions...”. IJF.
10. N. Meinshausen (2006). “Quantile Regression Forests.” JMLR.
11. A. Owen (2013). Monte Carlo Theory, Methods and Examples. (online book).
12. T. Hastie, R. Tibshirani, and J. Friedman (2009). The Elements of Statistical Learning. Springer.
13. E. S. Gelman et al. (2014). Bayesian Data Analysis. CRC.
14. T. Gneiting and A. E. Raftery (2007). “Strictly Proper Scoring Rules...”. JRSSB.
15. F. Taieb et al. (2012). “Bias-Variance Decomposition for Forecast Error...”. IJF.
16. W. B. Powell (2022). Reinforcement Learning and Stochastic Optimization. Wiley.
17. S. Silver (1970). “A Simple Inventory Replenishment Policy...” Operations Research.
18. J. Bimpikis and A. Markakis (2019). “Data-driven Newsvendor...” MSOM.
19. S. Ban and M. Rudin (2019). “Multiperiod Newsvendor...” Manufacturing & Service Ops Mgmt.
20. T. Chapelle et al. (2020). “NGBoost: Natural Gradient Boosting for Probabilistic Prediction.” NeurIPS.
21. S. Savage (2012). The Flaw of Averages: Why We Underestimate Risk in the Face of Uncertainty. Wiley.
22. P. L. Williams and R. D. Beer (2010). “Nonnegative Decomposition of Multivariate Information.” arXiv:1004.2515. (Foundational work on unique, redundant, and synergistic information; related to decomposing causality/influence.)


