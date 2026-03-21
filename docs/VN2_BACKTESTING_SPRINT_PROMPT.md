# VN2 Backtesting Sprint: Szabłowski Comparison & Hypothesis Testing

## Context

This repo contains my VN2 Inventory Planning Competition work. The competition required weekly ordering decisions for 599 store-product pairs under a two-week lead time, minimizing total cost = shortage_cost (cs=1.0) * lost_sales + holding_cost (ch=0.2) * end_of_week_inventory.

My approach uses density forecasting through an ensemble of models (SLURP Bootstrap, SLURP Stockout Aware, SLURP SURD Transformed, DeepAR, LightGBM Quantile, LightGBM SURD Transformed) feeding a stochastic optimizer that evaluates full probability distributions through the cost function. The critical fractile q* = cs/(cs+ch) ≈ 0.833 drives inventory positioning.

The winning solution (Szabłowski, arXiv:2601.18919v1) used a fundamentally different architecture: a single global CatBoost model producing point forecasts for h=1,2,3, combined with a lightweight newsvendor-derived ordering policy. His uncertainty proxy is σ = φ√(D̂), where φ is a single scalar calibrated on the validation cost function. He beat the VN2 benchmark by 13.2% (3.763€ vs 4.334€).

Szabłowski also published a post-competition extension replacing CatBoost with a custom TCN+FiLM+Seasonal Head deep learning forecaster and the analytical ordering policy with a PPO reinforcement learning agent. This achieved 17.3% cost reduction vs benchmark (3,582€) compared to the CatBoost solution's 13.2%. Summary at `docs/szablowski_dl_rl_deck_summary.md`. The neural architecture and RL policy comparison are scoped as a separate sprint extension in `backlog.md`, to be started after the CatBoost sprint completes.

I need to backtest his approach against mine and test specific hypotheses about where and why the approaches diverge in performance. Reference paper is at `/mnt/user-data/uploads/2601_18919v1.pdf` if you need to consult it.

---

## Sprint Objectives

### 1. Reproduce Szabłowski's Pipeline for Backtesting

Build a clean implementation of his two-stage pipeline so we can run it against the same backtest windows as our existing models. Key components:

**Stage 1 — Global CatBoost Forecaster:**
- Direct multi-horizon strategy: three independent CatBoost regressors for h=1, h=2, h=3
- Stockout-aware effective sales series: mask weeks where in_stock=False to NaN before computing any features
- Feature groups (replicate his feature set as described in Section 6.2):
  - Demand level: short-term lags (t, t-1, t-2, t-3), seasonal lags (t-51, t-52, t-53), rolling means/medians (windows 3, 5, 13), EWM (spans 5, 10), rolling std, rolling IQR
  - Trend/momentum: Δ1 and Δ5 differences, 4-week rolling slope proxy
  - Seasonality: week_of_year as categorical, Fourier sin/cos for harmonics k=1,2,3, last-year-window average, seasonality-strength proxy (correlation between current sales and seasonal lags)
  - Intermittency/spikes: rolling median + MAD → robust z-score, time_since_spike, rolling non-zero rate over 12 weeks
- Dynamic per-series scaling:
  ```
  scale_factor(i,t) = max(53 * mean_nonnmissing(y_eff[t-52:t]), 1)
  ```
  With warm-start proxy (expanding mean, annualized) when fewer than 45 in-stock observations available
- All target-based features computed on scaled y_eff
- Two-level median imputation: per-series median of scaled values first, then global scaled median for remaining NaNs
- Time-decayed observation weights: 1.0 for most recent ~53-week block, 0.5 for preceding block, 0.25 for the block before that
- Training loss: RMSE on scaled target
- Hyperparameter selection: Optuna (100 trials, TPE sampler) minimizing MAE in original units on chronological validation split (most recent 10% of training weeks)
- Early stopping: 500 rounds
- Local test holdout: last 18 weeks

**Stage 2 — Ordering Policy:**
- Post-process forecasts: round to nearest integer, clip to ≥ 0
- Project inventory forward through t+1, t+2 using point forecasts and lost-sales assumption (inventory can't go negative)
- Cost-aware target stock:
  ```
  q* = cs / (cs + ch) = 0.8333
  zq = Φ⁻¹(q*) ≈ 0.9674
  σ(i,t+3) = φ * sqrt(D̂(i,t+3))
  B(i,t+3) = D̂(i,t+3) + zq * σ(i,t+3)
  Q(i,t) = max(B(i,t+3) - Ĩ(i,t+3), 0)
  ```
- Calibrate φ on validation window by minimizing total cost (shortage + holding)

### 2. Build Comparison Backtesting Harness

Create a unified backtesting framework that runs both pipelines through identical evaluation:

- **Shared simulation engine**: Replicate the VN2 inventory dynamics exactly (equations 1-5 from the paper). Both approaches must feed into the same simulator so costs are comparable.
- **Rolling-origin backtest**: Use the last 18 weeks of history as test, with 6-week rolling windows matching competition rounds (6 rounds of ordering + 2 weeks of pipeline drain). If data supports it, run multiple origins.
- **Output for each approach per backtest window**:
  - Total cost (shortage + holding, broken out)
  - Cost by series (identify which store-product pairs drive differences)
  - Forecast accuracy metrics at each horizon: MAE, RMSE, MAPE (for non-zero series), bias
  - For my density models: CRPS, pinball loss at the 0.833 quantile, and reliability diagrams
  - For his point + buffer approach: effective quantile coverage at the 0.833 level
- **Head-to-head series-level comparison**: For each series, compute cost delta (his cost minus my cost). Segment results by series characteristics:
  - Demand scale (low/medium/high average weekly sales)
  - Intermittency (% zero weeks)
  - Coefficient of variation
  - Stockout frequency
  - Seasonality strength

### 3. Hypothesis Tests

These are the specific hypotheses I want to test. Structure the code so each produces clear quantitative evidence.

**H1: Jensen's Gap magnitude varies by demand regime.**
- For each series, compute the Jensen's Gap: the difference between (a) the cost from evaluating the full density through the cost function and (b) the cost from applying the cost function to the point forecast with a normal-approximation buffer.
- Segment by demand characteristics. The hypothesis is that Jensen's Gap is small for well-behaved, moderate-volume series but large for intermittent, heavy-tailed, or highly variable series.
- Quantify: what fraction of total cost difference between approaches is attributable to series where Jensen's Gap exceeds some threshold?

**H2: His annualized scaling ≈ my SURD transforms for global model performance.**
- Take my LightGBM models and swap SURD transforms for his simpler annualized scaling approach (and vice versa if tractable).
- Compare forecast accuracy and downstream cost. The hypothesis is that SURD provides marginal improvement for well-behaved series but meaningful improvement for series with strong heteroscedasticity.

**H3: Time-decayed weighting improves my models.**
- Apply his stepwise yearly decay (1.0 → 0.5 → 0.25) to my model training.
- Also test continuous exponential decay as an alternative.
- Compare both against my current uniform weighting. Evaluate on forecast accuracy and cost.

**H4: His feature engineering additions improve my density forecasts.**
- Specifically test adding to my LightGBM Quantile model:
  - Seasonality-strength proxy
  - Robust z-score spike detection + time_since_spike
  - Rolling non-zero rate
  - 4-week rolling slope proxy
- Evaluate whether these features improve the tails of my predictive distributions (check pinball loss at 0.833 and 0.95 quantiles).

**H5: The φ calibration absorbs distributional information that a proper density forecast provides explicitly.**
- Vary φ across a grid and plot total cost as a function of φ.
- Compare the cost-optimal φ to the implied buffer from my density forecasts at the critical fractile.
- The hypothesis is that the optimal φ is fragile (sensitive to the backtest window) while the density-based quantile is more stable across windows.

**H6: My ensemble's advantage (if any) concentrates in the tails.**
- Compare my approach vs his specifically on series where realized demand fell in the top or bottom 10% of the historical distribution during the test period.
- The hypothesis is that my density approach performs comparably on "normal" periods but significantly better during tail events.

### 4. Feature Engineering Revamp

Based on what we learn, integrate the best of his feature engineering into my pipeline:

- **Stockout-aware masking**: Audit my current OOS handling. My SLURP Stockout Aware model addresses this at the distributional level, but my other models may benefit from his feature-level masking. Implement y_eff construction across all models and compare.
- **Dynamic scaling as preprocessing**: Test his annualized scaling as a universal preprocessing step before my models, independent of SURD. They may be complementary — his scaling for level normalization, SURD for variance stabilization.
- **Spike/intermittency features**: His time_since_spike and rolling non-zero rate are compact representations of intermittency state. Add these to my feature sets.
- **Seasonality-strength proxy**: This lets the model learn to weight seasonal features differently for series with weak vs strong seasonality. Implement and test.

### 5. Calibration Comparison

This is critical. I want to understand whether his forecasts were better *calibrated* for the specific purpose of the cost function, even if they contain less distributional information.

- For my density forecasts: compute the 0.833 quantile from each model and from the ensemble. Check calibration — what fraction of realized demands actually fell below this quantile? Break out by series.
- For his approach: the effective quantile is D̂ + zq * φ * √D̂. Check what fraction of realized demands fall below this level. Break out by series.
- If his effective coverage is closer to 83.3% than mine, that indicates a calibration problem in my density models that feature engineering or post-processing could fix.
- Produce reliability diagrams for both approaches across the full quantile range (not just at 0.833).

---

## Implementation Notes

- Start by examining the existing repo structure and understanding my current pipeline before building anything new.
- Keep Szabłowski's implementation cleanly separated (e.g., `szablowski/` directory) so it's easy to compare and won't contaminate my existing code.
- The simulation engine is the single source of truth for cost comparison. Both pipelines must produce order quantities that feed into the same simulator.
- Use the competition's exact cost parameters: cs=1.0, ch=0.2, lead time=2 weeks, 599 series.
- All hypothesis tests should produce both summary statistics and series-level detail. I want to see distributions, not just averages.
- Save all backtest results to structured output (parquet or similar) for downstream analysis.
- Prioritize getting H1 (Jensen's Gap by regime) and H5 (φ stability) running first — these are the most theoretically important.

---

## Sprint extension: Neural architecture & policy comparison (post-CatBoost)

**Depends on:** Completed CatBoost reproduction (Stage 1–2 above), unified comparison harness and shared simulator, and availability of published code for the DL/RL pipeline where referenced.

### Residual diagnostic

After reproducing the CatBoost pipeline, characterize residual **ACF**, **heteroskedasticity**, and **tails**; compare to **SLURP** residuals; determine whether **external bootstrap** or **native distributional output** is the right integration path for combining point-global forecasts with uncertainty.

### TCN distributional forecaster

Adapt Szabłowski's **TCN + FiLM + Seasonal Head** (from DL/RL deck) with a **distributional output head** (NegBin or quantile); train under **CRPS**; evaluate as a new **SIP ensemble** arm.

**Hypothesis:** Learned per-series seasonal modulation improves **density calibration at the critical fractile** for seasonal / high-volume SKU cohorts.

### Three-way policy comparison (extends H1)

On **shared forecasts** through the **shared simulator**, run:

1. **Analytical** policy (φ√D̂ buffer as in Stage 2),
2. **RL** policy (if code is available),
3. **SIP density** policy (existing pipeline).

Test whether **density propagation** captures value that **neither** the heuristic φ calibration nor RL recovers.

### Implementation ordering

1. Finish CatBoost sprint + harness + H1/H5 baseline.  
2. Residual diagnostic → informs bootstrap vs native density path.  
3. TCN distributional model → CRPS training → eval as ensemble member.  
4. Three-way policy comparison once RL artifact (if any) is wired to the same simulator inputs.
