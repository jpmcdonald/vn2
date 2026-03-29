## Backlog / Next-Week Considerations

### 0) Recommended next steps and document testing
- **Recommended next steps (2026-03-12):** [docs/RECOMMENDED_NEXT_STEPS_2026-03-12.md](docs/RECOMMENDED_NEXT_STEPS_2026-03-12.md) — backtest-driven improvement steps, testable hypotheses (e.g. worse financial outcome with better point forecast, cost-based selector overfit, conformal at CF), and links to backlog.
- Document testing: when changing selector logic, eval pipeline, or backtest harness, re-run the selector comparison (`scripts/compare_selector_metrics.py`) and update or regenerate the recommended-next-steps doc if the baseline or priorities change.
- Add to this backlog item as new “golden” backtest scripts or hypothesis tests are added (e.g. holdout selector test, conformal backtest).

### Initiative: EDA expansion, time series regimes, and regime-aware ensemble selection

- **Roadmap:** [docs/EDA_REGIME_ENSEMBLE_SELECTOR_ROADMAP.md](docs/EDA_REGIME_ENSEMBLE_SELECTOR_ROADMAP.md) — expand EDA; derive static and rolling features from tests; classify series into regimes (rules, clusters, or pseudo-labels from historical best model); detect/predict regime changes; benchmark ensemble **selection policies** (baseline fold-cost + CF tie-break vs composite-only vs regime rules vs learned meta-router vs optional bandit) on shared checkpoints with portfolio cost and regime-sliced calibration.
- **Workstreams:** (1) EDA v2 spec and optional new parquets / data contract updates; (2) canonical per-SKU (+ time-varying) feature table and joins from `summary_statistics*`, `stationarity_tests*`, `eda_extensions`; (3) regime catalog and labeling code; (4) change detection / drift features; (5) selector benchmark harness comparing methods without changing champion path until validated.
- **Nomenclature:** disambiguate **SURD-transform** (variance-stabilizing pipeline) vs **SURD-information** (MI/entropy analyses) when specifying regime inputs.
- **Dependencies:** overlaps conceptually with Szabłowski harness stratification, meta-router/cohort tooling, and clean pre-competition retrain; sequence EDA features after data vintage is decided for any retrain wave.
- **Not started.**

### Initiative: Entropy-based regime detection and outcome PMFs (H7–H12)

- **Spec:** [docs/ENTROPY_REGIME_FRAMEWORK_VN2.md](docs/ENTROPY_REGIME_FRAMEWORK_VN2.md) — Shannon / Gaussian-reference entropy on demand and **outcome** (cost) PMFs; Jensen gap trajectories; sensitivity ratio; hypotheses H7–H12; links to scalar-proxy critique (pinball × W1 vs full outcome geometry).
- **Code:** `src/vn2/analyze/entropy_metrics.py`, `src/vn2/analyze/entropy_regime.py`; `scripts/compute_entropy_regime_metrics.py`; `scripts/compute_entropy_regime_labels.py` (EDA join); `scripts/run_entropy_hypotheses_h7_h12.py` → `reports/entropy_hypotheses/`; eval: `model_eval.py --sip --entropy-metrics`.
- **Tasks:** (1) entropy primitives; (2) persist outcome PMFs / scalars; (3) time series per SKU–fold; (4) regime prototype (stable/trending/volatile) + EDA join; (5) backtests H7–H12.
- **Nomenclature:** **Shannon entropy on discrete PMFs** (this initiative) is distinct from **SURD–information** (MI / decomposition) in the EDA roadmap — both may feed regime features; label which signal in any selector experiment.
- **Dependencies:** Align period index with `eval_folds` / sequential backtest; re-run after clean pre-competition retrain if data vintage changes.

### Sprint: Szabłowski comparison & hypothesis testing
- **Full prompt:** [docs/VN2_BACKTESTING_SPRINT_PROMPT.md](docs/VN2_BACKTESTING_SPRINT_PROMPT.md)
- Reproduce Szabłowski (arXiv:2601.18919v1) pipeline (global CatBoost point forecaster + φ-calibrated buffer) for backtesting; keep implementation in a separate area (e.g. `szablowski/`).
- Build a unified comparison backtesting harness: shared simulation engine, rolling-origin backtest (e.g. last 18 weeks, 6 rounds), series-level cost and forecast metrics for both approaches; segment by demand scale, intermittency, CV, stockout frequency, seasonality strength.
- Hypothesis tests (H1–H6): Jensen's Gap by demand regime; annualized scaling vs SURD; time-decayed weighting; his feature engineering in our LightGBM; φ stability vs density quantile; ensemble advantage in tails.
- Feature-engineering revamp and calibration comparison (0.833 quantile coverage, reliability diagrams) as in the prompt.
- **Not started:** planning and implementation deferred until explicitly requested.

### Sprint extension: Neural architecture & policy comparison (post-CatBoost sprint)

**A. Residual structure diagnostic (prerequisite for any integration of new forecasters into SLURP):**
- After reproducing CatBoost pipeline from the main sprint, compute residuals on the validation window.
- Characterize residual ACF, heteroskedasticity (variance vs demand level), and tail behavior.
- Compare to residual properties from existing SLURP, DeepAR, and LightGBM models.
- Purpose: determine whether external bootstrap (SLURP on CatBoost/TCN residuals) or native distributional output is the right integration path.
- This is cheap to run and should gate decisions about items B and C.

**B. TCN distributional forecaster as ensemble member:**
- Adapt the TCN+FiLM+Seasonal Head architecture from the DL/RL deck with a distributional output head (negative binomial parameters, quantile set, or mixture density) replacing the Softplus point-forecast output.
- Train under CRPS or pinball loss instead of Huber loss.
- Evaluate as a new arm in the SIP ensemble alongside existing SLURP, DeepAR, and LightGBM models.
- Hypothesis: the TCN's inductive biases (convolutional temporal extraction, multiplicative FiLM per-series modulation, learned Fourier seasonal head) produce densities that are better calibrated at the critical fractile for specific SKU cohorts — particularly high-volume seasonal series where the learned seasonal head has the most leverage.
- Also evaluate augmentation strategies (time, week, static covariate, input dropout) for applicability to existing neural models (DeepAR).
- Depends on: completed CatBoost sprint, residual diagnostic results from item A.

**C. Three-way policy comparison (extends sprint H1):**
- Run three ordering policies on shared forecasts through the shared simulator:
  - (a) Szabłowski's analytical newsvendor: point forecast + φ√D buffer
  - (b) Szabłowski's RL agent: PPO-learned demand multiplier + safety buffer (contingent on code availability)
  - (c) Our SIP density-based newsvendor: full density propagated through cost function
- Test whether density propagation captures value that neither the analytical proxy nor the RL agent recovers.
- If RL ≈ SIP but analytical diverges → distributional information matters but can be approximated empirically with enough rollouts.
- If SIP > both → explicit density propagation captures something neither heuristic recovers (Jensen's Gap).
- If RL > SIP → multi-period sequential effects matter and single-period newsvendor is insufficient.
- Depends on: completed CatBoost sprint, shared simulator, access to Szabłowski's DL/RL code (flag as conditional).

**Sequencing note:** Do not start this sprint extension until the main CatBoost sprint is complete. Results from H1 (Jensen's Gap by regime) and H5 (φ stability) should inform priority. If H1 shows Jensen's Gap is small across most regimes, item B becomes lower priority. If H5 shows φ is stable across windows, the distributional head becomes less urgent.

- **Not started:** blocked until CatBoost reproduction + harness land; see `docs/VN2_BACKTESTING_SPRINT_PROMPT.md` sprint extension section.

### Retrain all models on clean pre-competition data and repeat full analysis

- **Context:** The current `demand_long.parquet` includes competition-period actuals (Weeks 1-8, 2024-04-15 through 2024-06-03) appended via `scripts/build_demand_long.py`. The original SLURP/LightGBM/DeepAR models were trained before this data was added, so their training was clean. However, any future retraining on the current `demand_long.parquet` would leak competition data into those models. CatBoost (retrained v2) explicitly excludes competition weeks via `competition_split()`.
- **Task:** Rebuild `demand_long.parquet` in two variants: (a) `demand_long_precomp.parquet` containing only pre-competition data (up to 2024-04-08), and (b) the full version for post-hoc analysis. Retrain all models (SLURP bootstrap, SLURP stockout-aware, LightGBM quantile, DeepAR, CatBoost) on the pre-competition-only dataset with consistent `holdout_weeks` ensuring no fold can see competition-period demand. Re-run the full evaluation pipeline, SIP/analytical comparison, and hypothesis tests on the retrained models to produce a true apples-to-apples comparison.
- **Why it matters:** Without this, the policy comparison between SIP (using models trained on one data vintage) and analytical (using CatBoost retrained on another) is not on equal footing. A clean retrain ensures the cost comparison reflects model/policy differences, not data-availability differences.
- **Estimated effort:** 4-8 hours (mostly wall-clock for retraining all models across 599 SKUs x 12 folds).
- **Not started:** deferred to next weekend.

### 1) Online improvement loop (bandits + calibration)
- Contextual bandit for per-SKU model choice (arms: ZINB, QRF; optional SLURP/ETS)
  - Contexts: rate_bin, zero_bin, cv_bin, stockout_bin, recent CF calibration
  - Reward: − realized SIP cost at h2 per fold
  - Policy: Thompson or ε-greedy with small exploration; weekly updates
- Bandit for CF (service-level) delta around 0.833
  - Actions: {−0.03, −0.02, 0, +0.02, +0.03}; reward: − realized cost
  - SIP translates CF→Q; small caps; logging + safe fallback
- Decision-level pooling (online weights)
  - Maintain weights over ZINB/QRF expected cost curves; update via exponentiated gradient on regret
- Online conformal calibration near CF
  - Weekly recalibration targeting coverage/hit at critical fractile

### 2) Strict 8-of-8 selector and portfolio totals
- Re-run SIP evaluation to guarantee full 12-fold coverage for selected models (ZINB, QRF, Seasonal Naive)
- Compute strict 8-of-8 per-SKU selector on identical SKUs/folds
- Compare selector total vs single-model totals on same universe

### 3) SIP evaluation enhancements
- Emit per-SKU per-week realized costs (weeks 2–9) to allow true multi-week totals
- Add option to aggregate over last-N decision-affected weeks directly in pipeline
- Persist per-SKU coverage diagnostics to simplify strict selection

### 4) SLURP follow-ups
- Revisit stockout-aware neighbor selection and censoring feature design
- Increase PMF grain and ensure quantile-to-PMF fidelity near CF
- Evaluate SURD impact specifically on CF-local shape (pinball_cf, hit_cf, local_width)

### 5) Paper additions
- Add ensemble section (selector, cohort, decision-level); explain ZINB vs QRF dynamics
- Add online learning plan (bandits + calibration) and expected value proposition
- Clarify nomenclature: h1/h2 vs calendar weeks; week-2 = h2 each fold

### 6) Operational
- CLI commands for bandit state save/load; checkpoint JSON schema
- Shadow-mode logging for 1–2 folds before activation
- Safety rails: max CF delta, exploration cap, fallback to ZINB on uncertainty

### 7) Nice-to-haves
- Cohort visualizations for model advantage and Jensen deltas
- PID-weighted feature scaling checkpoints in SLURP
- MPC-like 3–4 step SIP lookahead for high-impact SKUs


- [ ] EDA expansion, regimes, regime-aware ensemble selector benchmark — [docs/EDA_REGIME_ENSEMBLE_SELECTOR_ROADMAP.md](docs/EDA_REGIME_ENSEMBLE_SELECTOR_ROADMAP.md)
- [ ] Entropy / outcome-PMF regime framework (H7–H12) — [docs/ENTROPY_REGIME_FRAMEWORK_VN2.md](docs/ENTROPY_REGIME_FRAMEWORK_VN2.md)
- [ ] Deep analysis: risk/payoff around critical fractile (p*=cu/(cu+co)), curvature near CF, breakpoint between EV vs CF-driven ordering; derive diagnostics and plots; per-SKU and portfolio views.
