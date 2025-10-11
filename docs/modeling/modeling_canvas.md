## Modeling Canvas (Density Forecasting)

Purpose: Brainstorm and scope candidate probabilistic forecasters, features, segmentation, metrics, compute expectations, risks, and down‑selection criteria before implementation.

### 1) Problem framing

- Objective: SKU‑level weekly density forecasts for order optimization and simulation
- Data: `data/processed/demand_imputed.parquet` (imputed stockouts), metadata (ABC/XYZ, hierarchy)
- Horizon: per config; near‑term weekly decisions (lead/review weeks) + rolling weekly updates
- Segmentation: ABC/XYZ for scale/volatility; optional seasonal clusters

### 2) Candidate model families (brainstorm)

Classical/time‑series
- Seasonal Naive + bootstrapped residuals (PI)
- ETS (Error‑Trend‑Seasonal) with state‑space PI
- STL + ARIMA/AR with residual bootstrap
- TBATS/Prophet with PI (multiple seasonalities)

Quantile/Distributional ML
- Linear Quantile Regression (global, with interactions)
- Gradient Boosting Quantile (LightGBM/XGBoost; per‑quantile or pinball‑averaged)
- Distributional Regression (NGBoost, CatBoost Gaussian/Poisson/lognormal heads)
- Distributional Forests (probabilistic random forests)

Intermittent demand (zeros/spikes)
- Croston / SBA / TSB (with simulated predictive distribution)
- Zero‑inflated variants (ZIP/ZINB) within distributional regression

Deep (time permitting)
- Temporal Convolutional / TFT quantile outputs; likely limited given schedule

Calibration/aggregation
- Conformalized quantiles; isotonic calibration
- Hierarchical reconciliation (store/product hierarchy) for distributions

### 3) Feature set (initial)

- Calendar: week‑of‑year, holiday flags, promo flags (if available)
- Recent stats: rolling mean/std/cv (SURD‑aware), trend slope, seasonality indicators
- Intermittency: zero‑rate, span since last nonzero
- Hierarchy metadata: department, product group, store format
- Lagged demand: lags 1,2,4,8 (on imputed series)

### 4) Evaluation & selection

Primary metrics
- Pinball loss averaged across quantiles
- Coverage vs nominal (80/90/95%) and interval width
- CRPS (optional)

By segment
- Evaluate per ABC/XYZ; pick champion + top 2 challengers per segment

Diagnostics
- Calibration plots; lift vs original; error by scale/intermittency

### 5) Runtime & compute expectations

- Training per model should fit inside overnight window when parallelized
- Use pilot slices (e.g., 2 stores × N products × 26 weeks) to estimate ETA
- Cap BLAS threads (1) and control `--n-jobs` for concurrency

### 6) Risks & mitigations

- Intermittent demand mis‑specification → include Croston family, zero‑inflated heads
- Miscalibration → conformal/isotonic calibration pass
- Overfitting across SKUs → global models with regularization; segment‑wise training
- Runtime blow‑ups → pilot, checkpoint/resume per model/SKU

### 7) Down‑selection criteria

- Calibration: coverage within ±2–3% of nominal
- Sharpness: lower interval width at same coverage
- Robustness: stable across weeks/stores; acceptable runtime/memory

### 8) Initial short‑list for implementation

Champion contenders (per segment)
- ETS with PI (fast, strong baseline)
- LightGBM Quantile (q∈{0.1..0.9} or config grid)
- NGBoost (lognormal/Poisson heads; handles skew)

Challengers
- Seasonal Naive + bootstrap (yardstick)
- Prophet/TBATS (seasonal heavy)
- Croston/TSB (XYZ‑Z high‑zero SKUs)

### 8b) Additional candidates and priorities

Below we record additional options with density readiness, typical fit, and triage rank (A=prioritize, B=targeted/compare, C=curiosity).

- Zero‑Inflated (ZIP/ZINB / hurdle)
  - Density: native (count distributions)
  - Fit: intermittent/low‑volume (XYZ‑Z)
  - Priority: A

- Hierarchical Bayesian (partial pooling: Poisson/NB/lognormal; BSTS/DLM)
  - Density: native (posterior predictive)
  - Fit: leverage cross‑SKU/store pooling; costly
  - Priority: B (slices/segments)

- Genetic Algorithms (metaheuristic)
  - Density: n/a (search for features/hparams/ensemble weights)
  - Fit: wrapper around QR/LGBM‑Q/NGBoost/GAMLSS
  - Priority: B

- Case‑Based Reasoning (nearest‑neighbor profile)
  - Density: empirical from neighbor set; akin to selective bootstrap
  - Fit: strong baseline for similarity‑driven demand
  - Priority: A‑

- Naive / Rolling averages
  - Density: via residual bootstrap or conformal prediction
  - Fit: baseline yardsticks per segment
  - Priority: A

- Regression families
  - Linear Quantile Regression; GAMLSS (NB/lognormal); Quantile Random Forest
  - Density: native (QR/GAMLSS), empirical (QRF)
  - Priority: A

- SLURP selective bootstrap (feature‑conditional resampling)
  - Density: native via SIP resampling
  - Fit: quick probabilistic baseline using SIP library
  - Priority: A

- Prophet (curiosity/compare)
  - Density: intervals; bootstrap for calibration
  - Priority: B

- Deep models: DeepAR / LSTM ("DeepR"→DeepAR, "LTMS"→LSTM), TFT
  - Density: native (Gaussian/NB heads or quantile)
  - Fit: higher‑volume segments; compute‑heavy
  - Priority: B

- Metalog (flexible distribution from quantiles)
  - Density: native from fitted quantile curves; caution on tails
  - Priority: A‑ (as a calibration/wrapper over quantile outputs)

### 8c) Metaheuristics for forecasting vs optimization

- Particle Swarm Optimization (PSO)
  - Forecasting: tune ARIMA/ETS/GAMLSS/ensemble weights under pinball/NLL — Rank B
  - Optimization: search policy parameters with simulation — Rank A

- Simulated Annealing (SA)
  - Forecasting: global search for rough likelihood/quantile landscapes — Rank B
  - Optimization: robust discrete search (use CRNs, checkpoint) — Rank A

- Genetic Algorithms (GA)
  - Forecasting: feature/hparam/ensemble search — Rank B
  - Optimization: optional alternative to PSO/SA — Rank B

### 9) Pilot protocol

1) Choose slice per segment (small but representative)
2) Train candidates; record metrics + runtime
3) Select champion + two challengers per segment; finalize hyperparams

### 10) Artifacts & logging

- Save per‑model artifacts under `models/<model>/<segment>/...`
- Log metrics table per segment; include data hash/version for idempotent resume

### 11) Next actions (when executing)

- Implement evaluation harness (pinball, coverage, CRPS optional)
- Implement calibration step (isotonic/conformal)
- Add checkpoint/resume per model and segment


