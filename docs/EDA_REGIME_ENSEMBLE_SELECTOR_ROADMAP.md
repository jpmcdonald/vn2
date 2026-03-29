# EDA expansion, time series regimes, and ensemble model selection

**Status:** Planned (see [backlog](../backlog.md))  
**Last updated:** 2026-03-28

This document captures a coordinated initiative: deepen exploratory analysis of demand series, turn diagnostics into features, define and detect **regimes**, and **compare** whether regime-aware signals improve **per-SKU ensemble / model selection** relative to the current fold-cost and composite-metric selectors.

---

## 1. Context

### 1.1 What we already do

- **Core EDA** ([`docs/eda/EDA_SUMMARY.md`](eda/EDA_SUMMARY.md), notebook `notebooks/02_comprehensive_time_series_eda.ipynb`, [`szablowski/generate_eda_base.py`](../szablowski/generate_eda_base.py)): summary statistics (including ADI / intermittency), ADF / KPSS, ARCH-LM heteroskedasticity, ACF/PACF, frequency domain, bootstrap.
- **Extensions** ([`szablowski/eda_extensions.py`](../szablowski/eda_extensions.py)): Taylor’s law (variance–mean scaling), STL seasonal strength, CUSUM-style structural break flags, Ljung–Box on levels, NegBin vs Poisson / zero-inflation / dispersion ratio.
- **Artifacts:** `summary_statistics.parquet`, `stationarity_tests.parquet`, optional `*_ext.parquet` merges.
- **Training features** ([`src/vn2/forecast/features.py`](../src/vn2/forecast/features.py)): rolling lags, intermittency proxies (zero rate, ADI, weeks since nonzero), etc.—**not** a direct join of frozen EDA test outputs into the default feature matrix.
- **Analysis joins:** [`szablowski/harness.py`](../szablowski/harness.py) `join_eda_artifacts()` merges EDA parquets into per-SKU-week detail for stratification and hypothesis work.
- **Champion selector today:** primarily **last-N-fold realized cost** with critical-fractile tie-breakers ([`src/vn2/analyze/model_selector.py`](../src/vn2/analyze/model_selector.py)); alternate path: composite / CRPS / Wasserstein on fold detail ([`scripts/build_dynamic_selector.py`](../scripts/build_dynamic_selector.py)). Neither path consumes EDA regime labels today.

### 1.2 Nomenclature: “SURD” in this repo

Two meanings appear in code and docs; keep them explicit in any regime or feature spec:

| Meaning | Role | Typical artifacts |
|--------|------|-------------------|
| **SURD–transform** | Per-SKU variance-stabilizing transform (CV-based choice among log, log1p, sqrt, cbrt); `SURDWrapper`, `surd_transforms*.parquet`, `*_surd` models | Modeling pipeline, H3 |
| **SURD–information** | MI / entropy–style exploratory or post-hoc analysis (notebook MI demo, [`scripts/surd_decomposition_analysis.py`](../scripts/surd_decomposition_analysis.py)) | Insight, reporting |

Regime features and selector experiments should state which “SURD” signals (if any) enter as **transform choice**, **CV reduction**, or **information / redundancy metrics**.

---

## 2. Goals

1. **Expand EDA** where gaps matter for retail demand: e.g. richer intermittency taxonomy (ADI + CV² classes), seasonal unit-root or seasonal strength consistency, formal change points beyond simple CUSUM, optional panel-aware summaries.
2. **Build features** from tests and rolling analogues: both **static per-SKU** (from historical window) and **time-varying** (rolling recomputation or change-point indicators) for use in tabular models and routers.
3. **Classify series into regimes** using those features (supervised labels from past selector wins, or unsupervised clusters, or rule-based buckets aligned to SBC/XYZ-style practice).
4. **Detect / predict regime changes** (structural breaks, drift in CV or zero rate, forecast-error regime shifts) so selection is not frozen on full-history EDA alone.
5. **Evaluate selector methods** side by side on the same backtest: baseline (cost + CF tie-break), composite-only, **regime-conditioned rules** (e.g. intermittent → prefer Croston-like or specific arms), **learned router** (gradient boosting / meta-classifier on regime + recent metrics), and optional **bandit** (see backlog §1).

### 2.1 Outcome-PMF entropy signals (VN2 extension)

Parallel track documented in [`ENTROPY_REGIME_FRAMEWORK_VN2.md`](ENTROPY_REGIME_FRAMEWORK_VN2.md):

- **Optional regime inputs:** Time-varying **Shannon entropy** of demand PMFs and of **cost-outcome** PMFs (after mapping demand through the single-period cost function at the relevant inventory position), plus Gaussian-reference **entropy gap** and **Jensen gap** trajectories. These are **discrete PMF / Shannon** metrics — not the same artifact as SURD–information (MI decomposition); label features clearly in the feature registry.
- **Selector benchmarks:** **H11** compares demand-scalar vs demand-entropy vs outcome-PMF-based regime predictors; **H12** tests an **entropy-gap** selector (model PMF vs empirical demand histogram) against composite (pinball × Wasserstein) and fold-cost selectors. Include both in the §3.5 harness when that script exists.

---

## 3. Proposed workstreams

### 3.1 EDA expansion

- Inventory current outputs vs. backlog hypotheses (heteroskedasticity, tails, CF-local shape).
- Add diagnostics that are **actionable** for model choice: e.g. Syntetos–Boylan classification, multi-break tests where justified, residual regime labels aligned with [`szablowski/residual_diagnostics.py`](../szablowski/residual_diagnostics.py).
- Document expected columns and refresh policy (full history vs. rolling window) in [`quarterdeck/governance/DATA_CONTRACT.md`](../quarterdeck/governance/DATA_CONTRACT.md) if new parquets are added.

### 3.2 Feature layer

- **Static join:** merge `summary_statistics*`, `stationarity_tests*` (and extensions) into a canonical **per-SKU feature table** with stable schema versioning.
- **Dynamic features:** rolling ADI, rolling ARCH proxy, rolling Taylor α, distance from last detected break, etc., aligned with training cutoff rules to avoid leakage.
- **SURD–transform:** optional features = `best_transform`, transform-induced CV reduction, or flags for “strong heteroskedasticity → prefer surd arm.”
- **SURD–information:** optional features from MI/entropy decomposition scripts where reproducible per SKU/week.
- Wire selected columns into **meta-router** training ([`src/vn2/forecast/meta_router.py`](../src/vn2/forecast/meta_router.py)) and/or cohort rules ([`src/vn2/analyze/ensemble.py`](../src/vn2/analyze/ensemble.py)) as experiments—not as silent changes to the production champion path until validated.

### 3.3 Regime definition and labeling

- **Unsupervised:** cluster on EDA + scale features; interpret clusters and map to model families.
- **Supervised:** use historical “best model per SKU” from eval folds as a pseudo-label; train a multiclass classifier; calibrate and check stability across folds.
- **Hybrid:** rule-based gates (e.g. high ADI) then classifier within segment.
- Deliver a **regime catalog** (definitions, prevalence, stability across rolling origins).

### 3.4 Regime change

- Online or rolling **change detection** on demand or on forecast errors (cheap CUSUM on mean/variance; optional dedicated libraries if needed).
- Features: time since break, probability of break, drift score vs. training window.
- Hypothesis: **dynamic** regime features outperform **static** EDA-only features for selector accuracy when non-stationarity is common (consistent with current ADF/KPSS split in EDA).

### 3.5 Ensemble / selector comparison (evaluation harness)

- Hold **forecast checkpoints fixed**; vary only the **selection policy**:
  - A. Baseline: current `select_per_sku_from_folds` (cost + CF tie-break).
  - B. Composite / Wasserstein static selector (existing script pattern).
  - C. Regime rules: if-then tables from EDA buckets → model family.
  - D. Learned router: features from §3.2–3.4 → predicted best model; train on inner folds, evaluate on outer.
  - E. Optional: contextual bandit (backlog §1) as upper-complexity reference.
- **Metrics:** portfolio SIP cost (primary), pinball at critical fractile, CRPS, calibration slices **by regime**.
- **Guardrails:** no leakage from test folds into transform or selector training; document fold alignment with [`scripts/build_dynamic_selector.py`](../scripts/build_dynamic_selector.py) and `full_L3_simulation` selector maps.

---

## 4. Deliverables

| Deliverable | Description |
|-------------|-------------|
| EDA v2 spec | List of new tests, columns, and parquet layouts |
| Feature registry | Table of features → source → use (router vs cohort vs paper only) |
| Regime playbook | Definitions, labeling code, cluster/rule documentation |
| Selector benchmark script | One entrypoint comparing policies A–E on shared eval artifacts |
| Short results note | Whether regime-aware selection beats baseline on cost and by which margin |

---

## 5. Dependencies and risks

- **Data vintage:** if `demand_long` is extended with competition weeks, align regime EDA with the **Retrain all models on clean pre-competition data** initiative in the root `backlog.md`.
- **Complexity:** regime classifiers can **overfit** fold noise; use nested or walk-forward evaluation mirroring existing eval_folds discipline.
- **Naming:** always disambiguate SURD-transform vs SURD-information in specs and PRs.

---

## 6. Related documents

- [`docs/eda/EDA_SUMMARY.md`](eda/EDA_SUMMARY.md)
- [`docs/RECOMMENDED_NEXT_STEPS_2026-03-12.md`](RECOMMENDED_NEXT_STEPS_2026-03-12.md)
- [`docs/VN2_BACKTESTING_SPRINT_PROMPT.md`](VN2_BACKTESTING_SPRINT_PROMPT.md)
- [`docs/ENTROPY_REGIME_FRAMEWORK_VN2.md`](ENTROPY_REGIME_FRAMEWORK_VN2.md)
- [`quarterdeck/governance/MODEL_ONTOLOGY.md`](../quarterdeck/governance/MODEL_ONTOLOGY.md)
