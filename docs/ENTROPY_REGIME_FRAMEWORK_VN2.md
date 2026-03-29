# VN2: Regime Detection and Entropy-Based Metrics

**Status:** Active research track (see [backlog](../backlog.md))  
**Implementation:** [`src/vn2/analyze/entropy_metrics.py`](../src/vn2/analyze/entropy_metrics.py), [`src/vn2/analyze/entropy_regime.py`](../src/vn2/analyze/entropy_regime.py), [`scripts/compute_entropy_regime_metrics.py`](../scripts/compute_entropy_regime_metrics.py), [`scripts/compute_entropy_regime_labels.py`](../scripts/compute_entropy_regime_labels.py), [`scripts/run_entropy_hypotheses_h7_h12.py`](../scripts/run_entropy_hypotheses_h7_h12.py); full eval: `uv run python src/vn2/analyze/model_eval.py --sip --state-path ... --entropy-metrics ...`.

This document adds research hypotheses and implementation tasks based on theoretical work connecting the VN2 density forecasting infrastructure to a broader regime detection framework. Read the existing backlog, hypotheses (H1–H6), and codebase structure when extending this work.

**Core insight:** The VN2 infrastructure already produces full demand PMFs per SKU per period. These PMFs contain more information than scalar summaries (pinball loss, Wasserstein distance, CRPS) used for model selection and evaluation. This work extends VN2 in two directions: (1) computing entropy-based regime indicators from PMFs, and (2) treating **outcome PMFs** (demand uncertainty propagated through the cost function) as primary objects for evaluation and regime detection alongside demand-side metrics.

---

## Theoretical Foundation: The Entropy Framework

### The Sacred Six Assumptions

Classical OR rests on six simplifying assumptions. When they hold, systems are tractable. When they break, systems transition between qualitatively different behavioral regimes (Simple → Complicated → Complex → Chaotic). The six assumptions, ordered by theoretical role:

1. **Deterministic vs. Probabilistic** — How much observed uncertainty is manufactured by data architecture vs. genuine? Calibration layer for all other indicators.
2. **Gaussian vs. Non-Gaussian** — Gaussian is the maximum entropy distribution for given mean and variance. Departure from Gaussianity reveals structure the Gaussian assumption discards.
3. **Linear vs. Non-Linear** — Non-linear transformation of a Gaussian produces a non-Gaussian. Collapses into Assumption 2: the level of entropy departure measures presence of non-linearity; the rate of change measures whether non-linearity is intensifying.
4. **Decomposability (Local vs. Global)** — Measured by mutual information between components relative to individual entropies. When coupling exceeds independent variability, decomposition fails.
5. **Static vs. Dynamic** — The Reynolds number analog. Ratio of forcing (uncertainty) to dissipative capacity (buffers + agility). When this ratio exceeds a critical threshold, the system transitions from damping perturbations to amplifying them.
6. **Single vs. Multiple Conflicting Objectives** — Prescriptive entropy. Unlike Assumptions 1–5, this cannot be reduced by better information — only by governance.

### Connection to VN2

- **Jensen's Gap IS Assumption 2 made financial.** Collapsing a PMF to a point forecast discards entropy. The Gap is the price paid for that information loss.
- **The critical fractile** is where Assumption 2 becomes an inventory decision. If the distribution is non-Gaussian, the 83.3rd percentile is in a different place relative to the mean than Gaussian assumes.
- **Pinball loss at q*** tests calibration at one point. But financial outcomes depend on the entire distribution interacting with the entire cost function.
- **Wasserstein distance** measures distributional shift but collapses full geometry to a scalar. Two distributions with the same W1 can have very different financial consequences when propagated through an asymmetric cost function.
- **Per-SKU model selection** is per-SKU regime classification — different SKUs operate in different regimes where different assumptions hold.
- **SURD transforms** address Assumption 2 directly (variance stabilization = handling heteroskedasticity departure from Gaussianity).
- **Stockout-aware modeling** is censored demand recovery — imputing manufactured entropy (Assumption 1) from supply-constrained observations.

---

## The Scalar Proxy Problem

### Current State

The VN2 model selector can use a composite metric: Pinball Loss × Wasserstein Distance (see [`scripts/build_dynamic_selector.py`](../scripts/build_dynamic_selector.py)). Separately, fold-cost selection uses realized SIP cost with CF tie-breakers ([`src/vn2/analyze/model_selector.py`](../src/vn2/analyze/model_selector.py)).

- **Pinball Loss at q* = 0.833**: Evaluates calibration at the critical fractile. Blind to much of the rest of the distribution.
- **Wasserstein Distance (W1)**: Single-number comparison; location vs tail changes can yield similar W1 but different costs under asymmetric holding/shortage.

### The Fix: Outcome PMFs

Instead of:

`Demand PMF → scalar summaries (pinball, wasserstein) → model selection / regime indicators`

Prefer:

`Demand PMF → cost function at inventory state → Outcome PMF on cost → regime indicators on outcome PMF`

VN2 already convolves demand with inventory logic for SIP. This program **persists** and **summarizes** the implied **cost-outcome** distribution per decision period where applicable.

---

## Conventions (implementation)

- **Entropy units:** All Shannon entropies in **nats** (natural log); `entropy_gap_gaussian` compares discrete Shannon \(H(p)\) to **differential** entropy \(H_{\mathcal{N}}(\mu,\sigma^2) = \tfrac{1}{2}\log(2\pi e \sigma^2)\) of the Gaussian with the same mean and variance as the discrete PMF. The **gap** is defined as \(H_{\mathcal{N}} - H(p)\) (larger when the discrete distribution is “peaked” relative to the variance-matched Gaussian).
- **Jensen gap (single-period newsvendor cost):** `E[cost(D)] - cost(round(E[D]))` with nonnegative integer demand support; `round(E[D])` is the plug-in mean demand for the point policy.
- **Empirical PMF (H12):** Histogram of recent **realized** demands on the training window of the fold (same support as PMF grain), normalized — documented to avoid lookahead; for strict tests use only history strictly before the decision week.

---

## New Hypotheses (H7–H12)

### H7: Outcome PMF Entropy as Regime Indicator

**Hypothesis:** Shannon entropy of the outcome PMF (financial consequence distribution) is a more reliable regime indicator than Shannon entropy of the demand PMF alone.

**Test:** Per SKU, per period: `H_demand`, `H_outcome`; track divergence when demand entropy is stable but outcome entropy moves (and vice versa). Compare predictive power for next-period cost degradation.

### H8: Jensen's Gap Trajectory as Regime Signal

**Hypothesis:** Jensen's Gap tracked over time per SKU is interpretable in dollar terms and signals growing value destroyed by deterministic planning.

**Test:** `E_outcome - outcome_at_mean` per period; correlate with demand shape features (CV, intermittency, skewness).

### H9: Entropy Gap from Gaussian as Non-Linearity Detector

**Hypothesis:** `H_gaussian - H_observed` (per convention above) correlates positively with Jensen's Gap magnitude.

**Test:** Correlate entropy gap with Jensen gap across SKUs/periods. Bridges to H1 (Jensen by regime).

### H10: Outcome PMF Sensitivity as Agility Measure

**Hypothesis:** `ΔH_outcome / ΔH_demand` across consecutive periods measures how financial uncertainty tracks demand uncertainty (fragility vs absorption).

**Test:** Correlate sensitivity with inventory position / buffer proxies where available.

### H11: Outcome PMF Shape Dynamics vs. Wasserstein

**Hypothesis:** Outcome-side entropy + tail summaries outperform demand-only scalars for predicting cost degradation and selector stability.

**Test:** Compare predictors from approaches 1–3 (composite scalars vs demand entropy vs outcome metrics).

### H12: Ensemble Diversity as Entropy Hedge

**Hypothesis:** Selector by **entropy gap** between model PMF and empirical demand PMF competes with pinball×Wasserstein composite.

**Test:** Build selector minimizing distributional gap; backtest total cost vs composite vs cost-based.

---

## Implementation Tasks

### Task 1: Entropy Computation Infrastructure

Module: [`src/vn2/analyze/entropy_metrics.py`](../src/vn2/analyze/entropy_metrics.py) — Shannon, Gaussian reference, entropy gap, Jensen gap, outcome PMF aggregation.

### Task 2: Outcome PMF Storage and Tracking

Phase A: Parquet from batch script (optional columns for sparse outcome support/prob). Phase B: optional scalar columns on `eval_folds` when `--entropy-metrics` is passed to evaluation.

### Task 3: Time Series of Entropy Metrics

Script builds rows per (store, product, model_name, fold_idx, horizon) with `H_demand`, `H_outcome`, `entropy_gap_demand`, `jensen_gap_cost`, then derives `sensitivity_ratio` where defined.

### Task 4: Regime Classification Prototype

Rule-based labels on entropy trajectories (stable / trending / volatile); join EDA features via [`szablowski/harness.py`](../szablowski/harness.py) when paths provided.

### Task 5: Backtest H7–H12

[`scripts/run_entropy_hypotheses_h7_h12.py`](../scripts/run_entropy_hypotheses_h7_h12.py) reads metrics parquet, writes CSV/MD summaries under `reports/entropy_hypotheses/`.

---

## Priority Order

1. Task 1 (entropy computation)  
2. Task 3 (time series)  
3. H9 → H7 → H8 → H10 → H11 → H12  
4. Tasks 2 and 4 (persistence hardening + regime labels)

---

## Connection to Broader Research Program

VN2 serves as an empirical test bed for a unified information-theoretic regime framework: full demand PMFs, outcome PMFs, 599 SKUs, 8-week sequential eval, ground-truth costs, 5:1 cost asymmetry. Results inform entropy-based monitoring and client supply-chain calibration.

**Dependencies:** If `demand_long` vintage changes (pre-competition retrain), re-run entropy pipelines on new checkpoints for comparable conclusions.
