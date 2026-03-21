# Cursor Task: Update VN2 Backlog with Neural Architecture & Policy Comparison Items

## Context

Two Szabłowski documents have been summarized and added to `docs/`:
- `docs/szablowski_arxiv_2601_18919_summary.md` — the winning CatBoost + analytical newsvendor solution (already covered by the existing Szabłowski sprint in the backlog)
- `docs/szablowski_dl_rl_deck_summary.md` — a post-competition extension replacing CatBoost with a TCN+FiLM+Seasonal Head deep learning forecaster and replacing the analytical ordering policy with a PPO reinforcement learning agent

The existing sprint (`docs/VN2_BACKTESTING_SPRINT_PROMPT.md`) and backlog (`backlog.md`) fully cover the arXiv paper. The DL/RL deck introduces new work that needs to be reflected in the backlog and recommended next steps.

## Files to read first

1. `backlog.md` — current backlog structure
2. `docs/RECOMMENDED_NEXT_STEPS_2026-03-12.md` — current recommended next steps
3. `docs/VN2_BACKTESTING_SPRINT_PROMPT.md` — existing sprint prompt (for context on H1–H6)
4. `docs/szablowski_dl_rl_deck_summary.md` — the new material to integrate

## Changes to make

### 1. Add new backlog section to `backlog.md`

Insert a new section **after** the existing "Sprint: Szabłowski comparison & hypothesis testing" block and **before** "1) Online improvement loop." Title it:

```
### Sprint extension: Neural architecture & policy comparison (post-CatBoost sprint)
```

Contents should include these three workstreams:

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

### 2. Update `docs/RECOMMENDED_NEXT_STEPS_2026-03-12.md`

**Under Section 2 "Hypotheses Worth Examining":**

Add a new subsection **2.7 Neural density forecaster vs tree-based ensemble at the critical fractile:**

> The TCN+FiLM+Seasonal Head architecture (see `docs/szablowski_dl_rl_deck_summary.md`) achieved ~1.5–2 WAPE point improvement over CatBoost as a point forecaster. The architecture is one output-head modification away from producing native density forecasts. **Hypothesis:** "A TCN with distributional output and learned per-series seasonal modulation produces better-calibrated densities at the critical fractile than tree-based quantile models for high-volume seasonal SKUs, reducing portfolio cost when added as an ensemble member." Test by training the modified architecture, plugging into SIP, and running the standard backtest. Compare per-cohort calibration and cost contribution to existing ensemble members.

**Under Section 2.6 (model diversity):**

Add a sentence noting that the TCN architecture from the DL/RL deck is a specific candidate with complementary inductive biases to tree-based and autoregressive models — convolutional temporal extraction vs. tree splits vs. autoregressive conditioning.

**Under Section 3 "Other Directions":**

Add a bullet:

> - **Three-way policy comparison:** Analytical (φ√D) vs RL (learned multiplier+buffer) vs SIP (density-based) on shared forecasts; disentangles whether cost differences come from forecast quality, policy expressiveness, or multi-period sequential effects. See sprint extension in `backlog.md`.

### 3. Add reference to the DL/RL deck in the sprint prompt

At the top of `docs/VN2_BACKTESTING_SPRINT_PROMPT.md`, in the Context section, after the sentence about the winning solution, add:

> Szabłowski also published a post-competition extension replacing CatBoost with a custom TCN+FiLM+Seasonal Head deep learning forecaster and the analytical ordering policy with a PPO reinforcement learning agent. This achieved 17.3% cost reduction vs benchmark (3,582€) compared to the CatBoost solution's 13.2%. Summary at `docs/szablowski_dl_rl_deck_summary.md`. The neural architecture and RL policy comparison are scoped as a separate sprint extension in `backlog.md`, to be started after the CatBoost sprint completes.

## What NOT to change

- Do not modify the existing sprint scope (H1–H6). The sprint prompt is implementation-ready for the CatBoost pipeline and should not be expanded.
- Do not change sequencing of existing backlog items 1–7.
- Do not assume access to Szabłowski's DL/RL code — flag it as a dependency.

## Validation

After making changes, verify:
- `backlog.md` has the new section in the correct position (after Sprint, before item 1)
- `RECOMMENDED_NEXT_STEPS_2026-03-12.md` has new subsections 2.7 and the additions to 2.6 and Section 3
- `VN2_BACKTESTING_SPRINT_PROMPT.md` has the new context paragraph
- No existing content was deleted or reordered
