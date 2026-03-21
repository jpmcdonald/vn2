# Recommended Next Steps for Backtest-Driven Improvement

**Created:** 2026-03-12

This document captures suggested next steps for incremental performance improvement through backtesting, plus testable hypotheses and references to the existing backlog.

---

## 1. Next Steps for Incremental Improvement (Backtesting)

### 1.1 Stabilize the comparison baseline
- Use a single eval setup: same 8 folds, same cost definition (`sip_realized_cost_w2`) everywhere.
- Use one selector rule: either composite (pinball×Wasserstein) or cost-based, with the **fixed** "last 8 folds" logic so the map has 599 SKUs.
- Re-run `compare_selector_metrics` and treat composite vs cost-based (e.g. €4,487 vs €4,688) as the baseline; any change should be compared on this same backtest.

### 1.2 Selector improvements
- **Cost-based vs composite:** Cost-based is ~€200 worse in the latest run. Try hybrid rules: e.g. use cost-based only when a model has 8-fold coverage and cost is clearly lower; otherwise keep composite.
- **Tie-breaking and thresholds:** In `model_selector`, add or tune tie-breakers (e.g. `pinball_cf_h2`, `hit_cf_h2`) and optionally require a minimum cost improvement over the next-best model before switching.
- **Cohort rules:** Use the backlog's "cohort-gated selector" idea (e.g. rate_bin, cv_bin, stockout_bin) so selection differs by demand type; backtest each variant on the same 8-week run.

### 1.3 Forecast improvements that feed the backtest
- **Calibration at CF:** The paper identifies calibration as the binding constraint. Add post-hoc calibration (e.g. conformal at τ=0.833) and re-run the **same** backtest; measure Δ total cost.
- **Per-SKU service level:** Test a small CF delta around 0.833 (e.g. ±0.02) per SKU or per cohort; backtest and compare total cost.
- **SLURP/PMF fidelity:** Increase PMF grain and check quantile→PMF near the critical fractile; re-run backtest to see if cost improves.

### 1.4 Make backtesting cheaper and repeatable
- **Strict 8-of-8 selector:** As in the backlog, ensure the selector only uses SKU–model pairs with full 8-fold coverage so the backtest is comparable across runs.
- **Per-SKU, per-week costs:** Emit and store per-SKU, per-week realized costs in the eval pipeline so you can aggregate over "last N decision-affected weeks" without re-running the full simulation.
- **Automate one "golden" run:** One script that (1) builds eval_folds, (2) builds the chosen selector, (3) runs `full_L3_simulation` once, (4) writes total cost and summary. Use it for every change (selector, calibration, CF delta, etc.).

---

## 2. Hypotheses Worth Examining

### 2.1 Worse financial outcome with better point forecast
**Yes, it's possible, and the codebase already has evidence.**

- LightGBM has **better** median MAE (1.91 vs 1.62–1.64 for SLURP) and better fill rate (88.7% vs ~80%), but **worse** 8-week cost (€6,756 vs €5,169) because it over-predicts and incurs much higher holding cost.
- The paper (e.g. §5.1, §4.4) and `docs/paper/revised_paper.md` spell this out: point accuracy (MAE, fill rate) can improve while cost worsens when the **distribution** is miscalibrated (e.g. upper quantiles too high), so the newsvendor policy orders too much.
- **Hypothesis to test formally:** "Among models with similar MAE, the one with better calibration at the critical fractile (e.g. hit_cf_h1 ≈ 0.833) has lower backtest cost." Run a correlation or rank comparison: MAE vs cost, and hit_cf vs cost, on the same backtest.

### 2.2 Better pinball at CF can imply worse cost (path dependence)
- The paper (§5.4) notes that slurp_stockout_aware has slightly better pinball at 0.833 but worse 8-week cost than slurp_bootstrap, and that CRPS/Wasserstein align better with simulation cost.
- **Hypothesis:** "Single-period pinball at CF is a weak proxy for multi-period cost when lead times and in-transit inventory create path dependence." Test by ranking models by (a) mean pinball_cf over folds vs (b) 8-week backtest cost, and by correlating per-SKU pinball_cf with per-SKU contribution to total cost.

### 2.3 Cost-based selection overfits to the eval folds
- The result (composite €4,487 vs cost-based €4,688) suggests that minimizing realized cost over the same 8 weeks used in the backtest does **not** win.
- **Hypothesis:** "Selecting by in-sample realized cost overfits to those 8 weeks; a proxy (composite) generalizes better." Test with a **holdout** design: build the selector on folds 0–5, backtest on weeks that use folds 6–7 (or a later period), and compare cost-based vs composite again.

### 2.4 Per-SKU CF beats a single global CF
- The paper shows that for miscalibrated models the "optimal" SL is 0.60, not 0.833.
- **Hypothesis:** "Allowing a small per-SKU or per-cohort CF band (e.g. 0.81–0.85) improves total cost vs fixed 0.833." Requires a policy that picks CF from demand/CV/stockout bins and then run the same backtest.

### 2.5 Conformal calibration at CF improves cost
- The backlog and paper stress calibration as the binding constraint.
- **Hypothesis:** "Post-hoc conformal calibration targeting the 0.833 quantile reduces backtest cost for models that are currently miscalibrated at CF." Implement conformal on the existing checkpoints, re-run the same backtest, compare total cost.

### 2.6 Diversity of models matters more than single-model quality
- The composite selector uses 9 models and beats the best single model (slurp_bootstrap €5,169) by a large margin (€4,487).
- **Hypothesis:** "Adding a new model family that is mediocre on average but strong on a subset of SKUs (e.g. high-CV or intermittent) reduces portfolio cost." Add one candidate model, re-build the selector and backtest, measure Δ cost.
- The TCN+FiLM+Seasonal Head architecture from the DL/RL deck (`docs/szablowski_dl_rl_deck_summary.md`) is a specific candidate with complementary inductive biases to tree-based and autoregressive models — convolutional temporal extraction vs. tree splits vs. autoregressive conditioning.

### 2.7 Neural density forecaster vs tree-based ensemble at the critical fractile
The TCN+FiLM+Seasonal Head architecture (see `docs/szablowski_dl_rl_deck_summary.md`) achieved ~1.5–2 WAPE point improvement over CatBoost as a point forecaster. The architecture is one output-head modification away from producing native density forecasts. **Hypothesis:** "A TCN with distributional output and learned per-series seasonal modulation produces better-calibrated densities at the critical fractile than tree-based quantile models for high-volume seasonal SKUs, reducing portfolio cost when added as an ensemble member." Test by training the modified architecture, plugging into SIP, and running the standard backtest. Compare per-cohort calibration and cost contribution to existing ensemble members.

---

## 3. Other Directions (from backlog and docs)

- **Online/bandit:** Contextual bandit for per-SKU model (or CF delta) with reward = −realized cost; run in shadow mode for 1–2 folds, then compare to the static selector in backtest.
- **Jensen gap by cohort:** Slice Jensen delta (cost(point policy) − cost(SIP)) by rate, CV, zero ratio, stockout rate; see where density helps most and target calibration or model choice there.
- **Deep dive at CF:** "Risk/payoff around critical fractile, curvature near CF, breakpoint between EV vs CF-driven ordering" (from backlog) as diagnostics; then test policies that adapt when curvature is flat vs steep.
- **Three-way policy comparison:** Analytical (φ√D) vs RL (learned multiplier+buffer) vs SIP (density-based) on shared forecasts; disentangles whether cost differences come from forecast quality, policy expressiveness, or multi-period sequential effects. See sprint extension in `backlog.md`.

### 3.1 Szabłowski sprint chain (backtest roadmap)

- **Phase A — CatBoost + harness:** Full spec in `docs/VN2_BACKTESTING_SPRINT_PROMPT.md` (Stages 1–2, H1–H6, calibration).
- **Phase B — Neural & policy extension (after CatBoost):** Same document, section **"Sprint extension: Neural architecture & policy comparison"** — residual diagnostics (ACF / heteroskedasticity / tails vs SLURP), TCN+FiLM distributional forecaster (CRPS, SIP arm), three-way policy test (φ√D vs RL vs SIP on shared simulator). **Blocked** until CatBoost sprint, shared simulator, and DL/RL code availability per that doc.

---

## 4. References

- **Selector comparison:** `scripts/compare_selector_metrics.py`, `reports/selector_comparison/comparison_summary.md`
- **Paper and hypotheses:** `docs/paper/revised_paper.md`, `docs/LESSONS_LEARNED_MEETUP.md`
- **Backlog:** `backlog.md` (includes Szabłowski sprint + neural/policy extension)
- **Szabłowski + extension plan:** `docs/VN2_BACKTESTING_SPRINT_PROMPT.md`
- **Evaluation and metrics:** `EVALUATION_V3_SUMMARY.md`, `docs/operations/model_evaluation.md`
