# Szabłowski arXiv:2601.18919v1 — VN2 Winner Report Summary

**Paper:** "One Global Model, Many Behaviors: Stockout-Aware Feature Engineering and Dynamic Scaling for Multi-Horizon Retail Demand Forecasting with a Cost-Aware Ordering Policy"  
**Author:** Bartosz Szabłowski  
**Submitted:** 2026-01-26  
**Result:** 1st place, VN2 Inventory Planning Challenge (3,763€ vs 4,334€ benchmark, 13.2% reduction)

---

## Architecture: Two-Stage Predict-Then-Optimize

### Stage 1 — Global CatBoost Forecaster
- **Direct multi-horizon strategy:** Three independent CatBoost regressors for h=1, h=2, h=3.
- **Stockout-aware effective sales:** Masks weeks where `in_stock=False` to NaN before computing any features. All target-based features computed on this effective sales series `y_eff`.
- **Feature groups:**
  - *Demand level:* Short-term lags (t, t-1, t-2, t-3), seasonal lags (t-51, t-52, t-53), rolling means/medians (windows 3, 5, 13), EWM (spans 5, 10), rolling std, rolling IQR.
  - *Trend/momentum:* Δ1 and Δ5 differences, 4-week rolling slope proxy.
  - *Seasonality:* `week_of_year` as categorical, Fourier sin/cos for harmonics k=1,2,3, last-year-window average, seasonality-strength proxy (correlation between current and seasonal lags).
  - *Intermittency/spikes:* Rolling median + MAD → robust z-score, `time_since_spike`, rolling non-zero rate over 12 weeks.
- **Dynamic per-series scaling:** `scale_factor(i,t) = max(53 * mean_nonmissing(y_eff[t-52:t]), 1)` with warm-start proxy (expanding mean, annualized) when fewer than 45 in-stock observations. Approximates annualized demand level for normalization.
- **Two-level median imputation:** Per-series median of scaled values first, then global scaled median for remaining NaNs.
- **Time-decayed observation weights:** 1.0 for most recent ~53-week block, 0.5 for preceding block, 0.25 for block before that. Emphasizes recent behavior without truncating history.
- **Training loss:** RMSE on scaled target. Hyperparameter selection via Optuna (100 trials, TPE) minimizing MAE in original units on chronological validation split (most recent 10% of training weeks). Early stopping at 500 rounds.
- **Top features by importance:** `week_of_year` dominates (22–36%), followed by Fourier terms, Store/Product categoricals, and `seasonality_strength`.

### Stage 2 — Cost-Aware Ordering Policy
- Post-process forecasts: round to nearest integer, clip ≥ 0.
- Project inventory forward through t+1, t+2 using point forecasts and lost-sales assumption.
- Newsvendor critical fractile: `q* = cs/(cs+ch) = 1.0/1.2 ≈ 0.8333`.
- Map to normal quantile: `zq = Φ⁻¹(q*) ≈ 0.9674`.
- Uncertainty proxy: `σ(i,t+3) = φ * √(D̂(i,t+3))` where φ is a single global scalar calibrated on validation window by minimizing total cost.
- Target stock: `B(i,t+3) = D̂(i,t+3) + zq * σ(i,t+3)`.
- Order: `Q(i,t) = max(B(i,t+3) - Ĩ(i,t+3), 0)`.
- φ and zq kept separate intentionally: zq is cost-driven, φ captures data-dependent buffer need. φ can be extended per-horizon, per-segment, etc.

---

## Key Design Principles
1. **Global model, not local:** Single model across 599 series. Cross-series learning via categorical store/product IDs. Heterogeneous behaviors handled through feature engineering + scaling, not separate models.
2. **Stockout censoring handled at feature level:** Masking out-of-stock periods before feature computation prevents downward bias from censored sales. Distinct from distributional approaches to censoring.
3. **Scaling decouples level from pattern:** Per-series annualized scaling lets the model learn relative dynamics rather than absolute magnitudes. Analogous in purpose (not mechanism) to SURD variance-stabilizing transforms.
4. **Simple policy, strong forecast:** The ordering policy is a lightweight analytical rule. All the complexity is in the forecast. The paper explicitly states "strong performance does not necessarily require a complex decision model when the objective is explicitly defined."

---

## Limitations and Extensions Noted by Author
- The √D uncertainty proxy is a convenience heuristic; production would use empirical quantiles or probabilistic forecasts.
- The normal mapping for the safety factor is a lightweight approximation; retail errors are rarely Gaussian.
- Single global φ could be extended to per-segment or per-series.
- Extensions suggested: probabilistic decision-making (learn predictive distributions, compute quantiles directly), censored regression for latent demand, simulator-based joint tuning, hybrid end-to-end approaches.

---

## Relevance to Our Framework
- **What he does well:** Feature engineering for heterogeneous retail demand under stockout censoring. The scaling and weighting schemes are simple, effective, and potentially transferable to our models.
- **What he doesn't do:** Density forecasting. The entire distributional information is collapsed into a single scalar φ. There is no propagation of a full distribution through the cost function. The Jensen's Gap between E[f(x)] and f(E[x]) is unaddressed.
- **Direct comparison opportunity:** His analytical newsvendor (point forecast + φ√D buffer) vs our SIP newsvendor (full density through cost function) on shared forecasts through a shared simulator tests Jensen's Gap empirically.
