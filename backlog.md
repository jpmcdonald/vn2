## Backlog / Next-Week Considerations

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


- [ ] Deep analysis: risk/payoff around critical fractile (p*=cu/(cu+co)), curvature near CF, breakpoint between EV vs CF-driven ordering; derive diagnostics and plots; per-SKU and portfolio views.
