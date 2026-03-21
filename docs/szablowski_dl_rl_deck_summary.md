# Szabłowski DL/RL Deck Summary — Post-Competition Target Solution

**Title:** "Deep Learning for Demand Forecasting / Reinforcement Learning for Inventory Decisions"  
**Author:** Bartosz Szabłowski  
**Context:** Post-competition presentation showing the solution he intended to submit but finished after VN2 ended. Extends the winning CatBoost pipeline with a neural forecasting architecture and RL-based ordering policy.

---

## Architecture Overview

Same two-stage predict-then-optimize structure as the winning solution, but both stages are replaced with neural approaches:

- **Stage 1 (Forecast):** CatBoost → Custom TCN + FiLM + Learnable Fourier Seasonal Head
- **Stage 2 (Policy):** Analytical newsvendor (φ√D) → PPO actor-critic with learned demand multiplier and safety buffer

---

## Stage 1: Deep Learning Forecaster

### Inputs
- `X_i[t-L:t]` — sales history (context length L=20)
- `stock_i[t-L:t]` — stock availability flag history
- `store_id`, `product_id` — static covariates
- `week_i[t]` — calendar week

### Architecture Components

**Per-Series Scaling:** Same annualized scaling as the CatBoost paper: `S = max(mean(sales[t-53:t] | in_stock) * 53, 1)`. Applied before model input. Targets also scaled. Standardization (z-score) fitted on train set only.

**Data Augmentation (train mode only):**
- TimeAugmenter: jitters temporal position
- WeekAugmenter: perturbs calendar week
- StaticCovAugmenter: randomly swaps store/product IDs
- Input Dropout: zeros out portions of the input
- Rationale: time series data is far smaller than NLP/CV; augmentation injects diversity to reduce overfitting.

**Store/Product Embeddings:** Learned lookup tables mapping each store ID and product ID to dense vectors. Concatenated to form a combined entity embedding.

**TCN (Temporal Convolutional Network):**
- 1×1 Conv input projection: 2 channels (sales, stock) → hidden channels. Channel mixing only, no temporal mixing.
- Stack of TCN blocks with exponentially increasing dilation (1, 2, 4, 8). Each block: CausalConv1d → GroupNorm → GELU → Dropout → CausalConv1d → GroupNorm → GELU → Dropout + residual connection.
- Causal convolutions ensure no future leakage. Dilation expands receptive field to cover full context window.
- Output: `h_last = h[:, -1]` — takes only the last timestep embedding (shape: [hidden]).

**FiLM (Feature-wise Linear Modulation):**
- Combines TCN context vector with store/product embeddings.
- Embeddings pass through Linear → SiLU → Dropout → Linear to produce γ (scaling) and β (shift) parameters.
- γ is squashed via `1 + Tanh(input)` to keep scaling near 1.
- Modulated context: `γ * TCN_output + β`.
- **Why FiLM over concatenation:** Simple concatenation risks shortcut learning — the model can memorize IDs and ignore temporal context. FiLM forces the embeddings to modulate the context rather than replace it.

**Seasonal Head:**
- Inputs: `week_i[t]` → Fourier terms sin/cos [H, K] for K harmonic features across H=3 horizons.
- Concatenated ID embeddings → Linear → SiLU → Dropout → Linear → Fourier coefficients (A, B) + gating logit.
- Sin/cos terms weighted by learned coefficients, summed, passed through Tanh, then gated by Sigmoid.
- Output: `exp(gated_seasonal_signal)` — multiplicative seasonal adjustment per horizon.
- Effect: series with strong seasonality get large adjustments; series with no seasonality get gate ≈ 0 → multiplier ≈ 1.

**Decoder:** FiLM output → MLP → Softplus (enforces non-negative forecasts) → elementwise multiply by Seasonal Head output → `ŷ_i[t+1:t+3]`.

### Training Setup
- **Loss:** Masked Huber (Smooth L1) on scaled targets. Mask M ignores periods where sales are censored by stockouts. Multi-horizon: average across t+1, t+2, t+3.
- **Optimizer:** AdamW. Betas (0.9, 0.999). Weight decay on most parameters; no decay on bias + LayerNorm.
- **Stability:** AMP (mixed precision) + gradient scaling + gradient clipping.
- **LR Schedule:** Base lr=1e-4. Linear warmup (0.1×lr → lr over 5 epochs) then cosine annealing (lr → 1e-6).
- **Train/Val/Test split:** Test = competition simulation weeks. Val = 8 decision weeks before test. Train = all earlier weeks.
- **Early stopping:** Patience=50 epochs on validation criterion.
- **Fine-tuning step:** After best validation checkpoint, fine-tune with small lr (3e-5) on *unscaled* values to improve calibration on high-volume items. Rationale: big-volume items drive most revenue and operational risk.

### Forecast Results (WAPE on test set)

| Model | H1 | H2 | H3 |
|---|---|---|---|
| Deep Learning (TCN) | 54.31% | 57.83% | 59.08% |
| Global Model (CatBoost) | 56.14% | 58.24% | 59.74% |
| ARIMA | 56.65% | 58.42% | 60.29% |

TCN improves ~1.5–2 WAPE points over CatBoost across all horizons.

---

## Stage 2: Reinforcement Learning Policy

### MDP Formulation
- **State:** Demand forecast (d_{t+1}, d_{t+2}, d_{t+3}), on-hand inventory (I_t), in-transit (T_{t+1}, T_{t+2}), week, store/product IDs.
- **Action:** Order quantity Q_t (placed now, arrives after lead time).
- **Reward:** R_t = −cost = −(holding + shortage).

### Architecture

**Frozen components:** Store/Product Embeddings and Seasonal Head from the trained forecaster. Weights frozen during RL training.

**Build Nodes:** Constructs inventory physics nodes for t+1, t+2, t+3. Each node contains: start inventory, transit, demand, end inventory, shortage, and seasonal head output. Nodes are linked causally: `start_t = end_{t-1} + transit_t`, `short_t = max(demand_t - start_t, 0)`, `end_t = max(start_t - demand_t, 0)`.

**Node Embedding:** Each node's 6 features → Linear → SiLU → LayerNorm, plus step embedding (learned for positions 1,2,3), plus projected ID embeddings. All summed → Node Embeddings [H, hidden].

**Chain Graph:** Stack of ChainGraphLayers. Each layer: Input nodes + shifted input nodes (previous node's output; zeros for first) → separate Linear projections → concatenated → SiLU → LayerNorm → output nodes. Captures temporal dependencies across the 3-step horizon.

**Policy Head:** Last node → Linear → Tanh → Linear → 2 outputs:
- Action 1 (Demand Multiplier): `1 + Tanh(output)` → range [0, 2]
- Action 2 (Safety Buffer): `Softplus(output)` → non-negative

**Value Head:** Last node → Linear → Tanh → Linear → scalar value estimate.

**Log Std:** Learned parameter for the Gaussian policy distribution.

### Action Interpretation
```
E1 = max(I0 + T1 - d1, 0)
E2 = max(E1 + T2 - d2, 0)
base = max(d3 - E2, 0)
order = round(max(base × multiplier + buffer × scale_factor, 0))
```

### Training
- PPO (Proximal Policy Optimization) with GAE (Generalized Advantage Estimation).
- Training episodes: random initial inventory state, random start date from training set.
- Observations scaled by per-series scale factor.
- Stochastic policy during training (sample from squashed Gaussian); deterministic (policy mean) at validation.
- Early stopping on validation total cost.

### Decision Results

| Approach | Total Cost | Cost Reduction vs Benchmark |
|---|---|---|
| RL Policy (TCN forecast) | 3,582€ | 17.3% |
| Analytical Policy (CatBoost forecast) | 3,763€ | 13.2% |
| Competition Benchmark | 4,334€ | — |

---

## Author's Own Assessment (Slide 52)
- Small dataset and short test horizon → "better results may still be FART, no real robustness."
- Slight improvement with much higher complexity → "is the extra effort really worth it?"

---

## Relevance to Our Framework

### What's transferable
- **TCN+FiLM architecture for density forecasting:** The architecture is one output-head swap away from producing distributional forecasts (replace Softplus point output with distributional parameters — NegBin, quantile set, or mixture density — and train under CRPS or log-likelihood). The FiLM conditioning already provides per-series modulation, meaning the *shape* of the predictive distribution could vary by store-product. The learned Fourier seasonal head could modulate not just the mean but the scale/shape of the density during seasonal transitions.
- **Augmentation strategies:** Time, week, static covariate, and input dropout augmentation for time series models with limited data. Potentially useful for DeepAR or any neural density model in our ensemble.
- **Fine-tuning on unscaled values:** A calibration step that re-weights the model toward high-volume items. Relevant to our observation that large-volume series drive portfolio cost.

### What's not transferable (as-is)
- **RL policy:** Learns an empirical approximation of the newsvendor bias through trial-and-error rollouts. Does not propagate a density through the cost function. Does not compute or use the critical fractile analytically. The demand multiplier + safety buffer action space is essentially a learned version of the φ√D heuristic with an additional degree of freedom. The Jensen's Gap remains unaddressed.
- **Point forecast output:** Both the CatBoost and TCN pipelines produce point forecasts. The policy stage never sees a distribution. The RL agent's stochastic exploration during training is not a substitute for density-based decision-making — it explores *actions*, not *demand scenarios*.

### Key observation
The RL agent's 4.1 percentage point improvement over the analytical policy could come from three sources: (a) better forecasts (TCN vs CatBoost), (b) the multiplier+buffer having more expressiveness than φ√D, or (c) sequential/multi-period effects the single-period newsvendor misses. The deck does not ablate these. A three-way policy comparison (his analytical, his RL, our SIP density) on shared forecasts would disentangle them.
