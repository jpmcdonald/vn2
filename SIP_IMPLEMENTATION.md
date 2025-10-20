# SIP-Based Sequential Newsvendor Implementation

## Overview

This implementation provides decision-focused evaluation of density forecasts using **Stochastic Inventory Position (SIP)** mathematics for the sequential newsvendor problem.

## Key Features

### 1. SIP Optimization (`vn2/analyze/sip_opt.py`)

Core functions for discrete optimization under uncertainty:

- **`quantiles_to_pmf()`**: Converts 13-point quantile forecasts to 1001-point PMF via interpolation
- **`convolve_inventory()`**: Computes end-of-period inventory distribution: `I_end = (I0 + Q - D)^+`
- **`optimize_order()`**: Brute-force search over integer order quantities [0, 1000]
- **`compute_realized_metrics()`**: Validates decisions using actual demand

### 2. Integration with Evaluation Pipeline

Extended `vn2/analyze/model_eval.py`:

- **`compute_sip_cost_metric()`**: Orchestrates SIP optimization for each (model, SKU, fold)
- **`evaluate_one()`**: Now accepts `use_sip`, `sip_grain`, and `state_df` parameters
- **`run_evaluation()`**: Loads initial state from `state.parquet` when SIP is enabled
- **`aggregate_results()`**: Aggregates SIP metrics (sum for costs, mean for service levels)

### 3. CLI Integration (`vn2/cli.py`)

New flags for `eval-models` command:

```bash
python -m vn2.cli eval-models \
  --use-sip-optimization \
  --sip-grain 1000 \
  --out-suffix v4 \
  --models slurp_stockout_aware slurp_bootstrap seasonal_naive \
  --n-jobs 12 \
  --cpu-fraction 1.0
```

## Problem Formulation

### Sequential Newsvendor with Lead Time

**State at fold origin (week 0):**
- `I0`: On-hand inventory
- `Q1`: Order arriving at start of week 1 (already placed)
- `Q2`: Order arriving at start of week 2 (already placed)

**Decision:** Choose order quantity `Q*` for week 3 arrival (not evaluated in 2-week horizon)

**Dynamics:**
1. **Week 1:**
   - Starting inventory: `I1_start = I0 + Q1`
   - Sales: `S1 = min(D1, I1_start)`
   - Ending inventory: `I1_end = max(0, I1_start - D1)`

2. **Week 2 (decision period):**
   - Starting inventory: `I2_start = I1_end + Q2`
   - Sales: `S2 = min(D2, I2_start)`
   - Ending inventory: `I2_end = max(0, I2_start - D2)`
   - **Cost:** `c2 = 0.2 × I2_end + 1.0 × max(0, D2 - I2_start)`

**Objective:** Minimize `E[c2]` by choosing optimal `Q*` (for week 3, not used in evaluation)

### Why Week 1 Costs Are Excluded

- Week 1 state (`I0`, `Q1`) is deterministic and identical for all models
- Week 1 costs are a shared baseline that doesn't differentiate model performance
- Decision focus is on week 2, where forecast uncertainty matters

## SIP Mathematics

### PMF Construction

Convert quantile forecast to discrete PMF:

```python
# Input: 13 quantiles at levels [0.01, 0.05, 0.1, ..., 0.99]
# Output: PMF over [0, 1000] with 1001 points

pmf = quantiles_to_pmf(quantiles, quantile_levels, grain=1000)
# pmf[i] = P(D = i) for i in [0, 1000]
```

### Inventory Convolution

Compute end-of-week-1 inventory distribution:

```python
# I1_end = (I0 + Q1 - D1)^+
pmf_I1_end = convolve_inventory(I0, Q1, pmf_D1)
```

This handles:
- Stockouts: All demand > inventory collapses to `I_end = 0`
- Leftovers: Remaining inventory after sales

### Order Optimization

For week 2, we have:
- `pmf_I1_end`: Uncertain starting inventory
- `Q2`: Deterministic order arriving
- `pmf_D2`: Uncertain demand

Compute `I2_pre = I1_end + Q2` (shift PMF by `Q2`), then:

```python
Q_opt, cost_opt = optimize_order(pmf_I1_end, Q2, pmf_D2, costs, max_Q=1000)
```

Brute-force search:
```
for Q in [0, 1000]:
    cost[Q] = E[0.2 × (I2_pre + Q - D2)^+ + 1.0 × (D2 - I2_pre - Q)^+]
Q* = argmin cost
```

## Metrics

### SIP-Specific Metrics

- `sip_order_qty`: Optimal integer order quantity
- `sip_expected_cost`: Expected cost under optimal policy
- `sip_realized_cost_w2`: Actual cost using true demand (week 2 only)
- `sip_holding_cost_w2`: Holding cost component
- `sip_shortage_cost_w2`: Shortage cost component
- `sip_service_level_w2`: 1 if no stockout, 0 otherwise
- `sip_fill_rate_w2`: Fraction of demand satisfied
- `sip_regret_qty`: `|Q* - y_true|` (oracle comparison)

### Aggregation

Per-SKU (across 8 folds):
- Costs: **SUM** (total cost over all folds)
- Service levels: **MEAN** (average performance)

Overall (across all SKUs):
- Costs: **SUM** (total cost over entire problem)
- Accuracy/density: **MEAN** (average performance)

## Data Requirements

### Initial State (`data/interim/state.parquet`)

Expected schema:
```
Index: (store, product, week)
Columns:
  - on_hand: int (units on shelf)
  - intransit_1: int (arriving at start of week+1)
  - intransit_2: int (arriving at start of week+2)
```

For fold `i`, we look up state at `(store, product, week=i)`.

### Demand Data

- **SIP mode:** Uses `data/processed/demand_imputed_capped.parquet` (outliers capped at 500)
- **Baseline mode:** Uses `data/processed/demand_imputed.parquet` (original)

## Usage Examples

### Run SIP Evaluation (v4)

```bash
# Activate environment
source activate.sh

# Run evaluation with SIP optimization
python -m vn2.cli eval-models \
  --use-sip-optimization \
  --sip-grain 1000 \
  --out-suffix v4 \
  --holdout 8 \
  --n-jobs 12 \
  --batch-size 2000 \
  --resume

# Results saved to:
# - models/results/eval_folds_v4.parquet
# - models/results/eval_agg_v4.parquet
# - models/results/leaderboards_v4.parquet
```

### Aggregate Only

```bash
python -m vn2.cli eval-models \
  --aggregate \
  --out-suffix v4
```

### Compare Models

```bash
# Evaluate specific models only
python -m vn2.cli eval-models \
  --use-sip-optimization \
  --models slurp_stockout_aware slurp_bootstrap seasonal_naive \
  --out-suffix v4_champion
```

## Performance

### Computational Complexity

Per (model, SKU, fold):
- PMF interpolation: O(G) where G = grain (1000)
- Inventory convolution: O(G²) worst case, O(G) typical
- Order optimization: O(G²) for brute-force search

Typical timing:
- Single task: ~0.5-2 seconds (depends on grain)
- Full evaluation (599 SKUs × 8 folds × 3 models): ~2-4 hours on 12 cores

### Parallelization

- Task-level parallelism via `joblib.Parallel`
- BLAS threads set to 1 to avoid oversubscription
- Configurable via `--n-jobs` or `--cpu-fraction`

### Checkpointing

- Progress tracked in `eval_progress_v4.json`
- Batch results saved atomically
- Resume with `--resume` flag

## Validation

### Sanity Checks

1. **Cost multiples:** All costs should be multiples of 0.2 (holding) or 1.0 (shortage)
2. **Monotonicity:** Expected cost should decrease as Q approaches optimal, then increase
3. **Critical fractile:** For uniform demand, Q* ≈ CF × max_demand where CF = p/(h+p) = 0.833

### Test Suite

Run unit tests:
```bash
python test_sip_opt.py
```

Tests cover:
- PMF normalization
- Deterministic cases (zero variance)
- Stockout scenarios
- Asymmetric cost optimization

## Comparison with Baseline

### Baseline (v3)

- Uses base-stock policy with normal approximation
- Zero initial inventory (fair comparison)
- Costs computed over full 2-week horizon
- **Flaw:** Lead time semantics caused all models to have identical costs

### SIP (v4)

- Uses actual initial state from historical data
- Discrete optimization over integer quantities
- PMF-based uncertainty propagation
- Week 2 costs only (decision-focused)
- **Fix:** Proper sequential newsvendor with SIP convolution

## Limitations

1. **Grain constraint:** PMF support limited to [0, 1000]; demands > 1000 truncated
2. **Brute-force search:** O(G²) complexity; could use gradient-free optimization for larger grains
3. **Lead time fixed:** Assumes 2-week lead time; not configurable per SKU
4. **No recourse:** Only week 2 decision evaluated; week 3+ orders not modeled

## Future Enhancements

1. **Adaptive grain:** Use SKU-specific max demand for grain
2. **Sparse PMF:** Use sparse representation for efficiency
3. **Multi-period:** Extend to full 8-week rolling horizon
4. **Policy comparison:** Add (s, S) policy, myopic policy, etc.
5. **Uncertainty quantification:** Bootstrap confidence intervals on cost metrics

## References

- Powell, W. B. (2011). *Approximate Dynamic Programming*. Wiley.
- Porteus, E. L. (2002). *Foundations of Stochastic Inventory Theory*. Stanford University Press.
- Zipkin, P. H. (2000). *Foundations of Inventory Management*. McGraw-Hill.

