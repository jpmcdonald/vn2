# Sequential 12-Week Backtest with L=2 Lead Time

## Overview

This pipeline runs a progressive 12-week backtest with fixed lead time L=2 (order placed at start of week t arrives at start of week t+2). It evaluates forecast models by computing optimal integer orders using newsvendor optimization with PMF-based uncertainty propagation.

## Key Features

- **Fixed L=2 lead time**: Order at week t → arrives at week t+2
- **PMF-based optimization**: Exact discrete newsvendor with FFT convolutions
- **Progressive backtest**: State updated with actual demand as it becomes available
- **Expected + realized costs**: Both decision-time expected and ex-post realized costs
- **Per-SKU model selection**: Selector picks best model per series by realized cost

## Cost Semantics

- **Shortage cost (cu)**: 1.0 per unit short (lost gross profit)
- **Holding cost (co)**: 0.2 per unit held (inventory carrying cost)
- **Newsvendor fractile**: p* = cu/(cu+co) = 1.0/(1.0+0.2) = 0.8333
- **Week 1 cost**: Included in totals (uncontrollable, same for all models)
- **Weeks 2-12**: Decision-affected costs

## Timeline

```
Week  | Order Decision | Order Arrives | Cost Charged
------|----------------|---------------|-------------
1     | q1 (→ week 3)  | Q-1 (from t=-1)| Yes (uncontrollable)
2     | q2 (→ week 4)  | q0 (from t=0)  | Yes
3     | q3 (→ week 5)  | q1 (from t=1)  | Yes
...   | ...            | ...            | ...
10    | q10 (→ week 12)| q8 (from t=8)  | Yes
11    | (none)         | q9 (from t=9)  | Yes
12    | (none)         | q10 (from t=10)| Yes
```

## Usage

### 1. Run Sequential Evaluation

Evaluate all models and SKUs:

```bash
python -m vn2.cli sequential-eval \
  --checkpoints models/checkpoints \
  --demand data/processed/demand_long.parquet \
  --state data/interim/state.parquet \
  --out-dir models/results \
  --run-tag seq12_v1 \
  --n-jobs 12 \
  --cu 1.0 \
  --co 0.2 \
  --sip-grain 500 \
  --holdout 12
```

**Outputs:**
- `sequential_results_seq12_v1.parquet`: Per-(model, SKU) results
- `model_totals_seq12_v1.parquet`: Aggregated model totals
- `selector_map_seq12_v1.parquet`: Per-SKU model selector
- `leaderboard_seq12_v1.md`: Markdown leaderboard

### 2. Generate Today's Order

Use best model to generate current week orders:

```bash
python -m vn2.cli today-order \
  --checkpoints models/checkpoints \
  --state data/interim/state.parquet \
  --model zinb \
  --out submissions/orders_today.csv \
  --cu 1.0 \
  --co 0.2 \
  --sip-grain 500
```

## Configuration

### Cost Parameters

- `--cu`: Shortage cost per unit (default: 1.0)
- `--co`: Holding cost per unit (default: 0.2)

### PMF Settings

- `--sip-grain`: PMF support size (default: 500)
  - Larger = more precision, slower
  - Should cover max demand + safety margin

### Parallelization

- `--n-jobs`: Number of parallel workers (default: 12)
  - Set to number of CPU cores for best performance

## Data Requirements

### Forecast Checkpoints

Structure:
```
models/checkpoints/
  {model_name}/
    {store}_{product}/
      fold_0.pkl
      fold_1.pkl
      ...
      fold_11.pkl
```

Each checkpoint (`.pkl`) contains:
```python
{
    'quantiles': pd.DataFrame,  # shape (2, 13): steps × quantiles
    'model': ...,               # fitted model object
    'metrics': {...},           # evaluation metrics
    'task': {...}               # task metadata
}
```

Quantile columns: `[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]`

### State Data

Required columns:
- `Store`, `Product`: SKU identifiers
- `on_hand`: On-hand inventory at start of week 1
- `intransit_1`: Order arriving at start of week 1 (placed at t=-1)
- `intransit_2`: Order arriving at start of week 2 (placed at t=0)

### Demand Data

Long-format with columns:
- `Store`, `Product`: SKU identifiers
- `Week`: Week number (1-indexed)
- `Demand`: Actual demand (integer)

## Algorithm

### Per-Week Decision Loop

For each week t ∈ {1, ..., 10}:

1. **Load forecasts**: h1_t (week t demand), h2_t (week t+1 demand)
2. **Convert to PMFs**: Quantiles → discrete PMF via interpolation
3. **Propagate uncertainty**:
   - L1 = max((I0 + Q1) - D_t, 0) using h1_t
   - Lpre = L1 + Q2
4. **Optimize order**:
   - W = D_{t+1} - Lpre
   - q* = Q_W(p*) where p* = 0.8333
   - Micro-refine: check q* ± {1, 2}
5. **Compute expected cost**: E[co·overage + cu·shortage] at week t+1
6. **Update state with actual demand**:
   - Realize cost using actual D_t
   - Update inventory: I_{t+1} = max((I_t + Q1) - D_t, 0)
   - Shift pipeline: Q1 ← Q2, Q2 ← q*

For weeks t ∈ {11, 12}:
- No new orders (would arrive after horizon)
- Compute costs from pending orders only

### PMF Operations

All operations use FFT-based exact convolutions:
- **Leftover**: L = max(S - D, 0)
- **Shift**: Add constant to support
- **Difference**: W = D - L via reversed convolution
- **Quantile**: CDF inversion

### Newsvendor Optimization

1. Compute W = D - L distribution
2. Find q0 = Q_W(p*) via CDF
3. Evaluate cost at {q0-2, q0-1, q0, q0+1, q0+2}
4. Return argmin

## Output Schema

### Sequential Results

Per-(model, SKU) results:

| Column | Type | Description |
|--------|------|-------------|
| `store` | int | Store ID |
| `product` | int | Product ID |
| `model_name` | str | Model name |
| `total_cost` | float | Total realized cost (weeks 1-12) |
| `total_cost_excl_w1` | float | Realized cost (weeks 2-12) |
| `total_expected_cost` | float | Total expected cost at decision time |
| `total_expected_cost_excl_w1` | float | Expected cost (weeks 2-12) |
| `n_missing` | int | Number of missing forecasts |
| `n_weeks` | int | Number of weeks evaluated |
| `orders` | list | Per-week orders [q1, ..., q12] |
| `costs_by_week` | list | Per-week realized costs |
| `max_pmf_residual` | float | Max PMF normalization residual |

### Model Totals

Aggregated by model:

| Column | Description |
|--------|-------------|
| `model_name` | Model name |
| `portfolio_cost` | Sum of costs across all SKUs |
| `mean_sku_cost` | Mean cost per SKU |
| `std_sku_cost` | Std dev of SKU costs |
| `n_skus` | Number of SKUs evaluated |
| `p05_sku`, `p50_sku`, `p95_sku` | Cost quantiles |
| `avg_coverage` | Mean forecast coverage |
| `total_missing` | Total missing forecasts |

### Selector Map

Per-SKU best model:

| Column | Description |
|--------|-------------|
| `store` | Store ID |
| `product` | Product ID |
| `model_name` | Selected model |
| `total_cost` | Best cost for this SKU |

## Validation

### Sanity Checks

1. **PMF normalization**: All PMFs sum to 1.0 ± 1e-6
2. **Cost finiteness**: All costs are finite and non-negative
3. **Order non-negativity**: All orders ≥ 0
4. **State consistency**: Inventory balance holds each week
5. **Week 11-12 orders**: Should be 0 (no arrivals within horizon)

### Unit Tests

Run tests:
```bash
pytest test/test_sequential_backtest.py -v
```

Tests cover:
- PMF operations (normalization, shift, convolution)
- Inventory propagation (leftover, stockout)
- Newsvendor optimization (fractile, micro-refine)
- Full backtest (deterministic, missing forecasts, L=2 semantics)

## Troubleshooting

### Missing Forecasts

**Symptom**: High `n_missing` counts

**Solutions**:
- Check checkpoint files exist for all folds
- Verify quantiles DataFrame has 2 rows (steps 1 and 2)
- Use fallback PMF (uniform or historical) for missing

### PMF Residuals

**Symptom**: `max_pmf_residual` > 0.01

**Causes**:
- Demand exceeds PMF grain (500)
- Quantile extrapolation issues

**Solutions**:
- Increase `--sip-grain` to 1000
- Check max demand in data
- Verify quantile levels cover [0.01, 0.99]

### High Costs

**Symptom**: Unexpectedly high realized costs

**Checks**:
- Compare expected vs realized costs (forecast quality)
- Check initial state (negative inventory?)
- Verify lead time semantics (orders arriving correctly)
- Inspect per-week costs for anomalies

### Slow Performance

**Symptom**: Evaluation takes > 1 hour

**Solutions**:
- Increase `--n-jobs` (up to CPU count)
- Reduce `--sip-grain` to 300 (faster, less precise)
- Filter to subset of models or SKUs for testing

## References

- **Newsvendor model**: Classic single-period inventory optimization
- **SIP (Stochastic Information Packets)**: PMF-based uncertainty representation
- **FFT convolution**: O(n log n) exact discrete convolution
- **Progressive backtest**: Walk-forward evaluation with state updates

## Contact

For issues or questions, see `backlog.md` or open a GitHub issue.

