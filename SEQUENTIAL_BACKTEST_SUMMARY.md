# Sequential 12-Week Backtest Implementation Summary

## Overview

Implemented a robust 12-week progressive backtest with fixed lead time L=2 (order placed at start of week t arrives at start of week t+2). The system evaluates forecast models using exact newsvendor optimization with PMF-based uncertainty propagation.

## Branch

`sequential-backtest-l2`

## Files Created

### Core Modules

1. **`src/vn2/analyze/sequential_backtest.py`** (280 lines)
   - `BacktestState`: Inventory state dataclass
   - `WeekResult`: Per-week result dataclass
   - `BacktestResult`: Complete backtest result dataclass
   - `run_12week_backtest()`: Main 12-week backtest engine
   - `reconstruct_initial_state()`: Initial state from historical data
   - `load_actual_demand()`: Load actual demand from sales CSV
   - `quantiles_to_pmf()`: Convert quantiles to discrete PMF

2. **`src/vn2/analyze/forecast_loader.py`** (120 lines)
   - `load_forecasts_for_sku()`: Load h1/h2 PMFs from checkpoints
   - `get_available_models()`: Discover available models
   - `get_available_skus()`: Discover available SKUs

3. **`src/vn2/analyze/backtest_runner.py`** (260 lines)
   - `run_single_backtest()`: Single (model, SKU) backtest
   - `run_all_backtests()`: Parallel evaluation across all combinations
   - `compute_selector()`: Per-SKU model selection by cost
   - `generate_leaderboard()`: Model ranking by portfolio cost

### Files Modified

4. **`src/vn2/analyze/sequential_planner.py`**
   - Fixed L=2 lead time semantics in docstrings
   - Clarified order arrival timing (t → t+2)
   - No logic changes (already correct)

5. **`src/vn2/analyze/sequential_eval.py`**
   - Updated to use new `run_12week_backtest()` function
   - Added support for both expected and realized costs
   - Integrated with new forecast loader

### Tests

6. **`test/test_sequential_backtest.py`** (357 lines, 17 tests)
   - `TestPMFOperations`: PMF normalization, shift, FFT convolution
   - `TestInventoryPropagation`: Leftover calculations, stockouts
   - `TestNewsvendorOptimization`: Order selection, fractile, initial inventory
   - `TestBacktestEngine`: Full backtest, missing forecasts, cost components
   - `TestQuantileToPMF`: Quantile conversion
   - `TestLeadTimeSemantics`: L=2 arrival timing

### Documentation

7. **`docs/pipelines/sequential_backtest.md`** (450 lines)
   - Complete usage guide
   - Algorithm description
   - Output schema
   - Validation and troubleshooting
   - CLI examples

## Key Features

### 1. Lead Time Semantics (L=2)

- Order placed at start of week t arrives at start of week t+2
- Maintains two in-transit buckets: `intransit_1` (arrives this week), `intransit_2` (arrives next week)
- Weeks 11-12 place no orders (would arrive after 12-week horizon)

### 2. Progressive Backtest

- State updated with actual demand as it becomes available
- Reduces uncertainty week-by-week
- Realistic simulation of decision-making under uncertainty

### 3. Cost Tracking

- **Expected cost**: Computed at decision time using PMFs
- **Realized cost**: Computed ex-post using actual demand
- **Include-week1**: Total cost over all 12 weeks
- **Exclude-week1**: Cost over weeks 2-12 (decision-affected only)

### 4. Newsvendor Optimization

- Critical fractile: p* = cu/(cu+co) = 1.0/(1.0+0.2) = 0.8333
- FFT-based exact PMF convolutions (O(n log n))
- Micro-refinement: check q* ± {1, 2} around fractile solution
- Integer nonnegative orders

### 5. Model Selection

- Per-SKU selector: picks model with lowest realized cost
- Portfolio totals: sum across all SKUs
- Leaderboard: models ranked by portfolio cost

## Usage

### Run Evaluation

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

### Generate Today's Order

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

## Test Results

All 17 unit tests pass:

```
test_safe_pmf_normalization ............................ PASSED
test_safe_pmf_with_negatives ........................... PASSED
test_shift_right ....................................... PASSED
test_conv_fft_matches_direct ........................... PASSED
test_leftover_deterministic ............................ PASSED
test_leftover_stockout ................................. PASSED
test_leftover_uniform_demand ........................... PASSED
test_choose_order_deterministic ........................ PASSED
test_choose_order_newsvendor_fractile .................. PASSED
test_choose_order_with_initial_inventory ............... PASSED
test_backtest_deterministic ............................ PASSED
test_backtest_with_missing_forecasts ................... PASSED
test_backtest_weeks_11_12_no_orders .................... PASSED
test_backtest_cost_components .......................... PASSED
test_quantiles_to_pmf_basic ............................ PASSED
test_quantiles_to_pmf_degenerate ....................... PASSED
test_order_arrives_at_t_plus_2 ......................... PASSED
```

## Output Files

### Sequential Results

Per-(model, SKU) results with columns:
- `store`, `product`, `model_name`
- `total_cost`, `total_cost_excl_w1` (realized)
- `total_expected_cost`, `total_expected_cost_excl_w1`
- `n_missing`, `n_weeks`
- `orders` (list), `costs_by_week` (list)
- `max_pmf_residual`

### Model Totals

Aggregated by model:
- `portfolio_cost`, `mean_sku_cost`, `std_sku_cost`
- `n_skus`, `p05_sku`, `p50_sku`, `p95_sku`
- `avg_coverage`, `total_missing`

### Selector Map

Per-SKU best model:
- `store`, `product`, `model_name`, `total_cost`

### Leaderboard

Markdown table with model rankings

## Data Requirements

### Forecast Checkpoints

```
models/checkpoints/
  {model_name}/
    {store}_{product}/
      fold_0.pkl  # Week 1 forecast
      fold_1.pkl  # Week 2 forecast
      ...
      fold_11.pkl # Week 12 forecast
```

Each `.pkl` contains:
```python
{
    'quantiles': pd.DataFrame,  # shape (2, 13): [step1, step2] × 13 quantiles
    'model': ...,
    'metrics': {...},
    'task': {...}
}
```

### Initial State

CSV with columns:
- `Store`, `Product`
- `End Inventory` (on-hand at end of week 0)
- `In Transit W+1` (arrives at start of week 1)
- `In Transit W+2` (arrives at start of week 2)

### Sales History

CSV with weekly sales columns (date format: YYYY-MM-DD)

## Algorithm

### Per-Week Loop (t = 1..10)

1. Load h1_t (week t demand), h2_t (week t+1 demand)
2. Convert quantiles → PMFs (n=500)
3. Propagate uncertainty:
   - L1 = max((I0 + Q1) - D_t, 0) using h1_t
   - Lpre = L1 + Q2
4. Optimize order:
   - W = D_{t+1} - Lpre
   - q* = Q_W(0.8333) with micro-refine
5. Compute expected cost at week t+1
6. Update state with actual D_t:
   - Realize cost
   - I_{t+1} = max((I_t + Q1) - D_t, 0)
   - Q1 ← Q2, Q2 ← q*

### Weeks 11-12

- No new orders (q = 0)
- Compute costs from pending orders only

## Validation

### Sanity Checks

- PMF normalization: sum ≈ 1.0
- Cost finiteness: all costs ≥ 0 and finite
- Order non-negativity: all q ≥ 0
- State consistency: inventory balance
- Week 11-12 orders: q = 0

### PMF Residuals

- Max residual ε = 1 - sum(PMF)
- Typical: ε < 0.001
- Warning if ε > 0.01

## Performance

- **Single SKU**: ~0.1-0.5 seconds
- **599 SKUs × 3 models**: ~10-30 minutes (12 cores)
- **FFT convolution**: O(n log n) where n=500

## Next Steps

1. Run full evaluation on all available models
2. Compare with existing SIP evaluation results
3. Analyze selector performance vs single-model portfolios
4. Investigate high-cost outliers
5. Consider adaptive PMF grain per SKU

## Notes

- Week 1 cost is uncontrollable (same initial state for all models)
- Both include-week1 and exclude-week1 totals are tracked
- Selector uses realized cost (not expected) for model selection
- Missing forecasts default to q=0 (conservative)

## Contact

For questions or issues, see the main README or open a GitHub issue.

