# Sequential L=2 Evaluation System - User Guide

## Overview

This system implements exact sequential multi-week inventory planning with L=2 (2-week) lead time. It evaluates forecast models by simulating 12 decision epochs, computing realized costs only for decision-affected periods.

## Key Concepts

### Lead Time (L=2)
- Orders placed at epoch `t` arrive at week `t+2`
- At each decision point, you have:
  - `I0`: Current on-hand inventory
  - `Q1`: Order arriving next week (already placed)
  - `Q2`: Order arriving in 2 weeks (already placed)
  - `q_t`: Order to place NOW (will arrive at t+2)

### Horizon Notation
- **h+1** (or h1): Demand forecast for next week (t+1)
- **h+2** (or h2): Demand forecast for 2 weeks out (t+2)
- Used h1 to estimate inventory state after unknown demand next week
- Use h2 to optimize order quantity for week t+2

### Cost Attribution
- **Week t+1 cost**: Uncontrollable (Q1 was already decided)
- **Week t+2 cost**: Controllable (affected by order `q_t`)
- We score models ONLY on controllable costs (t+2 for each decision at t)

## Components Implemented

### 1. Core Planner (`src/vn2/analyze/sequential_planner.py`)

**PMF Operations:**
- `leftover_from_stock_and_demand(S, D_pmf)`: Compute L = max(S - D, 0)
- `diff_pmf_D_minus_L(D_pmf, L_pmf)`: Compute W = D - L via FFT convolution
- `pmf_quantile(pmf, offset, p)`: Find quantile of discrete distribution

**Order Selection:**
- `choose_order_L2(h1_pmf, h2_pmf, I0, Q1, Q2, costs)`: Choose optimal order using newsvendor fractile
  - Computes p* = cu/(cu+co) = 1.0/(1.0+0.2) = 0.8333
  - Finds fractile of W = D_h2 - (leftover_h1 + Q2)
  - Micro-refines by checking q±2 around fractile

**Sequential Runner:**
- `run_sequential_L2(forecasts_h1, forecasts_h2, actuals, I0, Q1, Q2, costs)`: 
  - Runs full H=12 epoch simulation
  - Returns orders, costs per epoch, total cost, coverage

### 2. Evaluation Module (`src/vn2/analyze/sequential_eval.py`)

**Functions:**
- `run_sequential_evaluation(config)`: Evaluate all models × SKUs in parallel
- `aggregate_model_totals(results_df)`: Compute portfolio-level metrics
- `compute_selector(results_df)`: Per-SKU model selection (min cost)
- `render_leaderboard(...)`: Generate markdown leaderboard

**Outputs:**
- `sequential_results_{tag}.parquet`: Per-SKU, per-model detailed results
- `model_totals_{tag}.parquet`: Portfolio totals per model
- `selector_map_{tag}.parquet`: Best model per SKU
- `leaderboard_{tag}.md`: Ranked leaderboard

### 3. Naive-4wk Baseline (`src/vn2/forecast/models/naive4.py`)

Simple baseline forecaster:
- Point forecast: Mean of last 4 weeks
- Density: Poisson(lambda = point forecast)
- Useful as fallback for missing forecasts and benchmark

### 4. CLI Commands

**Sequential Evaluation:**
```bash
python -m vn2.cli sequential-eval \
  --checkpoints models/checkpoints \
  --demand data/processed/demand_long.parquet \
  --state data/interim/state.parquet \
  --out-dir models/results \
  --run-tag seq12 \
  --n-jobs 12 \
  --cu 1.0 \
  --co 0.2 \
  --sip-grain 500 \
  --holdout 12
```

**Today's Order:**
```bash
python -m vn2.cli today-order \
  --checkpoints models/checkpoints \
  --state data/interim/state.parquet \
  --model slurp_surd_stockout_aware \
  --out data/submissions/orders_today.csv \
  --cu 1.0 \
  --co 0.2 \
  --sip-grain 500
```

**Train Naive-4wk Model:**
First, add to `configs/forecast.yaml`:
```yaml
models:
  naive4:
    enabled: true
```

Then run:
```bash
python -m vn2.cli forecast --config configs/forecast.yaml
```

## Usage Workflow

### Step 1: Train Models (if not already done)
```bash
# Enable models in configs/forecast.yaml
python -m vn2.cli forecast --config configs/forecast.yaml --n-jobs 12
```

### Step 2: Run Sequential Evaluation
```bash
python -m vn2.cli sequential-eval \
  --run-tag seq12_baseline \
  --n-jobs 12
```

This will:
- Load all model checkpoints from `models/checkpoints/`
- For each (model, SKU) pair:
  - Load h+1 and h+2 PMFs for all 12 epochs
  - Run sequential planning over 12 decision epochs
  - Compute realized costs using actual demand
- Aggregate results and rank models
- Output leaderboard and detailed results

### Step 3: Review Results
```bash
# View leaderboard
cat models/results/leaderboard_seq12_baseline.md

# Load detailed results for analysis
python
>>> import pandas as pd
>>> results = pd.read_parquet('models/results/sequential_results_seq12_baseline.parquet')
>>> results.groupby('model_name')['total_cost'].describe()
```

### Step 4: Generate Today's Order
```bash
# Use best model from leaderboard
python -m vn2.cli today-order \
  --model slurp_surd_stockout_aware \
  --out data/submissions/week1_orders.csv
```

## Performance Characteristics

- **PMF Operations**: O(n log n) via FFT (n=500 default)
- **Per-SKU Evaluation**: ~50-100ms per SKU (12 epochs)
- **Full Run**: ~10-15 minutes for 599 SKUs × 16 models on 12 cores
- **Memory**: ~2-4 GB peak (PMFs kept ephemeral, only summaries persisted)

## Cost Function Details

For each decision epoch `t` affecting week `t+2`:

```
Cost_t+2 = co * overage_t+2 + cu * underage_t+2

where:
  S_t+2 = (leftover_t+1 + Q2 + q_t)
  overage_t+2 = max(0, S_t+2 - actual_demand_t+2)
  underage_t+2 = max(0, actual_demand_t+2 - S_t+2)
  
  leftover_t+1 = max(0, (I0 + Q1) - actual_demand_t+1)
```

Total cost = sum over epochs t=0..9 of Cost_t+2 (10 controllable costs)

## Handling Missing Forecasts

If h1 or h2 is missing at epoch t:
- Option 1: Set q_t = 0 (current default)
- Option 2: Use fallback PMF (e.g., Naive-4wk)
- Coverage metric tracks % of epochs with valid forecasts
- Models with poor coverage are NOT rewarded (unlike previous bug)

## Sanity Tests

Run tests to verify correctness:
```bash
pytest test/test_sequential_planner.py -v
```

Tests cover:
- PMF normalization and operations
- Leftover inventory calculations
- Newsvendor fractile optimization
- Sequential state propagation
- Cost calculations

## Configuration Parameters

**Key Parameters:**
- `cu` (shortage cost): Default 1.0 per unit short
- `co` (holding cost): Default 0.2 per unit held
- `sip_grain`: PMF support size (default 500)
- `holdout`: Number of epochs (default 12, matches training)
- `n_jobs`: Parallel workers (default 12)

**Critical Fractile:**
p* = cu/(cu+co) = 1.0/(1.0+0.2) = 0.8333

This means we target ~83% service level (balance shortage vs holding)

## Comparison to Previous w8_eval

**Old Approach (BROKEN):**
- Computed week t+1 cost (uncontrollable)
- All models tied except those missing forecasts
- Missing forecasts made models look BETTER (zero cost)

**New Approach (CORRECT):**
- Computes week t+2 cost (controllable by q_t)
- Models differentiated by decision quality
- Missing forecasts penalized (either q=0 or fallback)
- True sequential state propagation

## Troubleshooting

**Problem:** All models have identical costs
- Check that you're using `sequential-eval`, not `w8-eval`
- Verify checkpoints have both h=1 and h=2 forecasts
- Check logs for forecast loading errors

**Problem:** Costs seem too high/low
- Verify cu and co parameters match expectations
- Check that actuals (demand_long.parquet) cover holdout period
- Review a few SKU traces in diagnostics

**Problem:** Out of memory
- Reduce `n_jobs` (fewer parallel workers)
- Reduce `sip_grain` (smaller PMFs, e.g., 300)
- Run in batches by subsetting SKUs

## Next Steps

1. **Cross-validation**: Run on different holdout windows
2. **Sensitivity analysis**: Vary cu/co ratios
3. **Ensemble**: Use selector map for per-SKU model assignment
4. **Production**: Automate today-order generation weekly
5. **Monitoring**: Track realized costs vs forecasted costs over time

## References

- ChatGPT context document (included in original prompt)
- `configs/forecast.yaml`: Model and data configuration
- `src/vn2/analyze/sip_opt.py`: Original PMF conversion utilities
- `src/vn2/forecast/features.py`: Fold/split logic

