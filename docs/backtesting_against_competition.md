# Backtesting Against Competition Results

## Overview

This document describes the methodology for backtesting our forecasting and inventory optimization system against the VN2 competition results. The goal is to:

1. **Retrain models** (if needed) with corrected implementations
2. **Run full simulation** against competition data
3. **Calculate realized costs** using actual demand
4. **Compare to leaderboard** to determine what our competition outcome might have been
5. **Identify gaps** between our performance and top performers

**Status**: ⚠️ **Partial Implementation** — Actual sales files for Weeks 1–8 may have been added; see [Data inventory](DATA_INVENTORY.md) for current file status. When all 8 weeks of Sales CSVs are present, run simulation with `--weeks 1 2 3 4 5 6 7 8` and update this status to full.

---

## Data Requirements

### Available Data

| Data Type | Weeks Available | Location | Status |
|-----------|-----------------|----------|--------|
| Initial State | Week 0 | `data/raw/Week 0 - 2024-04-08 - Initial State.csv` | ✅ Complete |
| Historical Sales | Through Week 0 | `data/raw/Week 0 - 2024-04-08 - Sales.csv` | ✅ Complete |
| Competition actuals | Weeks 1–8 | `data/raw/Week N - YYYY-MM-DD - Sales.csv` (N=1..8) | See [DATA_INVENTORY.md](DATA_INVENTORY.md) |
| State Files | Optional | `data/states/state*.csv` | If available |
| Leaderboards | Per-week | `data/raw/leaderboards/Week*.txt` | Week3, Week4 present; others optional |
| Cumulative / Final | 8-week | `data/raw/leaderboards/cumulative-leaderboard.txt` or `FinalScore.txt` | cumulative-leaderboard present; FinalScore optional |

### Missing Data (when not yet added)

| Data Type | Weeks Missing | Impact |
|-----------|---------------|--------|
| **Actual Sales** | Weeks 1–8 (or 6–8) | Cannot compute realized costs for those weeks |
| **State Files** | Weeks 6–8 | Cannot propagate inventory state accurately |
| **Demand Data** | Weeks 6–8 | Cannot validate forecast accuracy |

**Note**: With only Week 0 and historical sales we can train and run benchmark; with Weeks 1–5 we can backtest through week 5; with all 8 weeks we can compute exact 8-week cost and compare to leaderboard.

### Data preparation: demand_long.parquet

Many scripts expect `data/processed/demand_long.parquet` (long format: Store, Product, week, demand). Build it from Week 0 and any Week 1–8 Sales CSVs using:

```bash
python scripts/build_demand_long.py --raw-dir data/raw --out data/processed/demand_long.parquet
```

See [DATA_INVENTORY.md](DATA_INVENTORY.md) for the full data inventory and leaderboard format (including `cumulative-leaderboard.txt` vs `FinalScore.txt`).

---

## Methodology

### 1. Model Retraining (If Needed)

#### When to Retrain

Models should be retrained if:
- **Code bugs fixed**: Lead time errors, transform selection bugs, etc.
- **New features added**: SURD transforms, stockout awareness, etc.
- **Data corrections**: Imputation improvements, data quality fixes
- **Hyperparameter changes**: Model configuration updates

#### Retraining Process

```bash
# 1. Check current model status
python scripts/check_model_coverage.py \
    --checkpoints-dir models/checkpoints \
    --demand-path data/processed/demand_long.parquet

# 2. Retrain specific models (if needed)
python -m vn2.cli forecast \
    --model slurp_surd_stockout_aware \
    --n-jobs 12 \
    --resume

# 3. Regenerate forecasts for new horizons (e.g., h=3)
python scripts/regenerate_h3_forecasts.py \
    --checkpoints-dir models/checkpoints \
    --output-dir models/checkpoints_h3

# 4. Validate checkpoints
python scripts/validate_checkpoints.py \
    --checkpoints-dir models/checkpoints \
    --expected-folds 12
```

#### No-cheating rules (8-week backtest)

- **Per decision**: At end of week t (order t+1), use only state at that time and forecasts from **fold t** (fold_0 for Order 1, …, fold_5 for Order 6). [full_L3_simulation.py](../scripts/full_L3_simulation.py) uses `fold_idx=0` for Order 1 and `fold_idx=week` for Orders 2–6; missing folds fall back to fold_0.
- **Benchmark**: [run_rolling_benchmark_8week.py](../scripts/run_rolling_benchmark_8week.py) uses only sales through week t for the order placed at end of week t.
- **Top 20%**: Cost threshold from [parse_cumulative_leaderboard](../src/vn2/competition/leaderboard_parser.py) and `top_20_percent_threshold()`; target is 8-week cost ≤ that threshold.

#### Key Considerations

- **No data leakage**: Only use data available at decision time
- **Fold structure**: Maintain 12-fold cross-validation structure
- **Horizon requirements**: Ensure forecasts cover required horizons (h=1, h=2, h=3 for L=3)
- **Transform consistency**: Use corrected SURD transforms if available

### 2. Full Simulation Run

#### Simulation Configuration

The backtest simulation must:
- ✅ Use **correct lead time** (L=3: orders placed at end of week t arrive at start of week t+3)
- ✅ Use **actual demand** for realized costs (when available)
- ✅ Propagate **inventory state** week-over-week
- ✅ Respect **information constraints** (no future data leakage)
- ✅ Use **model selector** trained on historical performance

#### Running the Simulation

```bash
# Full L=3 simulation with corrected transforms
python scripts/full_L3_simulation.py \
    --initial-state data/raw/Week\ 0\ -\ 2024-04-08\ -\ Initial\ State.csv \
    --sales-dir data/raw \
    --checkpoints-dir models/checkpoints \
    --selector-map models/results/selector_map_seq12_v1.parquet \
    --surd-transforms data/processed/surd_transforms.parquet \
    --output-dir reports/backtests/competition_simulation \
    --weeks 1 2 3 4 5 \
    --cu 1.0 \
    --co 0.2
```

When all 8 weeks of Sales files are present, run with `--max-weeks 8` for a full 8-week backtest. Use **correct fold indexing** (no leakage): Order 1 uses fold_0, Order 2 uses fold_1, …, Order 6 uses fold_5. See [scripts/full_L3_simulation.py](../scripts/full_L3_simulation.py). Compare to the **rolling benchmark** (same actuals, no leakage): `python scripts/run_rolling_benchmark_8week.py`. Compare both to **top-20%** threshold: `python scripts/compare_8week_results.py --our-cost <cost> --benchmark-cost <cost>`.

#### Simulation Steps

For each week t ∈ {1, 2, ..., 5} (or 1..8 when data available):

1. **Load state**: Current inventory, in-transit orders
2. **Load forecasts**: h=1, h=2, h=3 quantile forecasts for week t, t+1, t+2
3. **Optimize order**: Use `choose_order_L3()` with PMF-based newsvendor
4. **Simulate week**: Apply actual demand, compute holding/shortage costs
5. **Update state**: 
   - Inventory: `I_{t+1} = max(I_t + Q_arriving - D_t, 0)`
   - Pipeline: Shift orders (Q1 → Q2 → Q3, new order → Q1)
6. **Log metrics**: Orders, costs, forecast errors, inventory levels

#### State Propagation

```
Week 1:
  I1 = I0 + Q_{-1} - D1  (Q_{-1} from initial state)
  Order q1 placed → arrives at start of Week 4

Week 2:
  I2 = I1 + Q_0 - D2  (Q_0 from initial state)
  Order q2 placed → arrives at start of Week 5

Week 3:
  I3 = I2 - D3  (no arrivals)
  Order q3 placed → arrives at start of Week 6

Week 4:
  I4 = I3 + q1 - D4  (q1 arrives)
  Order q4 placed → arrives at start of Week 7

Week 5:
  I5 = I4 + q2 - D5  (q2 arrives)
  Order q5 placed → arrives at start of Week 8
```

### 3. Cost Calculation

#### Cost Components

For each week t:

```python
# Holding cost
holding_cost = co * max(0, I_t + Q_arriving - D_t)

# Shortage cost
shortage_cost = cu * max(0, D_t - (I_t + Q_arriving))

# Total cost
total_cost = holding_cost + shortage_cost
```

Where:
- `co = 0.2` (holding cost per unit)
- `cu = 1.0` (shortage cost per unit)
- `I_t`: Inventory at start of week t
- `Q_arriving`: Order arriving at start of week t
- `D_t`: Actual demand during week t

#### Portfolio Cost

```python
portfolio_cost = sum(total_cost for all SKUs across all weeks)
```

#### Week 1 Handling

Week 1 costs are **uncontrollable** (orders placed before competition start):
- Included in total for leaderboard comparison
- Excluded from decision-affected analysis
- Same for all competitors

### 4. Leaderboard Comparison

#### Leaderboard sources

- **FinalScore.txt** (optional): If exported from the competition site as tab-separated (e.g. name, score, rank), use it for programmatic comparison. Scripts such as `scripts/simulate_L3_costs.py` default to `data/raw/leaderboards/FinalScore.txt`.
- **cumulative-leaderboard.txt**: Web paste of the final leaderboard; contains "Cumulative" section with rank, name, and tab-separated fields (order cost, cumulative cost, etc.). Use `parse_cumulative_leaderboard()` in `src/vn2/competition/leaderboard_parser.py` to extract our cost, benchmark cost, and winner cost. Our entry: "Patrick McDonald"; benchmark: "Benchmark Benchmark".

#### Loading leaderboard data (example with FinalScore)

```python
import pandas as pd
from pathlib import Path

# If FinalScore.txt exists (tab-sep: name, score, rank)
leaderboard = pd.read_csv('data/raw/leaderboards/FinalScore.txt', sep='\t')

# Find our entry
our_entry = leaderboard[leaderboard['name'].str.contains('Patrick McDonald', case=False)]

# Extract costs and ranks
winner_cost = leaderboard['score'].min()
our_cost = our_entry['score'].iloc[0]
our_rank = our_entry['rank'].iloc[0]
```

For `cumulative-leaderboard.txt`, use `parse_cumulative_leaderboard()` in `src/vn2/competition/leaderboard_parser.py` to get our cost, benchmark cost, and winner cost.

#### Comparison Metrics

| Metric | Description | Calculation |
|--------|-------------|-------------|
| **Absolute Cost** | Total portfolio cost | Sum of all SKU-week costs |
| **Rank** | Competition placement | Position in sorted leaderboard |
| **Gap to Winner** | Cost difference | `our_cost - winner_cost` |
| **Gap to Median** | Distance from middle | `our_cost - median_cost` |
| **Percentile** | Relative performance | `rank / total_competitors` |
| **Cost Distribution** | Context for performance | Min, Q25, Median, Q75, Max |

#### Weekly Comparison

Compare our simulated costs to weekly leaderboards:

```python
# Load weekly leaderboards
week1_lb = load_leaderboard('data/raw/leaderboards/Week1.txt')
week2_lb = load_leaderboard('data/raw/leaderboards/Week2.txt')
# ... etc

# Compare cumulative costs
for week in [1, 2, 3, 4, 5]:
    our_cumulative = sum(costs[1:week])
    lb_week = load_leaderboard(f'data/raw/leaderboards/Week{week}.txt')
    our_rank = find_rank(lb_week, our_cumulative)
    print(f"Week {week}: Cost={our_cumulative:.2f}, Rank={our_rank}")
```

---

## Implementation Scripts

### Main Backtest Script

**File**: `scripts/full_L3_simulation.py`

**Features**:
- Full L=3 lead time simulation
- Rolling state propagation
- Actual demand integration
- Cost calculation and logging
- Leaderboard comparison

**Usage**:
```bash
python scripts/full_L3_simulation.py \
    --initial-state data/raw/Week\ 0\ -\ 2024-04-08\ -\ Initial\ State.csv \
    --sales-dir data/raw \
    --checkpoints-dir models/checkpoints \
    --selector-map models/results/selector_map_seq12_v1.parquet \
    --output-dir reports/backtests/competition_simulation \
    --weeks 1 2 3 4 5
```

### Backtest Harness

**File**: `scripts/run_backtest_harness.py`

**Features**:
- YAML-driven configuration
- Flexible strategy overrides
- Comprehensive logging
- Reproducible runs

**Usage**:
```bash
python scripts/run_backtest_harness.py \
    --run-spec configs/backtests/competition_backtest.yaml
```

### Leaderboard Analysis

**File**: `scripts/analyze_leaderboards.py`

**Features**:
- Parse all weekly leaderboards
- Calculate performance metrics
- Compare with our results
- Generate reports

**Usage**:
```bash
python scripts/analyze_leaderboards.py \
    --leaderboards-dir data/raw/leaderboards \
    --our-cost 7787.40 \
    --output-dir reports/leaderboard_analysis
```

---

## Results Interpretation

### What We Can Determine

With Weeks 1-5 data:

1. ✅ **Exact costs for Weeks 1-5**: Full backtest with actual demand
2. ✅ **Weekly performance trends**: How costs evolved week-over-week
3. ✅ **Model effectiveness**: Which models performed best in competition
4. ✅ **Forecast accuracy**: How well our forecasts matched actual demand
5. ✅ **Gap analysis**: Distance to top performers through Week 5

### What We Cannot Determine (Without Weeks 6-8)

1. ❌ **Final competition rank**: Cannot compute exact 8-week cost
2. ❌ **Weeks 6-8 performance**: Missing actual demand data
3. ❌ **Complete gap to winner**: Only have partial cost comparison
4. ❌ **Final model ranking**: Cannot validate selector on full competition

### Estimation Strategies

For Weeks 6-8, we can:

1. **Extrapolate from trends**: Use Weeks 3-5 improvement ratio
2. **Use forecasted demand**: Simulate with predicted demand (less accurate)
3. **Assume constant performance**: Extend Week 5 cost forward
4. **Sensitivity analysis**: Test different scenarios

**Example**:
```python
# Estimate 8-week cost
weeks_1_5_cost = 4562.60  # From simulation
avg_weekly_cost = weeks_1_5_cost / 5
estimated_weeks_6_8 = avg_weekly_cost * 3
estimated_8week_total = weeks_1_5_cost + estimated_weeks_6_8
```

---

## Example Results

### Weeks 1-5 Backtest Results

| Week | Simulated Cost | Actual Cost (L=2) | Difference | Notes |
|------|----------------|-------------------|------------|-------|
| 1-2 | €913.80 | €913.80 | €0.00 | Baseline (uncontrollable) |
| 3 | €1,001.60 | €931.60 | +€70.00 | L=3 correction impact |
| 4 | €1,328.60 | €1,780.40 | -€451.80 | Significant improvement |
| 5 | €1,318.60 | €1,004.20 | +€314.40 | Some degradation |
| **Total** | **€4,562.60** | **€4,630.00** | **-€67.40** | Net improvement |

### Estimated 8-Week Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Weeks 1-5 (actual) | €4,562.60 | From simulation |
| Weeks 6-8 (estimated) | €1,100.13 | Extrapolated |
| **Estimated 8-week total** | **€5,662.73** | vs Actual: €7,787.40 |
| **Gap to winner** | ~€985 | Winner: €4,677.00 |
| **Estimated rank** | ~105-108th | vs Actual: 110th |

**Note**: These estimates assume constant performance. Actual Weeks 6-8 may differ significantly.

### Leaderboard Comparison

| Team | 8-Week Cost | Gap to Us | Notes |
|------|-------------|-----------|-------|
| Winner (Bartosz Szabłowski) | €4,677.00 | -€3,110 | Top performer |
| Median | ~€6,500 | -€1,287 | Middle of pack |
| Our Actual (L=2) | €7,787.40 | - | 110th place |
| Our Corrected (L=3, est.) | €5,662.73 | -€2,125 | Estimated improvement |
| **Remaining gap** | | **~€985** | After corrections |

---

## Limitations and Caveats

### Data Limitations

1. **Missing Weeks 6-8**: Cannot validate final performance
2. **State file accuracy**: May not perfectly match competition state
3. **Demand data quality**: Stockout censoring may affect accuracy
4. **Leaderboard timing**: Weekly leaderboards may have different cutoffs

### Methodology Limitations

1. **Model retraining**: May not exactly match competition training
2. **Selector training**: Uses historical data, may differ from competition
3. **Transform selection**: Corrected SURD transforms may not match competition
4. **Estimation uncertainty**: Weeks 6-8 estimates have high variance

### Comparison Limitations

1. **Different implementations**: Other teams may use different approaches
2. **Timing differences**: Order submission timing may vary
3. **Data preprocessing**: Different imputation/cleaning methods
4. **Model selection**: Different model portfolios and selectors

---

## Next Steps

### Next two weeks (action list)

See **[docs/NEXT_TWO_WEEKS.md](NEXT_TWO_WEEKS.md)** for a concrete Week 1 / Week 2 task list (data, backtest, benchmark comparison, forecast error report, lessons learned).

### When Weeks 6-8 Data Arrives

1. **Complete backtest**: Run full 8-week simulation
2. **Validate estimates**: Compare estimated vs actual Weeks 6-8 costs
3. **Final ranking**: Determine exact competition placement
4. **Gap analysis**: Final comparison to winners

### Ongoing Improvements

1. **Model refinement**: Continue improving forecast accuracy
2. **Selector optimization**: Better model selection strategies
3. **Transform validation**: Verify SURD transform effectiveness
4. **Cost optimization**: Fine-tune inventory policies

### Research Questions

1. **What drove the gap?**: Forecast quality, model selection, or policy?
2. **Which SKUs were problematic?**: Identify high-cost SKUs
3. **Could we have done better?**: Counterfactual analysis
4. **What did winners do differently?**: Comparative analysis (if possible)

### AI agents vs our pipeline

See **[docs/HOW_WE_BEAT_THE_WINNER.md](HOW_WE_BEAT_THE_WINNER.md)** (“AI agents vs our pipeline”): the [notebooks/forecast_agents.ipynb](../notebooks/forecast_agents.ipynb) arena evaluates agents on MASE/RMSE, not inventory cost. For a fair comparison, run agent forecasts through our cost sim and compare 8-week cost.

---

## References

### Related Documentation

- [docs/DATA_INVENTORY.md](DATA_INVENTORY.md) - Data file inventory and demand_long build
- [docs/HOW_WE_BEAT_THE_WINNER.md](HOW_WE_BEAT_THE_WINNER.md) - Journey from €7,787 to €4,564; AI agents vs pipeline
- [docs/NEXT_TWO_WEEKS.md](NEXT_TWO_WEEKS.md) - Action list for next two weeks
- [docs/LESSONS_LEARNED_MEETUP.md](LESSONS_LEARNED_MEETUP.md) - Lessons learned (meetup)
- [docs/backtesting_harness.md](backtesting_harness.md) - Backtest harness usage
- [docs/pipelines/sequential_backtest.md](pipelines/sequential_backtest.md) - Sequential backtest pipeline
- [docs/L3_LEAD_TIME_ANALYSIS.md](L3_LEAD_TIME_ANALYSIS.md) - Lead time error analysis
- `reports/surd_effectiveness_report.md` - SURD transform analysis

### Key Scripts

- `scripts/full_L3_simulation.py` - Full L=3 simulation
- `scripts/run_backtest_harness.py` - Backtest harness
- `scripts/analyze_leaderboards.py` - Leaderboard analysis
- `scripts/compare_surd_vs_non_surd.py` - Model comparison

### Data Files

- `data/raw/leaderboards/` - Competition leaderboards
- `data/raw/Week 0 - 2024-04-08 - Sales.csv` - Historical sales
- `data/states/` - Weekly state files
- `models/results/` - Model evaluation results

---

## Conclusion

Backtesting against competition results provides valuable insights into our system's performance, but is currently limited by missing Weeks 6-8 data. With the available data, we can:

- ✅ Validate our implementation corrections
- ✅ Compare performance to competitors through Week 5
- ✅ Estimate final competition outcome
- ✅ Identify areas for improvement

Once Weeks 6-8 data becomes available, we can complete the full backtest and determine our exact competition placement.

---

**Last Updated**: 2024-12-04  
**Status**: ⚠️ Partial (Missing Weeks 6-8 data)  
**Next Review**: When Weeks 6-8 data becomes available

