# Order Analysis Utilities

## Overview

The `order_analysis` module provides reusable utilities for analyzing order decisions in the sequential backtest framework. These utilities support:

1. **Order Comparison**: Compare submitted orders vs recommended orders
2. **Expected Cost Analysis**: Calculate expected costs at decision time with confidence intervals
3. **Realized Cost Analysis**: Calculate actual costs after observing demand
4. **Cost Uncertainty Quantification**: Compute 5th/95th percentiles and standard deviations

## Key Concepts

### Expected Cost (Ex-Ante)

The **expected cost** is calculated at decision time, before observing actual demand. It represents the average cost across all possible demand realizations, weighted by their probabilities.

For a given order quantity `q`, starting inventory `I`, and demand PMF `P(D=d)`:

```
Expected Cost = Σ P(D=d) × Cost(d, I+q)
```

where:
```
Cost(d, I+q) = c_over × max(0, I+q-d) + c_under × max(0, d-(I+q))
```

**Confidence Intervals**: The 5th and 95th percentiles provide a 90% confidence interval for the cost, accounting for demand uncertainty.

### Realized Cost (Ex-Post)

The **realized cost** is calculated after observing actual demand. It represents the actual cost incurred for the decision made.

```
Realized Cost = c_over × max(0, I+q-d_actual) + c_under × max(0, d_actual-(I+q))
```

### Cost Comparison

Comparing realized vs expected cost reveals:
- **Within CI**: Realized cost falls within the 90% CI → outcome consistent with forecast
- **Outside CI**: Realized cost outside CI → potential forecast error or rare event
- **Systematic bias**: Consistent over/under-estimation across many SKUs

## Module Structure

### Data Classes

#### `OrderComparison`
Comparison between submitted and recommended orders.

```python
@dataclass
class OrderComparison:
    store: int
    product: int
    submitted_order: int
    recommended_order: int
    difference: int
    selected_model: str
```

#### `CostAnalysis`
Cost analysis for an order decision.

```python
@dataclass
class CostAnalysis:
    store: int
    product: int
    order_qty: int
    expected_cost: float
    expected_cost_5th: float  # 5th percentile
    expected_cost_95th: float  # 95th percentile
    expected_cost_std: float
    realized_cost: Optional[float] = None
    initial_on_hand: int = 0
    initial_intransit: int = 0
    actual_demand: Optional[int] = None
```

### Core Functions

#### `compute_expected_cost_with_ci`

Compute expected cost and confidence interval via PMF.

```python
def compute_expected_cost_with_ci(
    demand_pmf: np.ndarray,
    starting_inventory: int,
    order_qty: int,
    costs: Costs,
    confidence_level: float = 0.90
) -> Tuple[float, float, float, float]:
    """
    Returns:
        (expected_cost, cost_5th, cost_95th, cost_std)
    """
```

**Example**:
```python
from vn2.analyze.order_analysis import compute_expected_cost_with_ci
from vn2.analyze.sequential_planner import Costs

costs = Costs(holding=0.2, shortage=1.0)
exp_cost, c5, c95, std = compute_expected_cost_with_ci(
    demand_pmf=my_pmf,
    starting_inventory=10,
    order_qty=15,
    costs=costs
)

print(f"Expected cost: {exp_cost:.2f}")
print(f"90% CI: [{c5:.2f}, {c95:.2f}]")
```

#### `compute_realized_cost`

Compute realized cost given actual demand.

```python
def compute_realized_cost(
    starting_inventory: int,
    order_qty: int,
    actual_demand: int,
    costs: Costs
) -> float:
    """Returns realized cost."""
```

**Example**:
```python
realized = compute_realized_cost(
    starting_inventory=10,
    order_qty=15,
    actual_demand=22,
    costs=costs
)

print(f"Realized cost: {realized:.2f}")
```

#### `analyze_order_decision`

Comprehensive analysis of a single order decision.

```python
def analyze_order_decision(
    store: int,
    product: int,
    order_qty: int,
    demand_pmf: np.ndarray,
    initial_on_hand: int,
    initial_intransit: int,
    costs: Costs,
    actual_demand: Optional[int] = None
) -> CostAnalysis:
    """
    Analyze a single order decision with expected and realized costs.
    """
```

**Example**:
```python
analysis = analyze_order_decision(
    store=2,
    product=124,
    order_qty=18,
    demand_pmf=my_pmf,
    initial_on_hand=1,
    initial_intransit=0,
    costs=costs,
    actual_demand=20  # Optional
)

print(f"Expected: {analysis.expected_cost:.2f}")
print(f"90% CI: [{analysis.expected_cost_5th:.2f}, {analysis.expected_cost_95th:.2f}]")
if analysis.realized_cost is not None:
    print(f"Realized: {analysis.realized_cost:.2f}")
```

#### `compare_orders`

Compare submitted orders vs selector recommendations.

```python
def compare_orders(
    submitted_path: Path,
    selector_map_path: Path,
    results_path: Path,
    week_idx: int = 0
) -> pd.DataFrame:
    """
    Returns DataFrame with comparison results.
    """
```

## Scripts

### `analyze_submitted_order.py`

Comprehensive analysis script that answers:

1. **By how much did our order change compared to what was submitted?**
   - Compares submitted vs recommended orders
   - Reports agreement statistics (exact, within ±1, within ±5)
   - Identifies largest differences

2. **What was the expected cost when we submitted (with 5th/95th percentiles)?**
   - Calculates expected cost at decision time
   - Provides 90% confidence intervals
   - Accounts for demand uncertainty

3. **What is the realized cost now that we have Week 1 data?**
   - Calculates actual cost after observing demand
   - Compares against expected cost
   - Checks if realized cost falls within confidence interval

**Usage**:
```bash
python scripts/analyze_submitted_order.py
```

**Output**:
- Console summary with statistics
- `models/results/order_comparison_detailed.csv` with per-SKU details

## Use Cases

### 1. Post-Decision Analysis

After submitting orders, analyze the quality of the decision:

```python
from vn2.analyze.order_analysis import analyze_order_decision
from vn2.analyze.sequential_planner import Costs

# At decision time
analysis = analyze_order_decision(
    store=2, product=124, order_qty=18,
    demand_pmf=forecast_pmf,
    initial_on_hand=1, initial_intransit=0,
    costs=Costs(holding=0.2, shortage=1.0)
)

print(f"Expected cost: {analysis.expected_cost:.2f}")
print(f"90% CI: [{analysis.expected_cost_5th:.2f}, {analysis.expected_cost_95th:.2f}]")
```

### 2. Forecast Validation

After observing actual demand, validate forecast quality:

```python
# After observing demand
analysis_with_actual = analyze_order_decision(
    store=2, product=124, order_qty=18,
    demand_pmf=forecast_pmf,
    initial_on_hand=1, initial_intransit=0,
    costs=Costs(holding=0.2, shortage=1.0),
    actual_demand=20  # Observed
)

print(f"Realized cost: {analysis_with_actual.realized_cost:.2f}")
within_ci = (analysis_with_actual.expected_cost_5th <= 
             analysis_with_actual.realized_cost <= 
             analysis_with_actual.expected_cost_95th)
print(f"Within 90% CI: {within_ci}")
```

### 3. Model Comparison

Compare expected costs across different models:

```python
models = ['zinb', 'slurp_bootstrap', 'ets']
for model_name in models:
    pmf = load_forecast_pmf(store, product, model_name)
    analysis = analyze_order_decision(
        store, product, order_qty,
        pmf, on_hand, intransit, costs
    )
    print(f"{model_name}: {analysis.expected_cost:.2f} "
          f"[{analysis.expected_cost_5th:.2f}, {analysis.expected_cost_95th:.2f}]")
```

### 4. Sensitivity Analysis

Analyze how order quantity affects expected cost:

```python
for q in range(0, 30, 5):
    analysis = analyze_order_decision(
        store, product, q,
        demand_pmf, on_hand, intransit, costs
    )
    print(f"q={q}: {analysis.expected_cost:.2f}")
```

## Integration with Sequential Backtest

The order analysis utilities integrate seamlessly with the sequential backtest:

1. **During backtest**: Use `choose_order_L2` to select optimal orders
2. **After backtest**: Use `analyze_order_decision` to analyze each decision
3. **Validation**: Compare expected vs realized costs to validate forecasts

## Mathematical Details

### Cost Function

For inventory `I`, order `q`, and demand `D`:

```
Cost(I, q, D) = h × max(I+q-D, 0) + p × max(D-(I+q), 0)
```

where:
- `h` = holding cost per unit (default: 0.2)
- `p` = shortage cost per unit (default: 1.0)

### Expected Cost

```
E[Cost] = Σ_d P(D=d) × Cost(I, q, d)
```

### Cost Percentiles

The α-percentile of cost is:

```
Cost_α = min{c : P(Cost ≤ c) ≥ α}
```

Computed by:
1. Calculate cost for each demand realization
2. Sort costs and accumulate probabilities
3. Find cost value at cumulative probability α

### Confidence Interval

For confidence level γ (default 0.90):

```
α_lower = (1 - γ) / 2 = 0.05
α_upper = 1 - (1 - γ) / 2 = 0.95
CI = [Cost_0.05, Cost_0.95]
```

## Best Practices

1. **Always compute confidence intervals**: Provides uncertainty quantification
2. **Compare realized vs expected**: Validates forecast quality
3. **Aggregate across SKUs**: Portfolio-level metrics more stable than individual SKU
4. **Track coverage**: % of realized costs within confidence intervals
5. **Use consistent costs**: Same `c_under` and `c_over` for fair comparison

## Performance Notes

- **PMF size**: Default 500 is sufficient for most cases
- **Computation time**: ~0.1ms per SKU for expected cost
- **Memory**: Minimal (PMFs are small arrays)
- **Parallelization**: Can analyze SKUs in parallel (independent)

## References

- `src/vn2/analyze/order_analysis.py`: Core module
- `scripts/analyze_submitted_order.py`: Analysis script
- `docs/pipelines/sequential_backtest.md`: Sequential backtest documentation
- `docs/modeling/newsvendor.md`: Newsvendor optimization

