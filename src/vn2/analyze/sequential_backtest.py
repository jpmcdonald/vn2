"""
12-week progressive backtest with L=2 lead time and PMF-based optimization.

Implements a realistic backtest where:
- Order placed at start of week t arrives at start of week t+2
- State is updated with actual demand as it becomes available
- Expected costs use PMFs; realized costs use actual demand
- Both include-week1 and exclude-week1 totals are computed
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from .sequential_planner import (
    Costs,
    _safe_pmf,
    choose_order_L2,
    leftover_from_stock_and_demand,
    _shift_right,
    diff_pmf_D_minus_L,
    expected_pos_neg_from_Z,
    _conv_fft
)


@dataclass
class BacktestState:
    """Inventory state at a point in time."""
    week: int  # Current week number (1-indexed)
    on_hand: int  # On-hand inventory at start of week (before arrivals)
    intransit_1: int  # Arriving at start of current week (placed at week-2)
    intransit_2: int  # Arriving at start of next week (placed at week-1)
    
    def copy(self) -> 'BacktestState':
        return BacktestState(
            week=self.week,
            on_hand=self.on_hand,
            intransit_1=self.intransit_1,
            intransit_2=self.intransit_2
        )


@dataclass
class WeekResult:
    """Result for a single week."""
    week: int
    order_placed: int  # Order placed at start of week (arrives week+2)
    demand_actual: Optional[int]  # Actual demand (None if not yet observed)
    expected_cost: float  # Expected cost at decision time
    realized_cost: Optional[float]  # Realized cost (None if demand not observed)
    state_before: BacktestState  # State before week starts
    state_after: Optional[BacktestState]  # State after week ends (None if demand not observed)
    pmf_residual: float  # 1 - sum(PMF) for normalization check


@dataclass
class BacktestResult:
    """Complete backtest result for one (model, SKU)."""
    store: int
    product: int
    model_name: str
    weeks: List[WeekResult]
    total_expected_cost: float
    total_realized_cost: float
    total_expected_cost_excl_w1: float
    total_realized_cost_excl_w1: float
    n_weeks: int
    n_missing_forecasts: int
    diagnostics: Dict = field(default_factory=dict)


def reconstruct_initial_state(
    store: int,
    product: int,
    initial_state_df: pd.DataFrame,
    sales_history: pd.DataFrame,
    backtest_start_week: int = 1
) -> BacktestState:
    """
    Reconstruct initial inventory state for backtest start.
    
    Args:
        store: Store ID
        product: Product ID
        initial_state_df: DataFrame with columns [Store, Product, End Inventory, In Transit W+1, In Transit W+2]
        sales_history: DataFrame with weekly sales columns
        backtest_start_week: Week number to start backtest (1-indexed)
    
    Returns:
        BacktestState for start of backtest_start_week
    """
    # Get initial state from file (this is state at end of week 0)
    mask = (initial_state_df['Store'] == store) & (initial_state_df['Product'] == product)
    if not mask.any():
        raise ValueError(f"SKU ({store}, {product}) not found in initial state")
    
    row = initial_state_df[mask].iloc[0]
    
    # At start of week 1:
    # - on_hand = End Inventory from week 0
    # - intransit_1 = In Transit W+1 (arrives at start of week 1, placed at week -1)
    # - intransit_2 = In Transit W+2 (arrives at start of week 2, placed at week 0)
    
    if backtest_start_week == 1:
        return BacktestState(
            week=1,
            on_hand=int(row['End Inventory']),
            intransit_1=int(row['In Transit W+1']),
            intransit_2=int(row['In Transit W+2'])
        )
    else:
        # For later weeks, would need to simulate forward from week 1
        # Not implemented yet
        raise NotImplementedError(f"Backtest start week {backtest_start_week} > 1 not yet supported")


def load_actual_demand(
    store: int,
    product: int,
    sales_df: pd.DataFrame,
    n_weeks: int = 12
) -> List[int]:
    """
    Load actual demand for a SKU from sales history.
    
    Args:
        store: Store ID
        product: Product ID
        sales_df: Sales DataFrame with weekly columns
        n_weeks: Number of weeks to load
    
    Returns:
        List of integer demands (length n_weeks)
    """
    mask = (sales_df['Store'] == store) & (sales_df['Product'] == product)
    if not mask.any():
        raise ValueError(f"SKU ({store}, {product}) not found in sales data")
    
    row = sales_df[mask].iloc[0]
    
    # Get last n_weeks of sales (most recent columns)
    date_cols = [c for c in sales_df.columns if c not in ['Store', 'Product']]
    if len(date_cols) < n_weeks:
        raise ValueError(f"Not enough history: {len(date_cols)} < {n_weeks}")
    
    recent_cols = date_cols[-n_weeks:]
    demands = [int(row[col]) for col in recent_cols]
    
    return demands


def run_12week_backtest(
    store: int,
    product: int,
    model_name: str,
    forecasts_h1: List[Optional[np.ndarray]],  # PMF for each week's demand (week t)
    forecasts_h2: List[Optional[np.ndarray]],  # PMF for each week's next-week demand (week t+1)
    actuals: List[int],  # Actual demand for weeks 1..12
    initial_state: BacktestState,
    costs: Costs,
    pmf_grain: int = 500
) -> BacktestResult:
    """
    Run 12-week progressive backtest with L=2 lead time.
    
    Timeline:
    - Week 1-10: Place orders (arrive at week 3-12)
    - Week 11-12: No new orders; just compute costs from pending orders
    - Week 1: Cost is uncontrollable (everyone has same state)
    - Week 2-12: Costs affected by our decisions
    
    Args:
        store: Store ID
        product: Product ID
        model_name: Model name
        forecasts_h1: List of 12 PMFs for week t demand (or None if missing)
        forecasts_h2: List of 12 PMFs for week t+1 demand (or None if missing)
        actuals: List of 12 actual demands
        initial_state: Initial inventory state at start of week 1
        costs: Cost parameters
        pmf_grain: PMF support size (for validation)
    
    Returns:
        BacktestResult with per-week orders, costs, and totals
    """
    assert len(forecasts_h1) == 12, "Need 12 h1 forecasts"
    assert len(forecasts_h2) == 12, "Need 12 h2 forecasts"
    assert len(actuals) == 12, "Need 12 weeks of actual demand"
    
    state = initial_state.copy()
    weeks = []
    n_missing = 0
    
    for t in range(1, 13):  # Weeks 1-12
        week_idx = t - 1
        h1 = forecasts_h1[week_idx]
        h2 = forecasts_h2[week_idx]
        demand_actual = actuals[week_idx]
        
        state_before = state.copy()
        
        # Check for missing forecasts
        if h1 is None or h2 is None:
            # Missing forecast: place order of 0
            q_t = 0
            expected_cost = 0.0
            pmf_residual = 0.0
            n_missing += 1
        elif t >= 11:
            # Weeks 11-12: No new orders (would arrive after horizon)
            q_t = 0
            expected_cost = 0.0
            pmf_residual = 1.0 - (h1.sum() + h2.sum()) / 2.0
        else:
            # Weeks 1-10: Optimize order
            # Normalize and check PMFs
            h1 = _safe_pmf(h1)
            h2 = _safe_pmf(h2)
            pmf_residual = max(1.0 - h1.sum(), 1.0 - h2.sum())
            
            # Choose order using current state
            # I0 = on_hand, Q1 = intransit_1, Q2 = intransit_2
            q_t, expected_cost = choose_order_L2(
                h1, h2,
                state.on_hand,
                state.intransit_1,
                state.intransit_2,
                costs,
                micro_refine=True
            )
        
        # Compute realized cost for current week
        # Week t cost depends on:
        # - Starting inventory: on_hand + intransit_1
        # - Demand: demand_actual
        S_t = state.on_hand + state.intransit_1
        leftover_t = max(0, S_t - demand_actual)
        shortage_t = max(0, demand_actual - S_t)
        realized_cost = costs.holding * leftover_t + costs.shortage * shortage_t
        
        # Update state for next week
        state_after = BacktestState(
            week=t + 1,
            on_hand=leftover_t,
            intransit_1=state.intransit_2,
            intransit_2=q_t
        )
        
        weeks.append(WeekResult(
            week=t,
            order_placed=q_t,
            demand_actual=demand_actual,
            expected_cost=expected_cost,
            realized_cost=realized_cost,
            state_before=state_before,
            state_after=state_after,
            pmf_residual=pmf_residual
        ))
        
        state = state_after
    
    # Aggregate costs
    total_expected = sum(w.expected_cost for w in weeks)
    total_realized = sum(w.realized_cost for w in weeks)
    total_expected_excl_w1 = sum(w.expected_cost for w in weeks[1:])
    total_realized_excl_w1 = sum(w.realized_cost for w in weeks[1:])
    
    return BacktestResult(
        store=store,
        product=product,
        model_name=model_name,
        weeks=weeks,
        total_expected_cost=total_expected,
        total_realized_cost=total_realized,
        total_expected_cost_excl_w1=total_expected_excl_w1,
        total_realized_cost_excl_w1=total_realized_excl_w1,
        n_weeks=12,
        n_missing_forecasts=n_missing,
        diagnostics={
            'initial_state': initial_state,
            'final_state': state,
            'max_pmf_residual': max(w.pmf_residual for w in weeks)
        }
    )


def quantiles_to_pmf(
    quantiles: np.ndarray,
    quantile_levels: np.ndarray,
    grain: int = 500
) -> np.ndarray:
    """
    Convert quantile forecast to discrete PMF via interpolation.
    
    Args:
        quantiles: Array of quantile values (e.g., 13 quantiles from model)
        quantile_levels: Array of quantile levels (e.g., [0.01, 0.05, ..., 0.99])
        grain: Maximum support for PMF (0 to grain-1 inclusive)
    
    Returns:
        pmf: Array of length grain with probabilities
    """
    # Ensure quantiles are non-negative and sorted
    quantiles = np.maximum(quantiles, 0)
    quantiles = np.sort(quantiles)
    
    # Create CDF by interpolating quantile function
    # Add boundary points for extrapolation
    q_extended = np.concatenate([[0], quantile_levels, [1]])
    v_extended = np.concatenate([[0], quantiles, [quantiles[-1]]])
    
    # Interpolate CDF at integer points
    support = np.arange(grain)
    cdf = np.interp(support, v_extended, q_extended)
    
    # Convert CDF to PMF via differencing
    pmf = np.diff(cdf, prepend=0)
    
    # Ensure valid PMF (non-negative, sums to ~1)
    pmf = np.maximum(pmf, 0)
    pmf_sum = pmf.sum()
    if pmf_sum > 0:
        pmf = pmf / pmf_sum
    else:
        # Degenerate case: put all mass at zero
        pmf = np.zeros(grain)
        pmf[0] = 1.0
    
    return pmf

