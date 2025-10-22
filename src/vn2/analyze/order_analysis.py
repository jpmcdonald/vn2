"""
Order analysis utilities for sequential backtest.

This module provides functions to:
1. Compare submitted orders vs recommended orders
2. Calculate expected costs at decision time (with uncertainty)
3. Calculate realized costs after observing actual demand
4. Compute confidence intervals on costs

These utilities support post-hoc analysis of order decisions.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from pathlib import Path

from .sequential_planner import Costs, _safe_pmf, _shift_right, _conv_fft, expected_pos_neg_from_Z
from .sequential_backtest import quantiles_to_pmf


@dataclass
class OrderComparison:
    """Comparison between submitted and recommended orders."""
    store: int
    product: int
    submitted_order: int
    recommended_order: int
    difference: int
    selected_model: str


@dataclass
class CostAnalysis:
    """Cost analysis for an order decision."""
    store: int
    product: int
    order_qty: int
    
    # Expected cost (at decision time, before observing demand)
    expected_cost: float
    expected_cost_5th: float  # 5th percentile
    expected_cost_95th: float  # 95th percentile
    expected_cost_std: float
    
    # Realized cost (after observing actual demand)
    realized_cost: Optional[float] = None
    
    # State information
    initial_on_hand: int = 0
    initial_intransit: int = 0
    actual_demand: Optional[int] = None


def load_submitted_orders(
    submission_path: Path,
    week_col: str = '2024-04-15'
) -> pd.DataFrame:
    """
    Load submitted orders from CSV.
    
    Args:
        submission_path: Path to submission CSV file
        week_col: Column name for the week of interest
    
    Returns:
        DataFrame with Store, Product, submitted_order
    """
    df = pd.read_csv(submission_path)
    orders = df[['Store', 'Product', week_col]].rename(
        columns={week_col: 'submitted_order'}
    )
    return orders


def compare_orders(
    submitted_path: Path,
    selector_map_path: Path,
    results_path: Path,
    week_idx: int = 0
) -> pd.DataFrame:
    """
    Compare submitted orders vs selector recommendations.
    
    Args:
        submitted_path: Path to submitted orders CSV
        selector_map_path: Path to selector map parquet
        results_path: Path to sequential results parquet
        week_idx: Week index to compare (0 = first week)
    
    Returns:
        DataFrame with comparison results
    """
    # Load data
    submitted = load_submitted_orders(submitted_path)
    selector_map = pd.read_parquet(selector_map_path)
    results = pd.read_parquet(results_path)
    
    # Extract recommended orders
    comparisons = []
    for _, row in selector_map.iterrows():
        store, product, model = row['store'], row['product'], row['model_name']
        
        # Get submitted order
        mask_sub = (submitted['Store'] == store) & (submitted['Product'] == product)
        if not mask_sub.any():
            continue
        submitted_order = int(submitted[mask_sub].iloc[0]['submitted_order'])
        
        # Get recommended order
        mask_res = (results['store'] == store) & \
                   (results['product'] == product) & \
                   (results['model_name'] == model)
        if not mask_res.any():
            continue
        
        orders = results[mask_res].iloc[0]['orders']
        recommended_order = int(orders[week_idx]) if len(orders) > week_idx else 0
        
        comparisons.append(OrderComparison(
            store=store,
            product=product,
            submitted_order=submitted_order,
            recommended_order=recommended_order,
            difference=recommended_order - submitted_order,
            selected_model=model
        ))
    
    return pd.DataFrame([vars(c) for c in comparisons])


def compute_expected_cost_with_ci(
    demand_pmf: np.ndarray,
    starting_inventory: int,
    order_qty: int,
    costs: Costs,
    confidence_level: float = 0.90
) -> Tuple[float, float, float, float]:
    """
    Compute expected cost and confidence interval via PMF.
    
    This calculates the cost distribution for a given order decision,
    then computes mean and percentiles.
    
    Args:
        demand_pmf: PMF of demand (length n, support 0..n-1)
        starting_inventory: Inventory at start of period (after order arrives)
        order_qty: Order quantity
        costs: Cost parameters (holding, shortage)
        confidence_level: Confidence level (default 0.90 for 5th/95th percentiles)
    
    Returns:
        (expected_cost, cost_5th, cost_95th, cost_std)
    """
    demand_pmf = _safe_pmf(demand_pmf)
    total_inventory = starting_inventory + order_qty
    
    # Compute cost for each demand realization
    support = np.arange(len(demand_pmf))
    cost_values = np.zeros(len(demand_pmf))
    
    for d_idx, d in enumerate(support):
        overage = max(0, total_inventory - d)
        underage = max(0, d - total_inventory)
        cost_values[d_idx] = costs.holding * overage + costs.shortage * underage
    
    # Expected cost
    expected_cost = np.sum(demand_pmf * cost_values)
    
    # Compute percentiles from the cost distribution
    # Sort costs and find corresponding percentiles
    sorted_indices = np.argsort(cost_values)
    sorted_costs = cost_values[sorted_indices]
    sorted_pmf = demand_pmf[sorted_indices]
    cost_cdf = np.cumsum(sorted_pmf)
    
    alpha = (1 - confidence_level) / 2
    
    cost_5th = sorted_costs[np.searchsorted(cost_cdf, alpha)]
    cost_95th = sorted_costs[np.searchsorted(cost_cdf, 1 - alpha)]
    
    # Standard deviation
    cost_variance = np.sum(demand_pmf * (cost_values - expected_cost)**2)
    cost_std = np.sqrt(cost_variance)
    
    return expected_cost, cost_5th, cost_95th, cost_std


def compute_realized_cost(
    starting_inventory: int,
    order_qty: int,
    actual_demand: int,
    costs: Costs
) -> float:
    """
    Compute realized cost given actual demand.
    
    Args:
        starting_inventory: Inventory at start of period
        order_qty: Order quantity
        actual_demand: Actual realized demand
        costs: Cost parameters
    
    Returns:
        Realized cost
    """
    total_inventory = starting_inventory + order_qty
    overage = max(0, total_inventory - actual_demand)
    underage = max(0, actual_demand - total_inventory)
    return costs.holding * overage + costs.shortage * underage


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
    
    Args:
        store: Store ID
        product: Product ID
        order_qty: Order quantity placed
        demand_pmf: PMF of demand for the period when order arrives
        initial_on_hand: On-hand inventory before order arrives
        initial_intransit: In-transit inventory arriving with this order
        costs: Cost parameters
        actual_demand: Actual demand (if known, for realized cost)
    
    Returns:
        CostAnalysis with expected and realized costs
    """
    # Starting inventory = on_hand + intransit (order arrives)
    starting_inventory = initial_on_hand + initial_intransit
    
    # Expected cost with confidence interval
    exp_cost, cost_5th, cost_95th, cost_std = compute_expected_cost_with_ci(
        demand_pmf, starting_inventory, order_qty, costs
    )
    
    # Realized cost (if actual demand is known)
    realized_cost = None
    if actual_demand is not None:
        realized_cost = compute_realized_cost(
            starting_inventory, order_qty, actual_demand, costs
        )
    
    return CostAnalysis(
        store=store,
        product=product,
        order_qty=order_qty,
        expected_cost=exp_cost,
        expected_cost_5th=cost_5th,
        expected_cost_95th=cost_95th,
        expected_cost_std=cost_std,
        realized_cost=realized_cost,
        initial_on_hand=initial_on_hand,
        initial_intransit=initial_intransit,
        actual_demand=actual_demand
    )


def analyze_submitted_orders(
    submission_path: Path,
    checkpoints_dir: Path,
    state_df: pd.DataFrame,
    demand_df: Optional[pd.DataFrame] = None,
    costs: Costs = None,
    week_col: str = '2024-04-15',
    pmf_grain: int = 500
) -> pd.DataFrame:
    """
    Analyze all submitted orders with expected costs and confidence intervals.
    
    This function:
    1. Loads submitted orders
    2. Loads forecast PMFs for each SKU
    3. Computes expected cost with 5th/95th percentiles
    4. Optionally computes realized cost if demand data provided
    
    Args:
        submission_path: Path to submission CSV
        checkpoints_dir: Path to forecast checkpoints
        state_df: DataFrame with initial state (on_hand, intransit_1, intransit_2)
        demand_df: Optional DataFrame with actual demand
        costs: Cost parameters (default: holding=0.2, shortage=1.0)
        week_col: Column name for week in submission
        pmf_grain: PMF support size
    
    Returns:
        DataFrame with cost analysis for each SKU
    """
    if costs is None:
        costs = Costs(holding=0.2, shortage=1.0)
    
    # Load submitted orders
    submitted = load_submitted_orders(submission_path, week_col)
    
    # Ensure state is indexed
    if not isinstance(state_df.index, pd.MultiIndex):
        if 'Store' in state_df.columns and 'Product' in state_df.columns:
            state_df = state_df.set_index(['Store', 'Product'])
    
    # Quantile levels
    quantile_levels = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 
                                0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    
    analyses = []
    
    for _, row in submitted.iterrows():
        store = int(row['Store'])
        product = int(row['Product'])
        order_qty = int(row['submitted_order'])
        
        # Get initial state
        try:
            state_row = state_df.loc[(store, product)]
            on_hand = int(state_row['on_hand'])
            intransit_1 = int(state_row['intransit_1'])
            intransit_2 = int(state_row['intransit_2'])
        except KeyError:
            # Skip if state not found
            continue
        
        # Load forecast PMF (fold 0 = week 1)
        # For week 1 order arriving at week 3, we need h2 forecast from fold 0
        # But for simplicity, we'll use h1 from fold 1 (week 2 demand)
        # This is the demand when the order arrives
        checkpoint_path = checkpoints_dir / 'selector' / f'{store}_{product}' / 'fold_1.pkl'
        
        # Try to find any available model for this SKU
        sku_dir = None
        for model_dir in checkpoints_dir.iterdir():
            if model_dir.is_dir():
                potential_sku_dir = model_dir / f'{store}_{product}'
                if potential_sku_dir.exists():
                    sku_dir = potential_sku_dir
                    break
        
        if sku_dir is None:
            continue
        
        # Load fold 1 (week 2 forecast)
        fold_path = sku_dir / 'fold_1.pkl'
        if not fold_path.exists():
            continue
        
        try:
            import pickle
            with open(fold_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            quantiles_df = checkpoint.get('quantiles')
            if quantiles_df is None or len(quantiles_df) < 1:
                continue
            
            # Get h1 from fold 1 (demand for week 2)
            q1 = quantiles_df.loc[1].values if 1 in quantiles_df.index else quantiles_df.iloc[0].values
            demand_pmf = quantiles_to_pmf(q1, quantile_levels, pmf_grain)
            
        except Exception:
            continue
        
        # Get actual demand if available
        actual_demand = None
        if demand_df is not None:
            mask = (demand_df['Store'] == store) & \
                   (demand_df['Product'] == product) & \
                   (demand_df['Week'] == 2)  # Week 2 demand
            if mask.any():
                actual_demand = int(demand_df[mask].iloc[0]['Demand'])
        
        # Analyze order
        analysis = analyze_order_decision(
            store, product, order_qty, demand_pmf,
            on_hand, intransit_2,  # intransit_2 arrives with our order at week 3
            costs, actual_demand
        )
        
        analyses.append(vars(analysis))
    
    return pd.DataFrame(analyses)


def summarize_order_analysis(analysis_df: pd.DataFrame) -> Dict:
    """
    Summarize order analysis results.
    
    Args:
        analysis_df: DataFrame from analyze_submitted_orders
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_skus': len(analysis_df),
        'total_expected_cost': analysis_df['expected_cost'].sum(),
        'mean_expected_cost': analysis_df['expected_cost'].mean(),
        'median_expected_cost': analysis_df['expected_cost'].median(),
        'total_expected_5th': analysis_df['expected_cost_5th'].sum(),
        'total_expected_95th': analysis_df['expected_cost_95th'].sum(),
    }
    
    # Add realized cost if available
    if 'realized_cost' in analysis_df.columns and analysis_df['realized_cost'].notna().any():
        realized = analysis_df[analysis_df['realized_cost'].notna()]
        summary.update({
            'skus_with_realized': len(realized),
            'total_realized_cost': realized['realized_cost'].sum(),
            'mean_realized_cost': realized['realized_cost'].mean(),
            'median_realized_cost': realized['realized_cost'].median(),
        })
    
    return summary

