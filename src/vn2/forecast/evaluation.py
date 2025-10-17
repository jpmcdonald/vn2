"""
Evaluation metrics for density forecasts.

Includes traditional metrics (pinball, coverage) and cost-based metrics
for value-optimized model selection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats


def pinball_loss(y_true: float, y_pred: float, quantile: float) -> float:
    """
    Pinball loss for a single quantile prediction.
    
    Args:
        y_true: Actual value
        y_pred: Predicted quantile
        quantile: Quantile level (0-1)
        
    Returns:
        Pinball loss
    """
    error = y_true - y_pred
    return np.maximum(quantile * error, (quantile - 1) * error)


def average_pinball_loss(
    y_true: np.ndarray,
    quantiles_df: pd.DataFrame
) -> float:
    """
    Average pinball loss across all quantiles.
    
    Args:
        y_true: Actual values (length = steps)
        quantiles_df: DataFrame with columns = quantiles, rows = steps
        
    Returns:
        Mean pinball loss
    """
    quantile_levels = quantiles_df.columns.values
    
    total_loss = 0.0
    n = 0
    
    for step in range(len(y_true)):
        if step + 1 not in quantiles_df.index:
            continue
        
        y_step = y_true[step]
        preds_step = quantiles_df.loc[step + 1]
        
        for q, pred in zip(quantile_levels, preds_step.values):
            total_loss += pinball_loss(y_step, pred, q)
            n += 1
    
    return total_loss / n if n > 0 else np.nan


def coverage_metrics(
    y_true: np.ndarray,
    quantiles_df: pd.DataFrame,
    levels: List[float] = [0.8, 0.9, 0.95]
) -> Dict[str, float]:
    """
    Compute prediction interval coverage and width.
    
    Args:
        y_true: Actual values
        quantiles_df: Quantile predictions
        levels: Coverage levels to evaluate
        
    Returns:
        Dictionary with coverage_X and width_X metrics
    """
    results = {}
    quantile_levels = quantiles_df.columns.values
    
    for level in levels:
        lower_q = (1 - level) / 2
        upper_q = 1 - lower_q
        
        # Find closest quantiles
        lower_idx = np.argmin(np.abs(quantile_levels - lower_q))
        upper_idx = np.argmin(np.abs(quantile_levels - upper_q))
        
        lower_col = quantile_levels[lower_idx]
        upper_col = quantile_levels[upper_idx]
        
        # Check coverage
        in_interval = []
        widths = []
        
        for step in range(len(y_true)):
            if step + 1 not in quantiles_df.index:
                continue
            
            y_step = y_true[step]
            lower_pred = quantiles_df.loc[step + 1, lower_col]
            upper_pred = quantiles_df.loc[step + 1, upper_col]
            
            in_interval.append((y_step >= lower_pred) and (y_step <= upper_pred))
            widths.append(upper_pred - lower_pred)
        
        if len(in_interval) > 0:
            results[f'coverage_{int(level*100)}'] = np.mean(in_interval)
            results[f'width_{int(level*100)}'] = np.mean(widths)
        else:
            results[f'coverage_{int(level*100)}'] = np.nan
            results[f'width_{int(level*100)}'] = np.nan
    
    return results


def crps_score(y_true: float, quantiles: np.ndarray, quantile_levels: np.ndarray) -> float:
    """
    Continuous Ranked Probability Score (CRPS) for a single observation.
    
    Approximates CRPS from quantile predictions using trapezoidal integration.
    
    Args:
        y_true: Actual value
        quantiles: Predicted quantiles
        quantile_levels: Quantile levels (0-1)
        
    Returns:
        CRPS value
    """
    # Sort by quantile levels
    sorted_idx = np.argsort(quantile_levels)
    q_levels = quantile_levels[sorted_idx]
    q_values = quantiles[sorted_idx]
    
    # Compute indicator function I(y <= q)
    indicators = (y_true <= q_values).astype(float)
    
    # CRPS = integral |F(x) - I(y <= x)|² dx
    # Approximate via trapezoid over quantiles
    integrand = (q_levels - indicators) ** 2
    crps = np.trapz(integrand, q_values)
    
    return crps


def average_crps(y_true: np.ndarray, quantiles_df: pd.DataFrame) -> float:
    """
    Average CRPS across forecast horizon.
    
    Args:
        y_true: Actual values
        quantiles_df: Quantile predictions
        
    Returns:
        Mean CRPS
    """
    quantile_levels = quantiles_df.columns.values
    crps_values = []
    
    for step in range(len(y_true)):
        if step + 1 not in quantiles_df.index:
            continue
        
        y_step = y_true[step]
        q_step = quantiles_df.loc[step + 1].values
        
        crps_val = crps_score(y_step, q_step, quantile_levels)
        crps_values.append(crps_val)
    
    return np.mean(crps_values) if len(crps_values) > 0 else np.nan


def cost_based_metric(
    y_true: np.ndarray,
    quantiles_df: pd.DataFrame,
    initial_stock: float = 0,
    costs: Dict[str, float] = {'holding': 0.2, 'shortage': 1.0},
    n_sims: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Cost-based metric via Monte Carlo simulation of ordering decisions.
    
    This is the VALUE metric for model selection.
    
    Simulates:
    1. Sample demand from forecast SIP
    2. Determine optimal order (base-stock or newsvendor)
    3. Simulate week, compute costs
    4. Return expected cost
    
    Args:
        y_true: Actual observed demand
        quantiles_df: Forecast quantiles
        initial_stock: Starting inventory
        costs: Holding and shortage costs
        n_sims: Number of simulations
        seed: Random seed
        
    Returns:
        Dictionary with cost metrics
    """
    rng = np.random.default_rng(seed)
    quantile_levels = quantiles_df.columns.values
    
    # Sample demand scenarios from quantiles
    demand_sims = np.zeros((n_sims, len(y_true)))
    for step in range(len(y_true)):
        if step + 1 not in quantiles_df.index:
            demand_sims[:, step] = y_true[step]  # Fallback to actual
            continue
        
        q_vals = quantiles_df.loc[step + 1].values
        # Inverse transform sampling
        u = rng.uniform(0, 1, n_sims)
        demand_sims[:, step] = np.interp(u, quantile_levels, q_vals)
    
    # Simple newsvendor optimal order for each step
    # Critical fractile = c_s / (c_h + c_s)
    critical_fractile = costs['shortage'] / (costs['holding'] + costs['shortage'])
    
    total_costs = []
    for sim_idx in range(n_sims):
        stock = initial_stock
        sim_cost = 0.0
        
        for step in range(len(y_true)):
            demand = demand_sims[sim_idx, step]
            
            # Optimal order to newsvendor level (simplified)
            if step + 1 in quantiles_df.index:
                q_vals = quantiles_df.loc[step + 1].values
                target_stock = np.interp(critical_fractile, quantile_levels, q_vals)
                order = max(0, target_stock - stock)
            else:
                order = 0
            
            # Simulate week
            stock += order
            sales = min(demand, stock)
            shortage = demand - sales
            stock_end = stock - sales
            
            # Costs
            sim_cost += costs['holding'] * stock_end
            sim_cost += costs['shortage'] * shortage
            
            stock = stock_end
        
        total_costs.append(sim_cost)
    
    return {
        'expected_cost': np.mean(total_costs),
        'cost_std': np.std(total_costs),
        'cost_q05': np.quantile(total_costs, 0.05),
        'cost_q95': np.quantile(total_costs, 0.95),
    }


def evaluate_forecast(
    y_true: np.ndarray,
    quantiles_df: pd.DataFrame,
    include_cost: bool = True
) -> Dict[str, float]:
    """
    Comprehensive evaluation of a forecast.
    
    Args:
        y_true: Actual values
        quantiles_df: Quantile predictions
        include_cost: Whether to compute cost-based metrics
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # Pinball loss
    metrics['pinball_loss'] = average_pinball_loss(y_true, quantiles_df)
    
    # Coverage
    coverage_results = coverage_metrics(y_true, quantiles_df)
    metrics.update(coverage_results)
    
    # CRPS
    metrics['crps'] = average_crps(y_true, quantiles_df)
    
    # Point forecast error (median)
    if 0.5 in quantiles_df.columns:
        median_preds = quantiles_df[0.5].values
        if len(median_preds) == len(y_true):
            metrics['mae'] = np.mean(np.abs(y_true - median_preds))
            metrics['rmse'] = np.sqrt(np.mean((y_true - median_preds) ** 2))
    
    # Cost-based (computationally expensive)
    if include_cost:
        cost_metrics = cost_based_metric(y_true, quantiles_df)
        metrics.update(cost_metrics)
    
    return metrics

