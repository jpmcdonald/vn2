"""
SIP (Stochastic Inventory Position) optimization for sequential newsvendor problem.

This module implements discrete optimization over integer order quantities using
PMF convolution to handle uncertainty in both demand and inventory levels.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Costs:
    """Cost parameters for newsvendor problem."""
    holding: float = 0.2  # per unit per period
    shortage: float = 1.0  # per unit per period


def quantiles_to_pmf(
    quantiles: np.ndarray,
    quantile_levels: np.ndarray,
    grain: int = 1000
) -> np.ndarray:
    """
    Convert quantile forecast to discrete PMF via interpolation.
    
    Args:
        quantiles: Array of quantile values (e.g., 13 quantiles from model)
        quantile_levels: Array of quantile levels (e.g., [0.01, 0.05, ..., 0.99])
        grain: Maximum support for PMF (0 to grain inclusive)
    
    Returns:
        pmf: Array of length (grain+1) with probabilities summing to 1
    """
    # Ensure quantiles are non-negative and sorted
    quantiles = np.maximum(quantiles, 0)
    quantiles = np.sort(quantiles)
    
    # Create CDF by interpolating quantile function
    # Add boundary points for extrapolation
    q_extended = np.concatenate([[0], quantile_levels, [1]])
    v_extended = np.concatenate([[0], quantiles, [quantiles[-1]]])
    
    # Interpolate CDF at integer points
    support = np.arange(grain + 1)
    cdf = np.interp(support, v_extended, q_extended)
    
    # Convert CDF to PMF via differencing
    pmf = np.diff(cdf, prepend=0)
    
    # Ensure valid PMF (non-negative, sums to 1)
    pmf = np.maximum(pmf, 0)
    pmf_sum = pmf.sum()
    if pmf_sum > 0:
        pmf = pmf / pmf_sum
    else:
        # Degenerate case: put all mass at zero
        pmf[0] = 1.0
    
    return pmf


def convolve_inventory(
    I0: int,
    Q1: int,
    pmf_D1: np.ndarray
) -> np.ndarray:
    """
    Compute end-of-period inventory distribution: I1_end = (I0 + Q1 - D1)^+
    
    Args:
        I0: Initial inventory (integer)
        Q1: Incoming order quantity (integer)
        pmf_D1: Demand PMF for period 1
    
    Returns:
        pmf_I1_end: PMF of inventory at end of period 1
    """
    grain = len(pmf_D1) - 1
    starting_inv = I0 + Q1
    
    # I1_end[i] = P(starting_inv - D1 = i | D1 <= starting_inv)
    #           + P(D1 > starting_inv) if i == 0
    pmf_I1_end = np.zeros(grain + 1)
    
    for d in range(grain + 1):
        if pmf_D1[d] > 0:
            remaining = max(0, starting_inv - d)
            if remaining <= grain:
                pmf_I1_end[remaining] += pmf_D1[d]
    
    # Normalize (should already sum to 1, but ensure)
    pmf_sum = pmf_I1_end.sum()
    if pmf_sum > 0:
        pmf_I1_end = pmf_I1_end / pmf_sum
    
    return pmf_I1_end


def expected_cost_for_order(
    pmf_I2_pre: np.ndarray,
    pmf_D2: np.ndarray,
    Q: int,
    costs: Costs
) -> float:
    """
    Compute expected cost for week 2 given order quantity Q.
    
    Cost = h * E[(I2_pre + Q - D2)^+] + p * E[(D2 - I2_pre - Q)^+]
    
    Args:
        pmf_I2_pre: PMF of inventory at start of week 2 (before order arrives)
        pmf_D2: PMF of demand in week 2
        Q: Order quantity (integer)
        costs: Cost parameters
    
    Returns:
        expected_cost: Expected cost for week 2
    """
    grain = len(pmf_D2) - 1
    expected_cost = 0.0
    
    # Iterate over all (I_pre, D) combinations
    for i_pre in range(grain + 1):
        if pmf_I2_pre[i_pre] == 0:
            continue
        
        for d in range(grain + 1):
            if pmf_D2[d] == 0:
                continue
            
            prob = pmf_I2_pre[i_pre] * pmf_D2[d]
            inventory_after = i_pre + Q
            
            # Overage and underage
            overage = max(0, inventory_after - d)
            underage = max(0, d - inventory_after)
            
            cost = costs.holding * overage + costs.shortage * underage
            expected_cost += prob * cost
    
    return expected_cost


def optimize_order(
    pmf_I1_end: np.ndarray,
    Q2: int,
    pmf_D2: np.ndarray,
    costs: Costs,
    max_Q: int = 1000
) -> Tuple[int, float]:
    """
    Find optimal order quantity Q* that minimizes expected cost for week 2.
    
    Args:
        pmf_I1_end: PMF of inventory at end of week 1
        Q2: Order arriving at start of week 2 (already placed, deterministic)
        pmf_D2: PMF of demand in week 2
        costs: Cost parameters
        max_Q: Maximum order quantity to consider
    
    Returns:
        Q_opt: Optimal order quantity
        cost_opt: Expected cost at optimal Q
    """
    # I2_pre = I1_end + Q2 (convolution with deterministic Q2 is just a shift)
    grain = len(pmf_I1_end) - 1
    pmf_I2_pre = np.zeros(grain + 1)
    
    for i1 in range(grain + 1):
        if pmf_I1_end[i1] > 0:
            i2_pre = min(i1 + Q2, grain)
            pmf_I2_pre[i2_pre] += pmf_I1_end[i1]
    
    # Normalize
    pmf_sum = pmf_I2_pre.sum()
    if pmf_sum > 0:
        pmf_I2_pre = pmf_I2_pre / pmf_sum
    
    # Brute-force search over integer Q
    best_Q = 0
    best_cost = float('inf')
    
    for Q in range(max_Q + 1):
        cost = expected_cost_for_order(pmf_I2_pre, pmf_D2, Q, costs)
        if cost < best_cost:
            best_cost = cost
            best_Q = Q
    
    return best_Q, best_cost


def compute_realized_metrics(
    Q_opt: int,
    I0: int,
    Q1: int,
    Q2: int,
    y1_true: int,
    y2_true: int,
    costs: Costs
) -> dict:
    """
    Compute realized (deterministic) metrics using actual demand.
    
    Args:
        Q_opt: Optimal order quantity chosen for week 3
        I0: Initial inventory
        Q1: Order arriving at start of week 1
        Q2: Order arriving at start of week 2
        y1_true: Actual demand in week 1
        y2_true: Actual demand in week 2
        costs: Cost parameters
    
    Returns:
        metrics: Dict with realized costs, service level, fill rate, etc.
    """
    # Week 1 dynamics
    I1_start = I0 + Q1
    sales_1 = min(y1_true, I1_start)
    I1_end = max(0, I1_start - y1_true)
    
    # Week 2 dynamics
    I2_start = I1_end + Q2
    sales_2 = min(y2_true, I2_start)
    I2_end = max(0, I2_start - y2_true)
    
    # Costs (week 2 only for decision evaluation)
    holding_cost_2 = costs.holding * I2_end
    shortage_cost_2 = costs.shortage * max(0, y2_true - I2_start)
    total_cost_2 = holding_cost_2 + shortage_cost_2
    
    # Service metrics (week 2)
    service_level_2 = 1.0 if y2_true <= I2_start else 0.0
    fill_rate_2 = sales_2 / y2_true if y2_true > 0 else 1.0
    
    # Oracle comparison: what if we ordered exactly y2_true?
    oracle_cost_2 = 0.0  # Perfect forecast
    regret_qty = abs(Q_opt - y2_true)
    
    return {
        'realized_cost_w2': total_cost_2,
        'holding_cost_w2': holding_cost_2,
        'shortage_cost_w2': shortage_cost_2,
        'service_level_w2': service_level_2,
        'fill_rate_w2': fill_rate_2,
        'regret_qty': regret_qty,
        'sales_w1': sales_1,
        'sales_w2': sales_2,
    }


def cost_curve_vs_Q(
    pmf_h1: np.ndarray,
    pmf_h2: np.ndarray,
    I0: int,
    Q1: int,
    costs: Costs,
    Q_max: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute expected cost curve over integer order quantities Q.
    
    Args:
        pmf_h1: PMF for horizon 1 demand
        pmf_h2: PMF for horizon 2 demand
        I0: Initial inventory
        Q1: Order arriving at start of week 1
        costs: Cost parameters
        Q_max: Maximum Q to evaluate (default: len(pmf_h2))
    
    Returns:
        Q_grid: array of order quantities
        expected_costs: array of expected costs for each Q
    """
    if Q_max is None:
        Q_max = len(pmf_h2) - 1
    
    Q_grid = np.arange(0, Q_max + 1)
    expected_costs = np.zeros(len(Q_grid))
    
    for i, Q in enumerate(Q_grid):
        # Compute expected cost for this Q
        # Convolve inventory after week 1 with demand in week 2
        I1_pmf = convolve_inventory(I0, Q1, pmf_h1)
        I2_pmf = convolve_inventory_dist(I1_pmf, Q, pmf_h2)
        
        # Expected holding cost
        support_I2 = np.arange(len(I2_pmf))
        E_holding = costs.holding * np.dot(I2_pmf, support_I2)
        
        # Expected shortage cost
        # Shortage = max(0, demand - inventory_before_demand)
        # For each possible I2_start, compute expected shortage over demand
        I2_start_pmf = convolve_inventory_dist(I1_pmf, Q, np.ones(1))  # I1 + Q (deterministic Q)
        support_I2_start = np.arange(len(I2_start_pmf))
        
        E_shortage = 0.0
        for I2_start, p_I2_start in zip(support_I2_start, I2_start_pmf):
            if p_I2_start > 0:
                # E[max(0, D - I2_start)]
                support_D = np.arange(len(pmf_h2))
                shortage_vals = np.maximum(0, support_D - I2_start)
                E_shortage += p_I2_start * costs.shortage * np.dot(pmf_h2, shortage_vals)
        
        expected_costs[i] = E_holding + E_shortage
    
    return Q_grid, expected_costs


def convolve_inventory_dist(
    I_pmf: np.ndarray,
    Q: int,
    D_pmf: np.ndarray
) -> np.ndarray:
    """
    Convolve inventory distribution with deterministic order and demand PMF.
    
    I_end = max(0, I_start + Q - D)
    
    Args:
        I_pmf: PMF of starting inventory
        Q: deterministic order quantity
        D_pmf: PMF of demand
    
    Returns:
        I_end_pmf: PMF of ending inventory
    """
    max_I_end = len(I_pmf) + Q
    I_end_pmf = np.zeros(max_I_end)
    
    for I_start, p_I in enumerate(I_pmf):
        if p_I == 0:
            continue
        for D, p_D in enumerate(D_pmf):
            if p_D == 0:
                continue
            I_end = max(0, I_start + Q - D)
            if I_end < max_I_end:
                I_end_pmf[I_end] += p_I * p_D
    
    # Normalize
    total = I_end_pmf.sum()
    if total > 0:
        I_end_pmf /= total
    
    return I_end_pmf


def compute_realized_w2(
    Q: int,
    I0: int,
    Q1: int,
    Q2: int,
    y1: int,
    y2: int,
    costs: Costs
) -> float:
    """
    Compute realized cost for week 2 only (decision-affected week).
    
    This is a simplified version of compute_realized_metrics that returns
    only the week 2 cost, used for w8 evaluation.
    
    Args:
        Q: Order quantity placed at time 0 (arrives at start of week 3, not used here)
        I0: Initial inventory at time 0
        Q1: Order arriving at start of week 1 (deterministic, already placed)
        Q2: Order arriving at start of week 2 (deterministic, already placed)
        y1: Actual demand in week 1
        y2: Actual demand in week 2
        costs: Cost parameters
    
    Returns:
        realized_cost_w2: Total cost (holding + shortage) for week 2 only
    """
    # Week 1 dynamics (to compute inventory state entering week 2)
    I1_start = I0 + Q1
    I1_end = max(0, I1_start - y1)
    
    # Week 2 dynamics
    I2_start = I1_end + Q2
    sales_2 = min(y2, I2_start)
    I2_end = max(0, I2_start - y2)
    
    # Week 2 costs only
    holding_cost_2 = costs.holding * I2_end
    shortage_cost_2 = costs.shortage * max(0, y2 - I2_start)
    total_cost_2 = holding_cost_2 + shortage_cost_2
    
    return total_cost_2

