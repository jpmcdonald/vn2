"""Monte Carlo optimization over SIP samples to close Jensen's gap"""

from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from vn2.sim import Simulator, Costs, LeadTime


def candidate_grid(
    base: pd.Series, 
    span: float, 
    sigma_lt: pd.Series, 
    cap: int = 400
) -> Dict[tuple, np.ndarray]:
    """
    Generate candidate order quantities around base-stock estimate.
    
    Args:
        base: Base-stock center per SKU
        span: Grid span in multiples of sigma_lt
        sigma_lt: Lead-time demand standard deviation per SKU
        cap: Maximum order quantity
        
    Returns:
        Dict mapping SKU index to array of candidate quantities
    """
    step = (0.2 * sigma_lt).clip(lower=1).round().astype(int)
    k = ((span * sigma_lt) / step).clip(lower=2).round().astype(int)
    
    grids = {}
    
    for idx_val in base.index:
        ks = int(k.loc[idx_val])
        st = int(step.loc[idx_val])
        c = int(base.loc[idx_val])
        
        lo = max(0, c - ks * st)
        hi = min(cap, c + ks * st)
        
        grids[idx_val] = np.arange(lo, hi + 1, st)
    
    return grids


def expected_cost_for_order(
    state0: pd.DataFrame,
    order_now: pd.Series,
    demand_paths: np.ndarray,
    costs: Costs,
    lt: LeadTime,
    recourse_policy: str = "zero"
) -> float:
    """
    Compute expected cost over Monte Carlo samples.
    
    Args:
        state0: Initial state
        order_now: Order quantities to evaluate (week 0 decision)
        demand_paths: Array [n_sims, horizon, n_items]
        costs: Cost structure
        lt: Lead time configuration
        recourse_policy: How to handle future orders ("zero" or "base_stock")
        
    Returns:
        Expected total cost
    """
    sim = Simulator(costs, lt)
    idx = state0.index
    n_sims, H, N = demand_paths.shape
    
    total_cost = 0.0
    
    for s in range(n_sims):
        state = state0.copy()
        
        for t in range(H):
            demand_t = pd.Series(demand_paths[s, t, :], index=idx)
            
            # Week 0: use provided order; after: simple recourse
            if t == 0:
                order_t = order_now
            else:
                # Simple recourse: order = 0 (neutral baseline)
                # Could be replaced with base-stock or other policy
                order_t = pd.Series(0, index=idx)
            
            state, cost_dict = sim.step(state, demand_t, order_t)
            total_cost += cost_dict["total"]
    
    return total_cost / n_sims


def optimize_bruteforce_mc(
    state: pd.DataFrame,
    sip_samples: np.ndarray,
    base_upto: pd.Series,
    sigma_lt: pd.Series,
    costs: Costs,
    lt: LeadTime,
    span: float = 2.0,
    cap: int = 400,
) -> pd.Series:
    """
    Optimize orders via brute-force Monte Carlo over SIP samples.
    
    For each SKU independently:
    - Generate candidate grid around base-stock estimate
    - Evaluate expected cost for each candidate
    - Select minimum
    
    Args:
        state: Current inventory state
        sip_samples: Demand scenarios [n_sims, horizon, n_items]
        base_upto: Base-stock center estimate per SKU
        sigma_lt: Lead-time demand std dev per SKU
        costs: Cost structure
        lt: Lead time
        span: Grid span multiplier
        cap: Max order quantity
        
    Returns:
        Optimal order quantities per SKU
    """
    idx = state.index
    grids = candidate_grid(base_upto, span, sigma_lt, cap)
    
    best_orders = pd.Series(0, index=idx, dtype=int)
    
    # Independent optimization per SKU
    for i, idx_val in enumerate(idx):
        candidates = grids[idx_val]
        
        # Extract demand paths for this SKU
        d_i = sip_samples[:, :, [i]]  # shape [n_sims, horizon, 1]
        
        costs_candidates = []
        
        for q in candidates:
            # Order vector: q for this SKU, 0 for others
            order_now = pd.Series(0, index=idx)
            order_now.loc[idx_val] = q
            
            c = expected_cost_for_order(state, order_now, d_i, costs, lt)
            costs_candidates.append((q, c))
        
        # Select minimum cost
        q_star, _ = min(costs_candidates, key=lambda x: x[1])
        best_orders.loc[idx_val] = int(q_star)
    
    return best_orders

