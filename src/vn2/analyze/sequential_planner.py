"""
Sequential L=2 inventory planner with PMF-based stochastic optimization.

Uses FFT-based convolutions for exact discrete PMF operations and newsvendor
fractile optimization for order quantity selection.
"""

import numpy as np
from numpy.fft import rfft, irfft
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class Costs:
    """Cost parameters for newsvendor problem."""
    holding: float = 0.2  # per unit per period (co)
    shortage: float = 1.0  # per unit per period (cu)


# ==================== PMF Utilities ====================

def _safe_pmf(p: np.ndarray) -> np.ndarray:
    """Ensure PMF is non-negative and normalized."""
    p = np.asarray(p, dtype=float).copy()
    p[p < 0] = 0.0
    s = p.sum()
    if s <= 0:
        raise ValueError("PMF sums to zero.")
    return p / s


def _conv_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Linear convolution using FFT (faster for large arrays).
    
    Returns array of length len(a) + len(b) - 1.
    """
    n = len(a) + len(b) - 1
    L = 1 << (n - 1).bit_length()  # Next power of 2
    A = rfft(a, L)
    B = rfft(b, L)
    c = irfft(A * B, L)[:n]
    c[c < 0] = 0.0  # Numerical safety
    s = c.sum()
    if s > 0:
        c /= s
    return c


def _shift_right(pmf: np.ndarray, k: int) -> np.ndarray:
    """Shift PMF to the right by k positions (add k to all support values)."""
    if k < 0:
        raise ValueError("Negative shift not allowed")
    if k == 0:
        return pmf.copy()
    out = np.zeros(len(pmf) + k, dtype=float)
    out[k:] = pmf
    return out


def leftover_from_stock_and_demand(S: int, D_pmf: np.ndarray) -> np.ndarray:
    """
    Compute PMF of leftover inventory: L = max(S - D, 0).
    
    Args:
        S: Starting stock (integer)
        D_pmf: Demand PMF with support {0, 1, ..., len(D_pmf)-1}
    
    Returns:
        PMF of L with support {0, 1, ..., S}
    """
    D = _safe_pmf(D_pmf)
    S = int(S)
    out = np.zeros(S + 1)
    
    # For d <= S: leftover = S - d
    d_cap = min(S, len(D) - 1)
    for d in range(d_cap + 1):
        out[S - d] += D[d]
    
    # For d > S: leftover = 0 (collapse excess demand to zero inventory)
    if len(D) - 1 > S:
        out[0] += D[S + 1:].sum()
    
    return _safe_pmf(out)


def diff_pmf_D_minus_L(D_pmf: np.ndarray, L_pmf: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Compute PMF of W = D - L via convolution.
    
    Args:
        D_pmf: Demand PMF with support {0, ..., Dmax}
        L_pmf: Inventory PMF with support {0, ..., Lmax}
    
    Returns:
        (W_pmf, w_min): PMF and minimum support value
        W has support {-Lmax, ..., Dmax}
    """
    D = _safe_pmf(D_pmf)
    L = _safe_pmf(L_pmf)
    L_rev = L[::-1]  # Reverse for subtraction convolution
    W = _conv_fft(D, L_rev)
    w_min = -(len(L) - 1)
    return W, w_min


def pmf_quantile(pmf: np.ndarray, offset: int, p: float) -> int:
    """
    Find smallest integer q such that P(W <= q) >= p.
    
    Args:
        pmf: Probability mass function
        offset: Minimum support value (pmf[0] corresponds to offset)
        p: Probability level (0 to 1)
    
    Returns:
        Quantile value as integer
    """
    cdf = np.cumsum(_safe_pmf(pmf))
    idx = int(np.searchsorted(cdf, p))
    return offset + min(idx, len(pmf) - 1)


def expected_pos_neg_from_Z(z_pmf: np.ndarray, z_min: int) -> Tuple[float, float]:
    """
    Compute E[max(Z, 0)] and E[max(-Z, 0)] from PMF of Z.
    
    Args:
        z_pmf: PMF of random variable Z
        z_min: Minimum support value
    
    Returns:
        (E_over, E_under): Expected overage and underage
    """
    idx = np.arange(len(z_pmf))
    z_vals = z_min + idx
    pos_mask = z_vals > 0
    neg_mask = z_vals < 0
    E_over = float((z_pmf[pos_mask] * z_vals[pos_mask]).sum())
    E_under = float((z_pmf[neg_mask] * (-z_vals[neg_mask])).sum())
    return E_over, E_under


def leftover_from_Z(z_pmf: np.ndarray, z_min: int) -> np.ndarray:
    """
    Compute PMF of max(Z, 0) from PMF of Z.
    
    Used to propagate inventory state: leftover = max(inventory_before - demand, 0).
    
    Args:
        z_pmf: PMF of Z
        z_min: Minimum support value
    
    Returns:
        PMF of max(Z, 0) with support {0, ..., max(Z)}
    """
    idx = np.arange(len(z_pmf))
    z_vals = z_min + idx
    max_z = int(max(0, z_vals.max()))
    out = np.zeros(max_z + 1)
    
    # All negative values collapse to 0
    out[0] += float(z_pmf[z_vals <= 0].sum())
    
    # Positive values map directly
    pos_idx = np.where(z_vals > 0)[0]
    for i in pos_idx:
        out[int(z_vals[i])] += z_pmf[i]
    
    s = out.sum()
    if s > 0:
        out /= s
    return out


# ==================== Order Selection ====================

def choose_order_L2(
    h1_pmf: np.ndarray,
    h2_pmf: np.ndarray,
    I0: int,
    Q1: int,
    Q2: int,
    costs: Costs,
    micro_refine: bool = True
) -> Tuple[int, float]:
    """
    Choose optimal order quantity for L=2 lead time using newsvendor fractile.
    
    LEAD TIME SEMANTICS: Order placed at start of week t arrives at start of week t+2.
    
    At decision epoch t:
    - I0 = on-hand entering week t (before any arrivals)
    - Q1 = order arriving at start of week t (placed at t-2)
    - Q2 = order arriving at start of week t+1 (placed at t-1)
    - h1_pmf = demand PMF for week t
    - h2_pmf = demand PMF for week t+1
    - q = order to place now at t (arrives at week t+2)
    
    Steps:
    1) Compute leftover after week t: L1 = max((I0 + Q1) - D_t, 0)
    2) Inventory at start of week t+1: Lpre = L1 + Q2
    3) Choose q to minimize expected cost at week t+1: cu*shortage + co*overage
       where shortage = max(D_{t+1} - (Lpre + q), 0)
             overage = max((Lpre + q) - D_{t+1}, 0)
    4) Use newsvendor fractile p* = cu/(cu+co) on W = D_{t+1} - Lpre
    
    Args:
        h1_pmf: Demand PMF for current week t
        h2_pmf: Demand PMF for next week t+1
        I0: On-hand inventory at start of epoch t (before Q1 arrival)
        Q1: Order arriving at start of week t (placed at t-2)
        Q2: Order arriving at start of week t+1 (placed at t-1)
        costs: Cost parameters (holding=co, shortage=cu)
        micro_refine: If True, check q±{1,2} around fractile solution
    
    Returns:
        (q_star, cost_star): Optimal order quantity and expected cost at week t+1
    """
    h1 = _safe_pmf(h1_pmf)
    h2 = _safe_pmf(h2_pmf)
    cu = costs.shortage
    co = costs.holding
    
    # Step 1: L1 distribution after week t+1
    S1 = int(I0) + int(Q1)
    L1 = leftover_from_stock_and_demand(S1, h1)
    
    # Step 2: Lpre = L1 + Q2 (shift PMF by Q2)
    Lpre = _shift_right(L1, int(Q2))
    
    # Step 3: Compute newsvendor fractile
    W, w_min = diff_pmf_D_minus_L(h2, Lpre)
    p_star = cu / (cu + co)
    q0 = max(0, pmf_quantile(W, w_min, p_star))
    
    # Step 4: Micro-refinement (evaluate q0 and neighbors)
    D_rev = h2[::-1]
    
    def eval_q(q):
        Z = _conv_fft(_shift_right(Lpre, q), D_rev)  # Z = (Lpre + q) - D2
        z_min = -(len(D_rev) - 1)
        E_over, E_under = expected_pos_neg_from_Z(Z, z_min)
        return co * E_over + cu * E_under
    
    if micro_refine:
        candidates = [q0, max(0, q0 - 1), q0 + 1, max(0, q0 - 2), q0 + 2]
        best_q, best_cost = min(
            ((q, eval_q(q)) for q in candidates),
            key=lambda x: x[1]
        )
    else:
        best_q = q0
        best_cost = eval_q(q0)
    
    return int(best_q), float(best_cost)


# ==================== Sequential Runner ====================

@dataclass
class SequentialResult:
    """Result of sequential L=2 planning."""
    orders_by_epoch: List[int]  # q_t for epochs 0..H-1
    costs_by_epoch: List[float]  # realized cost at each epoch (week t+2 cost for decision at t)
    total_cost: float  # sum of decision-affected costs
    coverage: float  # fraction of epochs with valid forecasts
    n_missing: int  # number of epochs with missing forecasts
    diagnostics: Dict  # additional info


def run_sequential_L2(
    forecasts_h1: List[Optional[np.ndarray]],  # PMF for each epoch's h+1 demand
    forecasts_h2: List[Optional[np.ndarray]],  # PMF for each epoch's h+2 demand
    actuals: List[int],  # actual demand for each week (index 0 = week 1, etc.)
    I0: int,
    Q1: int,
    Q2: int,
    costs: Costs,
    fallback_pmf: Optional[np.ndarray] = None
) -> SequentialResult:
    """
    Run sequential L=2 planning over H epochs.
    
    At each epoch t (t=0..H-1):
    - Use forecasts_h1[t] and forecasts_h2[t] to choose order q_t
    - q_t arrives at week t+2
    - Realize cost at week t+2 using actuals[t+2]
    - Advance state using actuals[t+1]
    
    Args:
        forecasts_h1: List of h+1 PMFs (or None if missing) for each epoch
        forecasts_h2: List of h+2 PMFs (or None if missing) for each epoch
        actuals: List of actual demands for weeks 1..H+2 (need H+2 weeks of actuals)
        I0: Initial on-hand inventory
        Q1: Order arriving at week 1
        Q2: Order arriving at week 2
        costs: Cost parameters
        fallback_pmf: PMF to use when forecast is missing (if None, use q=0)
    
    Returns:
        SequentialResult with orders, costs, and diagnostics
    """
    H = len(forecasts_h1)
    assert len(forecasts_h2) == H, "h1 and h2 must have same length"
    assert len(actuals) >= H + 2, f"Need at least {H+2} weeks of actuals for {H} epochs"
    
    # Initialize state
    state_I0 = int(I0)
    state_Q1 = int(Q1)
    state_Q2 = int(Q2)
    
    orders = []
    costs_realized = []
    n_missing = 0
    epoch_flags = []
    
    for t in range(H):
        # Check for missing forecasts
        h1 = forecasts_h1[t]
        h2 = forecasts_h2[t]
        
        if h1 is None or h2 is None:
            # Missing forecast: use fallback or q=0
            if fallback_pmf is not None:
                h1 = fallback_pmf if h1 is None else h1
                h2 = fallback_pmf if h2 is None else h2
                flag = 'fallback'
            else:
                q_t = 0
                orders.append(q_t)
                n_missing += 1
                epoch_flags.append('missing')
                
                # Still need to advance state and compute cost
                # Cost at week t+2 from decision at epoch t
                if t + 2 < len(actuals):
                    # Week t+1 dynamics
                    S1 = state_I0 + state_Q1
                    L1 = max(0, S1 - actuals[t + 1])
                    
                    # Week t+2 dynamics
                    S2 = L1 + state_Q2
                    y2 = actuals[t + 2]
                    overage = max(0, S2 - y2)
                    underage = max(0, y2 - S2)
                    cost = costs.holding * overage + costs.shortage * underage
                    costs_realized.append(cost)
                else:
                    costs_realized.append(0.0)
                
                # Advance state
                if t + 1 < len(actuals):
                    S1 = state_I0 + state_Q1
                    state_I0 = max(0, S1 - actuals[t + 1])
                    state_Q1 = state_Q2
                    state_Q2 = q_t
                
                continue
        
        # Valid forecasts: choose order
        q_t, exp_cost = choose_order_L2(h1, h2, state_I0, state_Q1, state_Q2, costs)
        orders.append(q_t)
        epoch_flags.append('ok' if fallback_pmf is None else 'fallback')
        
        # Compute realized cost at week t+2
        if t + 2 < len(actuals):
            # Week t+1 dynamics
            S1 = state_I0 + state_Q1
            y1 = actuals[t + 1]
            L1 = max(0, S1 - y1)
            
            # Week t+2 dynamics
            S2 = L1 + state_Q2
            y2 = actuals[t + 2]
            overage = max(0, S2 - y2)
            underage = max(0, y2 - S2)
            cost = costs.holding * overage + costs.shortage * underage
            costs_realized.append(cost)
        else:
            costs_realized.append(0.0)
        
        # Advance state for next epoch
        if t + 1 < len(actuals):
            S1 = state_I0 + state_Q1
            state_I0 = max(0, S1 - actuals[t + 1])
            state_Q1 = state_Q2
            state_Q2 = q_t
    
    # Aggregate
    total_cost = sum(costs_realized)
    coverage = 1.0 - (n_missing / H) if H > 0 else 0.0
    
    return SequentialResult(
        orders_by_epoch=orders,
        costs_by_epoch=costs_realized,
        total_cost=total_cost,
        coverage=coverage,
        n_missing=n_missing,
        diagnostics={
            'H': H,
            'initial_state': (I0, Q1, Q2),
            'epoch_flags': epoch_flags,
            'final_state': (state_I0, state_Q1, state_Q2)
        }
    )

