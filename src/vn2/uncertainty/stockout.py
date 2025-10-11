"""Stock-out imputation using full predictive distribution"""

import numpy as np
import pandas as pd


def trapz_integral_over_quantiles(
    q_levels: np.ndarray, 
    q_values: np.ndarray, 
    p0: float
) -> float:
    """
    Integrate Q(p) dp from p0 to 1 using trapezoids.
    
    Args:
        q_levels: Quantile levels (probabilities)
        q_values: Quantile values
        p0: Lower integration limit
        
    Returns:
        Integral value
    """
    p = q_levels
    v = q_values
    
    mask = p >= p0
    p2, v2 = p[mask], v[mask]
    
    if len(p2) == 0:
        return 0.0
    
    # Insert p0 by linear interpolation if needed
    if p2[0] > p0:
        i = np.searchsorted(p, p0)
        if i > 0:
            p_lo, p_hi = p[i - 1], p[i]
            v_lo, v_hi = v[i - 1], v[i]
            v0 = v_lo + (v_hi - v_lo) * (p0 - p_lo) / (p_hi - p_lo + 1e-12)
            p2 = np.insert(p2, 0, p0)
            v2 = np.insert(v2, 0, v0)
    
    return float(np.trapz(v2, p2))


def expected_tail(
    q_levels: np.ndarray, 
    q_values: np.ndarray, 
    s: float
) -> tuple[float, float]:
    """
    Compute tail expectations from quantile function.
    
    E[D | D >= s] = integral_{p0..1} Q(p) dp / (1 - p0)
    E[(D - s)+]   = integral_{p0..1} (Q(p) - s) dp
    
    where p0 = F(s) via inverse quantile
    
    Args:
        q_levels: Quantile levels
        q_values: Quantile values  
        s: Stock level
        
    Returns:
        (conditional_mean, expected_positive_part)
    """
    p = q_levels
    v = q_values
    
    # Locate p0 = F(s) by inverting Q
    i = np.searchsorted(v, s, side="right")
    
    if i == 0:
        p0 = 0.0
    elif i >= len(v):
        p0 = 1.0 - 1e-12
    else:
        v_lo, v_hi = v[i - 1], v[i]
        p_lo, p_hi = p[i - 1], p[i]
        p0 = p_lo + (p_hi - p_lo) * (s - v_lo) / (v_hi - v_lo + 1e-12)
    
    area = trapz_integral_over_quantiles(p, v, p0)
    tail_mean = area / max(1e-12, 1 - p0)
    pos_part = area - s * (1 - p0)
    
    return float(tail_mean), max(0.0, float(pos_part))


def impute_demand_for_stockout(
    Q: pd.Series, 
    stock: float, 
    method: str = "tail_mean"
) -> float:
    """
    Impute true demand when observed sales = stock (likely stock-out).
    
    Uses the full quantile function Q(p) to compute E[D | D >= stock].
    
    Args:
        Q: Quantiles indexed by float levels (e.g., 0.01 to 0.99)
        stock: Stock level (observed sales ceiling)
        method: Imputation method (currently only "tail_mean")
        
    Returns:
        Imputed demand
    """
    p = np.array(sorted(Q.index.values), dtype=float)
    v = Q.loc[p].values.astype(float)
    
    tail_mean, _ = expected_tail(p, v, stock)
    
    return float(tail_mean)


def detect_stockout_flags(
    state_start: pd.DataFrame,
    received: pd.Series,
    sales: pd.Series,
    state_end: pd.Series,
    epsilon: float = 1e-6
) -> pd.Series:
    """
    Detect likely stock-outs via heuristic.
    
    Flags SKUs where:
    - End inventory is ~0
    - Sales >= available supply
    
    Args:
        state_start: Starting on_hand
        received: Goods received this week
        sales: Sales this week
        state_end: Ending on_hand
        epsilon: Tolerance for zero
        
    Returns:
        Boolean Series indicating likely stock-outs
    """
    supply = state_start["on_hand"] + received
    
    return (state_end <= epsilon) & (sales >= supply - epsilon)

