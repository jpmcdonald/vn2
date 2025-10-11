"""Base-stock (order-up-to) inventory policy"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def base_stock_orders(
    mu: pd.Series, 
    sigma: pd.Series, 
    state: pd.DataFrame,
    lead_weeks: int = 2, 
    review_weeks: int = 1, 
    service: float = 0.8333
) -> pd.Series:
    """
    Calculate order quantities using base-stock (order-up-to) policy.
    
    S = μ×(L+R) + z×σ×√(L+R)
    Order = max(0, S - inventory_position)
    
    where inventory_position = on_hand + intransit_1 + intransit_2
    
    Args:
        mu: Mean weekly demand forecast per SKU
        sigma: Standard deviation of weekly demand per SKU
        state: Current inventory state [on_hand, intransit_1, intransit_2]
        lead_weeks: Lead time in weeks
        review_weeks: Review period in weeks
        service: Service level (critical fractile)
        
    Returns:
        Order quantities per SKU
    """
    L = lead_weeks + review_weeks
    z = norm.ppf(service)
    
    # Base stock level
    S = mu * L + z * (np.sqrt(L) * sigma)
    
    # Inventory position
    position = state["on_hand"] + state["intransit_1"] + state["intransit_2"]
    
    # Order quantity
    orders = (S - position).clip(lower=0).round().astype(int)
    
    return orders

