"""
Analytical newsvendor ordering policy from Szabłowski (arXiv:2601.18919v1).

Order quantity: Q = max(B(i,t+3) − Ĩ(i,t+3), 0)
where B(i,t+3) = D̂(i,t+3) + zq * φ * √D̂(i,t+3)

Lead time semantics (L=3 transit slots):
  - Order placed at END of week t arrives at START of week t+3.
  - We must project inventory forward through weeks t+1 and t+2 using point
    forecasts before computing the target stock level for week t+3.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PolicyParams:
    """Parameters for the analytical newsvendor policy."""
    cs: float = 1.0   # shortage cost per unit
    ch: float = 0.2   # holding cost per unit per week
    phi: float = 1.0   # uncertainty scaling parameter

    @property
    def critical_fractile(self) -> float:
        return self.cs / (self.cs + self.ch)

    @property
    def z_q(self) -> float:
        return float(norm.ppf(self.critical_fractile))


def compute_order(
    d_hat: Tuple[float, float, float],
    on_hand: int,
    in_transit: Tuple[int, int, int],
    params: PolicyParams,
) -> int:
    """Compute order quantity using the analytical newsvendor policy.

    Parameters
    ----------
    d_hat : (d_h1, d_h2, d_h3)
        Point forecasts for horizons 1, 2, 3 (already rounded/clipped).
    on_hand : int
        Current on-hand inventory.
    in_transit : (Q arriving week t+1, Q arriving week t+2, Q arriving week t+3)
        In-transit pipeline.  in_transit[2] is the order placed last week.
    params : PolicyParams

    Returns
    -------
    Q : int
        Non-negative integer order quantity.
    """
    d1, d2, d3 = d_hat
    q1, q2, q3 = in_transit

    # Project inventory forward through the transit period (lost-sales assumption)
    # Week t+1: stock = on_hand + q1 arriving; demand d1
    stock_t1 = on_hand + q1
    leftover_t1 = max(stock_t1 - d1, 0)

    # Week t+2: stock = leftover_t1 + q2 arriving; demand d2
    stock_t2 = leftover_t1 + q2
    leftover_t2 = max(stock_t2 - d2, 0)

    # Week t+3: before our new order, stock = leftover_t2 + q3 (last week's order)
    projected_inventory = leftover_t2 + q3

    # Target stock level for week t+3
    sigma_h3 = params.phi * np.sqrt(max(d3, 0.0))
    target_stock = d3 + params.z_q * sigma_h3

    order = max(int(np.round(target_stock - projected_inventory)), 0)
    return order


def compute_orders_for_all_skus(
    forecasts: Dict[Tuple[int, int], Tuple[float, float, float]],
    states: Dict[Tuple[int, int], dict],
    params: PolicyParams,
) -> Dict[Tuple[int, int], int]:
    """Compute orders for all SKUs.

    Parameters
    ----------
    forecasts : {(store, product): (d_h1, d_h2, d_h3)}
    states : {(store, product): {"on_hand": int, "in_transit": [q1, q2, q3]}}
    params : PolicyParams

    Returns
    -------
    orders : {(store, product): Q}
    """
    orders = {}
    for key, d_hat in forecasts.items():
        state = states.get(key)
        if state is None:
            orders[key] = 0
            continue
        orders[key] = compute_order(
            d_hat,
            state["on_hand"],
            tuple(state["in_transit"]),
            params,
        )
    return orders
