"""
Entropy and outcome-PMF metrics for demand and newsvendor cost distributions.

Conventions (see docs/ENTROPY_REGIME_FRAMEWORK_VN2.md):
- Shannon entropy in nats (natural log).
- entropy_gap_gaussian: differential entropy of N(mu, sigma^2) minus discrete Shannon H(p)
  on the demand PMF support (same mu, sigma from p).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Union

import numpy as np

from .sip_opt import Costs, convolve_inventory, optimize_order

Array = np.ndarray


def normalize_pmf(pmf: Array, eps: float = 1e-15) -> Array:
    p = np.asarray(pmf, dtype=float).copy()
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s <= eps:
        out = np.zeros_like(p)
        out[0] = 1.0
        return out
    return p / s


def shannon_entropy(pmf: Array, base: float = math.e) -> float:
    """Shannon entropy of a discrete distribution (nats if base=e)."""
    p = normalize_pmf(pmf)
    mask = p > 1e-20
    return float(-np.sum(p[mask] * np.log(p[mask]) / np.log(base)))


def mean_variance_discrete(pmf: Array) -> Tuple[float, float]:
    p = normalize_pmf(pmf)
    support = np.arange(len(pmf), dtype=float)
    mu = float(np.sum(support * p))
    var = float(np.sum(p * (support - mu) ** 2))
    return mu, max(var, 0.0)


def gaussian_differential_entropy(mu: float, variance: float) -> float:
    """Differential entropy of N(mu, sigma^2) in nats. Undefined if variance <= 0."""
    if variance <= 1e-20:
        return float("-inf")
    return 0.5 * math.log(2 * math.pi * math.e * variance)


def entropy_gap_gaussian(pmf: Array) -> float:
    """
    H_gaussian_diff - H_discrete (both in nats).

    Larger values indicate the discrete PMF is more concentrated than a
    variance-matched Gaussian (for unimodal-ish cases).
    """
    mu, var = mean_variance_discrete(pmf)
    h_disc = shannon_entropy(pmf)
    h_g = gaussian_differential_entropy(mu, var)
    if not math.isfinite(h_g):
        return float("nan")
    return h_g - h_disc


def single_period_cost(total_inventory: int, demand: int, costs: Costs) -> float:
    overage = max(0, total_inventory - demand)
    underage = max(0, demand - total_inventory)
    return costs.holding * overage + costs.shortage * underage


def outcome_pmf_univariate_demand(
    demand_pmf: Array,
    total_inventory: int,
    costs: Costs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map demand PMF to outcome PMF on cost values (may have duplicate costs).

    Returns:
        cost_values: length n_support
        probs: same length (one atom per demand outcome)
    """
    p = normalize_pmf(demand_pmf)
    d_sup = np.arange(len(p), dtype=int)
    costs_v = np.array([single_period_cost(total_inventory, int(d), costs) for d in d_sup])
    return costs_v.astype(float), p.copy()


def collapse_duplicate_outcomes(values: Array, probs: Array) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate probabilities for identical outcome values (e.g. cost)."""
    v = np.asarray(values, dtype=float)
    p = normalize_pmf(probs)
    order = np.argsort(v)
    v_s = v[order]
    p_s = p[order]
    out_v: List[float] = []
    out_p: List[float] = []
    i = 0
    n = len(v_s)
    while i < n:
        j = i
        acc = 0.0
        while j < n and abs(v_s[j] - v_s[i]) < 1e-12:
            acc += p_s[j]
            j += 1
        out_v.append(float(v_s[i]))
        out_p.append(acc)
        i = j
    return np.array(out_v), normalize_pmf(np.array(out_p))


def pmf_I2_pre_from_I1_end(pmf_I1_end: Array, Q2: int) -> np.ndarray:
    """Same shift + cap as optimize_order in sip_opt."""
    grain = len(pmf_I1_end) - 1
    pmf_I2_pre = np.zeros(grain + 1)
    for i1 in range(grain + 1):
        if pmf_I1_end[i1] > 0:
            i2_pre = min(i1 + Q2, grain)
            pmf_I2_pre[i2_pre] += pmf_I1_end[i1]
    s = pmf_I2_pre.sum()
    if s > 0:
        pmf_I2_pre /= s
    return pmf_I2_pre


def joint_week2_outcome_cost_pmf(
    pmf_I1_end: Array,
    Q2: int,
    pmf_D2: Array,
    order_qty: int,
    costs: Costs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full joint over (I2_pre, D2) implied PMFs → PMF on single-period week-2 cost.

    Inventory before demand is I2_pre + order_qty; demand is D2.
    """
    pmf_I2 = pmf_I2_pre_from_I1_end(pmf_I1_end, Q2)
    grain = len(pmf_D2) - 1
    vals: List[float] = []
    probs: List[float] = []
    for i_pre in range(grain + 1):
        if pmf_I2[i_pre] <= 0:
            continue
        inv_before = i_pre + order_qty
        for d in range(grain + 1):
            if pmf_D2[d] <= 0:
                continue
            pr = pmf_I2[i_pre] * pmf_D2[d]
            c = single_period_cost(inv_before, d, costs)
            vals.append(c)
            probs.append(pr)
    if not vals:
        return np.array([0.0]), np.array([1.0])
    return collapse_duplicate_outcomes(np.array(vals), np.array(probs))


def expected_cost_joint_week2(
    pmf_I1_end: Array,
    Q2: int,
    pmf_D2: Array,
    order_qty: int,
    costs: Costs,
) -> float:
    v, p = joint_week2_outcome_cost_pmf(pmf_I1_end, Q2, pmf_D2, order_qty, costs)
    return float(np.sum(v * p))


def jensen_gap_joint_week2(
    pmf_I1_end: Array,
    Q2: int,
    pmf_D2: Array,
    order_qty: int,
    costs: Costs,
) -> float:
    """
    E_{I,D}[cost] - cost(round(E[I2_pre])+order_qty, round(E[D]), ...).

    Uses integer-rounded mean inventory position before demand and mean demand.
    """
    pmf_I2 = pmf_I2_pre_from_I1_end(pmf_I1_end, Q2)
    mu_i, _ = mean_variance_discrete(pmf_I2)
    mu_d, _ = mean_variance_discrete(pmf_D2)
    inv_point = int(round(mu_i + order_qty))
    d_point = int(round(mu_d))
    inv_point = max(0, min(inv_point, len(pmf_D2) - 1 + order_qty))
    d_point = max(0, min(d_point, len(pmf_D2) - 1))
    e_joint = expected_cost_joint_week2(pmf_I1_end, Q2, pmf_D2, order_qty, costs)
    c_point = single_period_cost(inv_point, d_point, costs)
    return float(e_joint - c_point)


def shannon_entropy_outcome_pmf(
    pmf_I1_end: Array,
    Q2: int,
    pmf_D2: Array,
    order_qty: int,
    costs: Costs,
) -> float:
    v, p = joint_week2_outcome_cost_pmf(pmf_I1_end, Q2, pmf_D2, order_qty, costs)
    return shannon_entropy(p)


def demand_metrics_week2(
    pmf_D2: Array,
) -> Dict[str, float]:
    return {
        "H_demand": shannon_entropy(pmf_D2),
        "entropy_gap_demand": entropy_gap_gaussian(pmf_D2),
        "mean_demand": mean_variance_discrete(pmf_D2)[0],
    }


def full_week2_entropy_metrics(
    I0: int,
    Q1: int,
    Q2: int,
    pmf_D1: Array,
    pmf_D2: Array,
    costs: Costs,
    max_Q: int = 1000,
) -> Dict[str, Union[float, int, None]]:
    """
    Week-2 SIP path: I1_end from D1, optimal Q*, joint outcome PMF on cost for week 2.

    Returns scalar metrics for logging / parquet rows.
    """
    pmf_I1_end = convolve_inventory(I0, Q1, pmf_D1)
    Q_opt, expected_cost = optimize_order(pmf_I1_end, Q2, pmf_D2, costs, max_Q=max_Q)
    dm = demand_metrics_week2(pmf_D2)
    H_out = shannon_entropy_outcome_pmf(pmf_I1_end, Q2, pmf_D2, Q_opt, costs)
    j_gap = jensen_gap_joint_week2(pmf_I1_end, Q2, pmf_D2, Q_opt, costs)
    return {
        "sip_order_qty": int(Q_opt),
        "sip_expected_cost_w2": float(expected_cost),
        "H_demand_h2": dm["H_demand"],
        "entropy_gap_demand_h2": dm["entropy_gap_demand"],
        "mean_demand_h2": dm["mean_demand"],
        "H_outcome_w2": H_out,
        "jensen_gap_w2": j_gap,
    }


def empirical_demand_pmf_from_counts(
    demands: Array,
    grain: int,
) -> np.ndarray:
    """
    Histogram of nonnegative integer demands onto {0,...,grain}.

    For H12: empirical PMF from a history vector (no lookahead if caller slices train only).
    """
    pmf = np.zeros(grain + 1, dtype=float)
    for y in np.asarray(demands).ravel():
        yi = int(round(float(y)))
        if yi < 0:
            yi = 0
        if yi > grain:
            yi = grain
        pmf[yi] += 1.0
    return normalize_pmf(pmf)


def entropy_gap_model_vs_empirical(
    model_pmf: Array,
    empirical_pmf: Array,
) -> float:
    """
    Mean cross-entropy style gap: sum p_emp * log(p_emp / p_model) = KL(emp || model)
    in nats, with epsilon smoothing on model.

    Lower is better match of model to empirical histogram.
    """
    pe = normalize_pmf(empirical_pmf)
    pm = normalize_pmf(model_pmf)
    eps = 1e-12
    pm = np.clip(pm, eps, None)
    pm = pm / pm.sum()
    mask = pe > 1e-20
    return float(np.sum(pe[mask] * np.log(pe[mask] / pm[mask])))


def sensitivity_ratio(
    H_demand_prev: float,
    H_demand_curr: float,
    H_outcome_prev: float,
    H_outcome_curr: float,
    eps: float = 1e-8,
) -> float:
    """Delta H_outcome / Delta H_demand; nan if denominator ~ 0."""
    dd = H_demand_curr - H_demand_prev
    do = H_outcome_curr - H_outcome_prev
    if abs(dd) < eps:
        return float("nan")
    return float(do / dd)
