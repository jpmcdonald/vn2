import numpy as np

from vn2.analyze.entropy_metrics import (
    Costs,
    empirical_demand_pmf_from_counts,
    entropy_gap_gaussian,
    full_week2_entropy_metrics,
    mean_variance_discrete,
    shannon_entropy,
    joint_week2_outcome_cost_pmf,
    pmf_I2_pre_from_I1_end,
)
from vn2.analyze.sip_opt import quantiles_to_pmf as q2p


def test_normalize_and_entropy_uniform():
    p = np.ones(5) / 5
    assert abs(shannon_entropy(p) - np.log(5)) < 1e-10
    mu, v = mean_variance_discrete(p)
    assert abs(mu - 2.0) < 1e-10


def test_entropy_gap_deterministic():
    p = np.zeros(11)
    p[5] = 1.0
    g = entropy_gap_gaussian(p)
    assert g > 0


def test_joint_outcome_pmf_sums_to_one():
    costs = Costs(0.2, 1.0)
    pmf_D = np.zeros(21)
    pmf_D[10] = 0.5
    pmf_D[11] = 0.5
    pmf_I1 = np.zeros(21)
    pmf_I1[5] = 1.0
    pmf_I2 = pmf_I2_pre_from_I1_end(pmf_I1, Q2=0)
    v, p = joint_week2_outcome_cost_pmf(pmf_I1, 0, pmf_D, order_qty=3, costs=costs)
    assert abs(p.sum() - 1.0) < 1e-9


def test_full_week2_runs():
    costs = Costs(0.2, 1.0)
    levels = np.array([0.1, 0.5, 0.9])
    q1 = np.array([2.0, 5.0, 8.0])
    q2 = np.array([2.0, 5.0, 9.0])
    pmf1 = q2p(q1, levels, grain=30)
    pmf2 = q2p(q2, levels, grain=30)
    out = full_week2_entropy_metrics(3, 1, 1, pmf1, pmf2, costs, max_Q=50)
    assert "H_demand_h2" in out
    assert "H_outcome_w2" in out
    assert "jensen_gap_w2" in out
    assert np.isfinite(out["H_outcome_w2"])


def test_empirical_pmf():
    y = np.array([0, 0, 1, 2, 2, 2])
    p = empirical_demand_pmf_from_counts(y, grain=5)
    assert abs(p.sum() - 1.0) < 1e-9
    assert p[2] == 0.5
