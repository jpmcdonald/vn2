"""
Sanity tests for sequential L=2 planner.

Tests basic PMF operations, cost calculations, and sequential planning logic.
"""

import numpy as np
import pytest
from vn2.analyze.sequential_planner import (
    _safe_pmf, _conv_fft, _shift_right,
    leftover_from_stock_and_demand, diff_pmf_D_minus_L,
    pmf_quantile, expected_pos_neg_from_Z,
    choose_order_L2, run_sequential_L2, Costs
)


class TestPMFUtilities:
    """Test PMF utility functions."""
    
    def test_safe_pmf_normalizes(self):
        """Test that _safe_pmf normalizes correctly."""
        p = np.array([1, 2, 3, 4])
        p_norm = _safe_pmf(p)
        assert np.isclose(p_norm.sum(), 1.0)
        assert np.all(p_norm >= 0)
    
    def test_safe_pmf_handles_negatives(self):
        """Test that _safe_pmf removes negatives."""
        p = np.array([-1, 2, 3])
        p_norm = _safe_pmf(p)
        assert np.all(p_norm >= 0)
        assert np.isclose(p_norm.sum(), 1.0)
    
    def test_shift_right(self):
        """Test PMF shifting."""
        pmf = np.array([0.5, 0.5])
        shifted = _shift_right(pmf, 2)
        assert len(shifted) == 4
        assert np.isclose(shifted[2], 0.5)
        assert np.isclose(shifted[3], 0.5)
        assert np.isclose(shifted[0], 0.0)
    
    def test_shift_zero(self):
        """Test zero shift returns copy."""
        pmf = np.array([0.5, 0.5])
        shifted = _shift_right(pmf, 0)
        assert len(shifted) == len(pmf)
        assert np.allclose(shifted, pmf)


class TestLeftoverLogic:
    """Test leftover inventory calculation."""
    
    def test_deterministic_no_leftover(self):
        """Test when demand exactly meets stock."""
        # Stock = 10, demand is certain 10
        D_pmf = np.zeros(20)
        D_pmf[10] = 1.0
        
        L_pmf = leftover_from_stock_and_demand(10, D_pmf)
        
        # Should have leftover = 0 with probability 1
        assert np.isclose(L_pmf[0], 1.0)
    
    def test_deterministic_leftover(self):
        """Test when stock exceeds demand."""
        # Stock = 10, demand is certain 5
        D_pmf = np.zeros(20)
        D_pmf[5] = 1.0
        
        L_pmf = leftover_from_stock_and_demand(10, D_pmf)
        
        # Should have leftover = 5 with probability 1
        assert np.isclose(L_pmf[5], 1.0)
    
    def test_deterministic_stockout(self):
        """Test when demand exceeds stock."""
        # Stock = 10, demand is certain 15
        D_pmf = np.zeros(20)
        D_pmf[15] = 1.0
        
        L_pmf = leftover_from_stock_and_demand(10, D_pmf)
        
        # Should have leftover = 0 with probability 1
        assert np.isclose(L_pmf[0], 1.0)
    
    def test_uniform_demand(self):
        """Test with uniform demand."""
        # Stock = 5, demand uniform in {0, 1, 2, 3, 4}
        D_pmf = np.ones(5) / 5
        
        L_pmf = leftover_from_stock_and_demand(5, D_pmf)
        
        # Leftover should be uniform in {1, 2, 3, 4, 5}
        assert np.isclose(L_pmf.sum(), 1.0)
        assert len(L_pmf) == 6  # {0, 1, 2, 3, 4, 5}


class TestNewsvendorFractile:
    """Test newsvendor fractile optimization."""
    
    def test_deterministic_demand(self):
        """Test with deterministic demand."""
        # h1: demand = 5 (certain)
        h1 = np.zeros(20)
        h1[5] = 1.0
        
        # h2: demand = 10 (certain)
        h2 = np.zeros(20)
        h2[10] = 1.0
        
        costs = Costs(holding=0.2, shortage=1.0)  # cu/(cu+co) = 0.8333
        
        # I0=0, Q1=5, Q2=0 → after week 1, leftover = 0
        # Need to order 10 for week 2
        q, cost = choose_order_L2(h1, h2, I0=0, Q1=5, Q2=0, costs=costs)
        
        # Should order 10 (or close to it)
        assert q >= 8 and q <= 12
    
    def test_zero_demand(self):
        """Test when demand is zero."""
        # No demand
        h1 = np.zeros(20)
        h1[0] = 1.0
        h2 = np.zeros(20)
        h2[0] = 1.0
        
        costs = Costs(holding=0.2, shortage=1.0)
        
        # Should order 0
        q, cost = choose_order_L2(h1, h2, I0=0, Q1=0, Q2=0, costs=costs)
        
        assert q == 0
        assert np.isclose(cost, 0.0)
    
    def test_high_inventory(self):
        """Test when starting inventory is high."""
        # Low demand
        h1 = np.zeros(20)
        h1[1] = 1.0
        h2 = np.zeros(20)
        h2[1] = 1.0
        
        costs = Costs(holding=0.2, shortage=1.0)
        
        # High starting inventory
        q, cost = choose_order_L2(h1, h2, I0=100, Q1=0, Q2=0, costs=costs)
        
        # Should order 0 (we have plenty)
        assert q == 0


class TestSequentialRunner:
    """Test sequential planning over multiple epochs."""
    
    def test_deterministic_scenario(self):
        """Test with fully deterministic forecasts and actuals."""
        H = 3
        
        # Forecasts: demand is always 5
        h1_list = []
        h2_list = []
        for _ in range(H):
            pmf = np.zeros(20)
            pmf[5] = 1.0
            h1_list.append(pmf)
            h2_list.append(pmf.copy())
        
        # Actuals: actual demand is always 5
        actuals = [5] * (H + 2)
        
        costs = Costs(holding=0.2, shortage=1.0)
        
        result = run_sequential_L2(
            h1_list, h2_list, actuals,
            I0=0, Q1=5, Q2=5, costs=costs
        )
        
        # Should have minimal cost (forecasts match actuals)
        assert result.total_cost >= 0
        assert result.coverage == 1.0
        assert result.n_missing == 0
        assert len(result.orders_by_epoch) == H
    
    def test_missing_forecasts(self):
        """Test handling of missing forecasts."""
        H = 3
        
        # First epoch has forecasts, rest are None
        h1_list = [np.array([0.5, 0.5])] + [None] * (H - 1)
        h2_list = [np.array([0.5, 0.5])] + [None] * (H - 1)
        
        actuals = [2] * (H + 2)
        
        costs = Costs(holding=0.2, shortage=1.0)
        
        result = run_sequential_L2(
            h1_list, h2_list, actuals,
            I0=5, Q1=0, Q2=0, costs=costs
        )
        
        # Should flag missing forecasts
        assert result.n_missing == H - 1
        assert result.coverage < 1.0
        assert len(result.orders_by_epoch) == H
    
    def test_zero_orders_zero_demand(self):
        """Test that zero demand leads to zero orders and zero cost."""
        H = 2
        
        # Zero demand
        h1_list = []
        h2_list = []
        for _ in range(H):
            pmf = np.zeros(10)
            pmf[0] = 1.0
            h1_list.append(pmf)
            h2_list.append(pmf.copy())
        
        actuals = [0] * (H + 2)
        
        costs = Costs(holding=0.2, shortage=1.0)
        
        result = run_sequential_L2(
            h1_list, h2_list, actuals,
            I0=0, Q1=0, Q2=0, costs=costs
        )
        
        # Zero demand → zero orders → zero cost
        assert all(q == 0 for q in result.orders_by_epoch)
        assert np.isclose(result.total_cost, 0.0)


class TestCostCalculations:
    """Test cost calculation logic."""
    
    def test_holding_cost_only(self):
        """Test pure holding cost (no shortage)."""
        H = 1
        
        # Low demand
        h1 = np.zeros(20)
        h1[1] = 1.0
        h2 = np.zeros(20)
        h2[1] = 1.0
        
        actuals = [1, 1, 1]
        
        costs = Costs(holding=1.0, shortage=0.0)
        
        result = run_sequential_L2(
            [h1], [h2], actuals,
            I0=10, Q1=10, Q2=0, costs=costs
        )
        
        # Should have holding cost (excess inventory)
        assert result.total_cost > 0
    
    def test_shortage_cost_only(self):
        """Test pure shortage cost (no holding)."""
        H = 1
        
        # High demand
        h1 = np.zeros(20)
        h1[15] = 1.0
        h2 = np.zeros(20)
        h2[15] = 1.0
        
        actuals = [15, 15, 15]
        
        costs = Costs(holding=0.0, shortage=1.0)
        
        result = run_sequential_L2(
            [h1], [h2], actuals,
            I0=0, Q1=0, Q2=0, costs=costs
        )
        
        # Should have shortage cost (not enough inventory)
        assert result.total_cost > 0


def test_pmf_quantile():
    """Test PMF quantile calculation."""
    # Uniform PMF on {0, 1, 2, 3, 4}
    pmf = np.ones(5) / 5
    offset = 0
    
    # Median should be around 2
    q50 = pmf_quantile(pmf, offset, 0.5)
    assert q50 >= 1 and q50 <= 3
    
    # 0.9 quantile should be 3 or 4
    q90 = pmf_quantile(pmf, offset, 0.9)
    assert q90 >= 3


def test_expected_pos_neg():
    """Test expected overage/underage calculation."""
    # Z = {-2, -1, 0, 1, 2} uniform
    z_pmf = np.ones(5) / 5
    z_min = -2
    
    E_over, E_under = expected_pos_neg_from_Z(z_pmf, z_min)
    
    # E[max(Z, 0)] = 0.2*1 + 0.2*2 = 0.6
    # E[max(-Z, 0)] = 0.2*2 + 0.2*1 = 0.6
    assert np.isclose(E_over, 0.6, atol=0.01)
    assert np.isclose(E_under, 0.6, atol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

