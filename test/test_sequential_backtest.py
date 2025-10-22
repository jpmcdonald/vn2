"""
Unit tests for sequential backtest with L=2 lead time.

Tests PMF operations, inventory propagation, newsvendor optimization, and full backtest.
"""

import numpy as np
import pytest
from pathlib import Path

from vn2.analyze.sequential_planner import (
    Costs,
    _safe_pmf,
    _shift_right,
    leftover_from_stock_and_demand,
    diff_pmf_D_minus_L,
    expected_pos_neg_from_Z,
    choose_order_L2,
    _conv_fft
)
from vn2.analyze.sequential_backtest import (
    BacktestState,
    run_12week_backtest,
    quantiles_to_pmf
)


class TestPMFOperations:
    """Test PMF utility functions."""
    
    def test_safe_pmf_normalization(self):
        """Test PMF normalization."""
        p = np.array([0.1, 0.2, 0.3, 0.4])
        p_norm = _safe_pmf(p)
        assert np.isclose(p_norm.sum(), 1.0)
        assert np.all(p_norm >= 0)
    
    def test_safe_pmf_with_negatives(self):
        """Test PMF with negative values."""
        p = np.array([0.1, -0.1, 0.3, 0.4])
        p_norm = _safe_pmf(p)
        assert np.isclose(p_norm.sum(), 1.0)
        assert np.all(p_norm >= 0)
    
    def test_shift_right(self):
        """Test PMF right shift."""
        p = np.array([0.5, 0.3, 0.2])
        p_shifted = _shift_right(p, 2)
        assert len(p_shifted) == 5
        assert p_shifted[0] == 0.0
        assert p_shifted[1] == 0.0
        assert p_shifted[2] == 0.5
        assert p_shifted[3] == 0.3
        assert p_shifted[4] == 0.2
    
    def test_conv_fft_matches_direct(self):
        """Test FFT convolution matches direct convolution."""
        a = np.array([0.3, 0.5, 0.2])
        b = np.array([0.4, 0.6])
        
        # FFT convolution
        c_fft = _conv_fft(a, b)
        
        # Direct convolution
        c_direct = np.convolve(a, b)
        c_direct = c_direct / c_direct.sum()
        
        assert np.allclose(c_fft, c_direct, atol=1e-10)


class TestInventoryPropagation:
    """Test inventory leftover calculations."""
    
    def test_leftover_deterministic(self):
        """Test leftover with deterministic demand."""
        # Stock = 10, Demand = 5 with certainty
        S = 10
        D_pmf = np.zeros(20)
        D_pmf[5] = 1.0
        
        L_pmf = leftover_from_stock_and_demand(S, D_pmf)
        
        # Leftover should be 5 with certainty
        assert len(L_pmf) == S + 1
        assert L_pmf[5] == 1.0
        assert np.isclose(L_pmf.sum(), 1.0)
    
    def test_leftover_stockout(self):
        """Test leftover when demand exceeds stock."""
        # Stock = 5, Demand = 10 with certainty
        S = 5
        D_pmf = np.zeros(20)
        D_pmf[10] = 1.0
        
        L_pmf = leftover_from_stock_and_demand(S, D_pmf)
        
        # Leftover should be 0 with certainty
        assert L_pmf[0] == 1.0
    
    def test_leftover_uniform_demand(self):
        """Test leftover with uniform demand."""
        # Stock = 10, Demand uniform on {0, 1, ..., 19}
        S = 10
        D_pmf = np.ones(20) / 20.0
        
        L_pmf = leftover_from_stock_and_demand(S, D_pmf)
        
        # Check properties
        assert len(L_pmf) == S + 1
        assert np.isclose(L_pmf.sum(), 1.0)
        # L[0] should have mass from demands >= 10 (10 values: 10..19)
        # When D=10, leftover = 10-10 = 0
        # When D>10, leftover = 0
        assert np.isclose(L_pmf[0], 10.0 / 20.0)


class TestNewsvendorOptimization:
    """Test newsvendor order selection."""
    
    def test_choose_order_deterministic(self):
        """Test order choice with deterministic demand."""
        # I0=0, Q1=0, Q2=0, D1=0, D2=10 (deterministic)
        h1 = np.zeros(20)
        h1[0] = 1.0
        h2 = np.zeros(20)
        h2[10] = 1.0
        
        costs = Costs(holding=0.2, shortage=1.0)
        
        q_star, cost_star = choose_order_L2(h1, h2, 0, 0, 0, costs)
        
        # Optimal order should be 10 (to exactly meet demand)
        assert q_star == 10
        assert np.isclose(cost_star, 0.0)
    
    def test_choose_order_newsvendor_fractile(self):
        """Test order choice respects newsvendor fractile."""
        # Uniform demand on {0, 1, ..., 99}
        h1 = np.zeros(100)
        h1[0] = 1.0  # No demand in period 1
        h2 = np.ones(100) / 100.0
        
        costs = Costs(holding=0.2, shortage=1.0)
        # Critical fractile = 1.0 / (1.0 + 0.2) = 0.8333
        
        q_star, cost_star = choose_order_L2(h1, h2, 0, 0, 0, costs)
        
        # Optimal should be around 83rd percentile
        assert 80 <= q_star <= 86
    
    def test_choose_order_with_initial_inventory(self):
        """Test order choice with existing inventory."""
        # I0=50, Q1=0, Q2=0, D1=0, D2=60 (deterministic)
        h1 = np.zeros(100)
        h1[0] = 1.0
        h2 = np.zeros(100)
        h2[60] = 1.0
        
        costs = Costs(holding=0.2, shortage=1.0)
        
        q_star, cost_star = choose_order_L2(h1, h2, 50, 0, 0, costs)
        
        # Optimal order should be 10 (50 + 10 = 60)
        assert q_star == 10


class TestBacktestEngine:
    """Test full 12-week backtest."""
    
    def test_backtest_deterministic(self):
        """Test backtest with deterministic demands."""
        # Setup: constant demand of 10 per week
        demands = [10] * 12
        
        # Forecasts: perfect foresight
        h1_pmfs = []
        h2_pmfs = []
        for _ in range(12):
            h1 = np.zeros(50)
            h1[10] = 1.0
            h2 = np.zeros(50)
            h2[10] = 1.0
            h1_pmfs.append(h1)
            h2_pmfs.append(h2)
        
        initial_state = BacktestState(week=1, on_hand=0, intransit_1=0, intransit_2=0)
        costs = Costs(holding=0.2, shortage=1.0)
        
        result = run_12week_backtest(
            store=0, product=1, model_name="test",
            forecasts_h1=h1_pmfs,
            forecasts_h2=h2_pmfs,
            actuals=demands,
            initial_state=initial_state,
            costs=costs
        )
        
        # Check basic properties
        assert result.n_weeks == 12
        assert len(result.weeks) == 12
        assert result.n_missing_forecasts == 0
        
        # With perfect foresight and L=2, should have low cost
        # (some cost in early weeks due to cold start)
        assert result.total_realized_cost >= 0
    
    def test_backtest_with_missing_forecasts(self):
        """Test backtest with missing forecasts."""
        demands = [10] * 12
        
        # Half forecasts missing
        h1_pmfs = [None if i % 2 == 0 else np.ones(50)/50 for i in range(12)]
        h2_pmfs = [None if i % 2 == 0 else np.ones(50)/50 for i in range(12)]
        
        initial_state = BacktestState(week=1, on_hand=0, intransit_1=0, intransit_2=0)
        costs = Costs(holding=0.2, shortage=1.0)
        
        result = run_12week_backtest(
            store=0, product=1, model_name="test",
            forecasts_h1=h1_pmfs,
            forecasts_h2=h2_pmfs,
            actuals=demands,
            initial_state=initial_state,
            costs=costs
        )
        
        # Should handle missing forecasts gracefully
        assert result.n_missing_forecasts == 6
        assert result.total_realized_cost > 0
    
    def test_backtest_weeks_11_12_no_orders(self):
        """Test that weeks 11-12 place no orders."""
        demands = [10] * 12
        h1_pmfs = [np.ones(50)/50 for _ in range(12)]
        h2_pmfs = [np.ones(50)/50 for _ in range(12)]
        
        initial_state = BacktestState(week=1, on_hand=0, intransit_1=0, intransit_2=0)
        costs = Costs(holding=0.2, shortage=1.0)
        
        result = run_12week_backtest(
            store=0, product=1, model_name="test",
            forecasts_h1=h1_pmfs,
            forecasts_h2=h2_pmfs,
            actuals=demands,
            initial_state=initial_state,
            costs=costs
        )
        
        # Weeks 11 and 12 should have q=0
        assert result.weeks[10].order_placed == 0  # Week 11
        assert result.weeks[11].order_placed == 0  # Week 12
    
    def test_backtest_cost_components(self):
        """Test that cost components are tracked correctly."""
        # High demand followed by low demand
        demands = [20, 20, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        
        h1_pmfs = []
        h2_pmfs = []
        for d in demands:
            h1 = np.zeros(50)
            h1[d] = 1.0
            h2 = np.zeros(50)
            h2[d] = 1.0
            h1_pmfs.append(h1)
            h2_pmfs.append(h2)
        
        initial_state = BacktestState(week=1, on_hand=0, intransit_1=0, intransit_2=0)
        costs = Costs(holding=0.2, shortage=1.0)
        
        result = run_12week_backtest(
            store=0, product=1, model_name="test",
            forecasts_h1=h1_pmfs,
            forecasts_h2=h2_pmfs,
            actuals=demands,
            initial_state=initial_state,
            costs=costs
        )
        
        # Check that both totals are computed
        assert result.total_realized_cost > 0
        assert result.total_expected_cost > 0
        assert result.total_realized_cost_excl_w1 >= 0
        assert result.total_expected_cost_excl_w1 >= 0
        
        # Excluding week 1 should be less than or equal to total
        assert result.total_realized_cost_excl_w1 <= result.total_realized_cost


class TestQuantileToPMF:
    """Test quantile to PMF conversion."""
    
    def test_quantiles_to_pmf_basic(self):
        """Test basic quantile to PMF conversion."""
        quantiles = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
        quantile_levels = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
        
        pmf = quantiles_to_pmf(quantiles, quantile_levels, grain=200)
        
        # Check properties
        assert len(pmf) == 200
        assert np.isclose(pmf.sum(), 1.0)
        assert np.all(pmf >= 0)
    
    def test_quantiles_to_pmf_degenerate(self):
        """Test quantile to PMF with constant quantiles."""
        quantiles = np.array([10] * 13)
        quantile_levels = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
        
        pmf = quantiles_to_pmf(quantiles, quantile_levels, grain=50)
        
        # Should have all mass near 10
        assert np.isclose(pmf.sum(), 1.0)
        assert pmf[10] > 0.5  # Most mass at 10


class TestLeadTimeSemantics:
    """Test L=2 lead time semantics."""
    
    def test_order_arrives_at_t_plus_2(self):
        """Test that order placed at t arrives at t+2."""
        # Place order of 30 at week 1, demand = 0 everywhere
        demands = [0] * 12
        
        h1_pmfs = [np.array([1.0] + [0.0]*49) for _ in range(12)]
        h2_pmfs = [np.array([1.0] + [0.0]*49) for _ in range(12)]
        
        # Force a specific order in week 1 by setting high demand forecast
        h2_pmfs[0] = np.zeros(50)
        h2_pmfs[0][30] = 1.0  # Fake demand of 30 to trigger order
        
        initial_state = BacktestState(week=1, on_hand=0, intransit_1=0, intransit_2=0)
        costs = Costs(holding=0.2, shortage=1.0)
        
        result = run_12week_backtest(
            store=0, product=1, model_name="test",
            forecasts_h1=h1_pmfs,
            forecasts_h2=h2_pmfs,
            actuals=demands,
            initial_state=initial_state,
            costs=costs
        )
        
        # Order placed in week 1 should arrive at week 3
        # Check state progression
        week1_order = result.weeks[0].order_placed
        if week1_order > 0:
            # Week 3 should have this order available (either in intransit_1 or already on_hand)
            week3_state = result.weeks[2].state_before
            # The order should be in intransit_2 at week 2, intransit_1 at week 3
            # Since demand is 0, it should accumulate in on_hand
            assert week3_state.intransit_1 == week1_order or week3_state.on_hand >= week1_order


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

