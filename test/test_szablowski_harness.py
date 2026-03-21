"""System tests for szablowski.harness — PolicyAdapter and simulation engine."""

import numpy as np
import pandas as pd
import pytest

from szablowski.harness import (
    AnalyticalPolicy,
    CostParams,
    PolicyAdapter,
    SKUState,
    simulate_week_l3,
)
from szablowski.policy import PolicyParams


# ---------------------------------------------------------------------------
# Mock policy for testing
# ---------------------------------------------------------------------------

class MockPolicy(PolicyAdapter):
    """Always orders a fixed amount."""

    def __init__(self, fixed_order: int = 10, label: str = "mock"):
        self._fixed_order = fixed_order
        self._label = label

    def generate_order(self, state, order_number, fold_idx, costs):
        return self._fixed_order

    def name(self):
        return self._label


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSKUState:
    def test_copy_is_independent(self):
        s = SKUState(0, 100, 50, [10, 20, 0])
        s2 = s.copy()
        s2.on_hand = 0
        s2.in_transit[0] = 999
        assert s.on_hand == 50
        assert s.in_transit[0] == 10


class TestAnalyticalPolicy:
    def test_generates_nonneg_order(self):
        forecasts = {(0, 100, 1): (10.0, 10.0, 10.0)}
        policy = AnalyticalPolicy(forecasts, phi=1.0)
        state = SKUState(0, 100, 20, [5, 5, 0])
        costs = CostParams()
        order = policy.generate_order(state, order_number=1, fold_idx=0, costs=costs)
        assert order >= 0
        assert isinstance(order, int)

    def test_missing_forecast_returns_zero(self):
        policy = AnalyticalPolicy({}, phi=1.0)
        state = SKUState(0, 999, 0, [0, 0, 0])
        order = policy.generate_order(state, order_number=1, fold_idx=0, costs=CostParams())
        assert order == 0


class TestSimulateWeekL3:
    def test_transit_pipeline_shifts_correctly(self):
        """in_transit[0] arrives this week; pipeline shifts left after."""
        states = {
            (0, 100): SKUState(0, 100, on_hand=10, in_transit=[5, 15, 25]),
        }
        sales = {(0, 100): 3}
        costs = CostParams(holding=0.2, shortage=1.0)

        h, s, stockouts = simulate_week_l3(states, sales, costs, week_num=1)

        st = states[(0, 100)]
        # available = 10 + 5 (arriving) = 15; demand = 3 -> leftover = 12
        assert st.on_hand == 12
        # transit shifted: [15, 25, 0]
        assert st.in_transit == [15, 25, 0]

    def test_three_slot_transit_correctness(self):
        """Order placed at in_transit[2] arrives 3 weeks later, NOT 2."""
        states = {
            (0, 100): SKUState(0, 100, on_hand=0, in_transit=[0, 0, 100]),
        }
        sales_empty = {(0, 100): 0}
        costs = CostParams()

        # Week 1: in_transit[0]=0 arrives; state becomes on_hand=0, transit=[0, 100, 0]
        simulate_week_l3(states, sales_empty, costs, 1)
        assert states[(0, 100)].on_hand == 0
        assert states[(0, 100)].in_transit == [0, 100, 0]

        # Week 2: in_transit[0]=0 arrives; state becomes on_hand=0, transit=[100, 0, 0]
        simulate_week_l3(states, sales_empty, costs, 2)
        assert states[(0, 100)].on_hand == 0
        assert states[(0, 100)].in_transit == [100, 0, 0]

        # Week 3: in_transit[0]=100 arrives; state becomes on_hand=100, transit=[0, 0, 0]
        simulate_week_l3(states, sales_empty, costs, 3)
        assert states[(0, 100)].on_hand == 100
        assert states[(0, 100)].in_transit == [0, 0, 0]

    def test_shortage_computed_correctly(self):
        states = {(0, 100): SKUState(0, 100, on_hand=5, in_transit=[0, 0, 0])}
        sales = {(0, 100): 10}
        costs = CostParams(holding=0.2, shortage=1.0)

        records = []
        h, s, stockouts = simulate_week_l3(states, sales, costs, 1, records)

        assert s == pytest.approx(5.0)  # shortage = 10 - 5 = 5, cost = 5 * 1.0
        assert h == pytest.approx(0.0)
        assert stockouts == 1
        assert records[0]["shortage"] == 5

    def test_holding_computed_correctly(self):
        states = {(0, 100): SKUState(0, 100, on_hand=20, in_transit=[0, 0, 0])}
        sales = {(0, 100): 5}
        costs = CostParams(holding=0.2, shortage=1.0)

        h, s, stockouts = simulate_week_l3(states, sales, costs, 1)

        assert h == pytest.approx(3.0)  # leftover = 15, cost = 15 * 0.2
        assert s == pytest.approx(0.0)
        assert stockouts == 0


class TestRunComparisonWithMocks:
    def test_two_policies_same_shape(self, tmp_path):
        """Two mock policies on 2 SKUs, 2 weeks produce same-shape detail DataFrames."""
        # Create a minimal initial state CSV
        initial_csv = tmp_path / "initial.csv"
        pd.DataFrame({
            "Store": [0, 0],
            "Product": [100, 200],
            "End Inventory": [50, 50],
            "In Transit W+1": [10, 10],
            "In Transit W+2": [10, 10],
        }).to_csv(initial_csv, index=False)

        # Create minimal sales CSVs
        sales_dir = tmp_path / "sales"
        sales_dir.mkdir()
        for week_num in range(1, 3):
            dates = {
                1: "2024-04-15",
                2: "2024-04-22",
            }
            filename = {
                1: "Week 1 - 2024-04-15 - Sales.csv",
                2: "Week 2 - 2024-04-22 - Sales.csv",
            }
            pd.DataFrame({
                "Store": [0, 0],
                "Product": [100, 200],
                dates[week_num]: [10, 20],
            }).to_csv(sales_dir / filename[week_num], index=False)

        from szablowski.harness import load_initial_states, load_weekly_sales, run_comparison

        policy_a = MockPolicy(fixed_order=10, label="policy_a")
        policy_b = MockPolicy(fixed_order=20, label="policy_b")

        results = run_comparison(
            [policy_a, policy_b],
            initial_csv,
            sales_dir,
            max_weeks=2,
        )

        assert "policy_a" in results
        assert "policy_b" in results
        # 2 SKUs * 2 weeks = 4 records each
        assert len(results["policy_a"]) == 4
        assert len(results["policy_b"]) == 4
        assert set(results["policy_a"].columns) == set(results["policy_b"].columns)
