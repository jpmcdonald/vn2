"""Tests for simulation engine"""

import pytest
import pandas as pd
import numpy as np
from vn2.sim import Simulator, Costs, LeadTime


@pytest.fixture
def simple_state():
    """Simple 2-SKU state"""
    idx = pd.MultiIndex.from_tuples([(0, 100), (0, 101)], names=["Store", "Product"])
    state = pd.DataFrame({
        "on_hand": [10, 20],
        "intransit_1": [5, 0],
        "intransit_2": [0, 10]
    }, index=idx)
    return state


def test_pipeline_roll(simple_state):
    """Test that pipeline rolls correctly with 2-week lead time"""
    sim = Simulator(Costs(), LeadTime())
    
    demand = pd.Series([5, 5], index=simple_state.index)
    order = pd.Series([10, 15], index=simple_state.index)
    
    next_state, costs = sim.step(simple_state, demand, order)
    
    # Check on_hand: (10 + 5 - 5, 20 + 0 - 5)
    assert next_state.loc[(0, 100), "on_hand"] == 10
    assert next_state.loc[(0, 101), "on_hand"] == 15
    
    # Check pipeline roll
    assert next_state.loc[(0, 100), "intransit_1"] == 0  # was intransit_2
    assert next_state.loc[(0, 100), "intransit_2"] == 10  # new order
    
    assert next_state.loc[(0, 101), "intransit_1"] == 10  # was intransit_2
    assert next_state.loc[(0, 101), "intransit_2"] == 15  # new order


def test_shortage_cost(simple_state):
    """Test shortage cost when demand exceeds stock"""
    sim = Simulator(Costs(holding=0.2, shortage=1.0), LeadTime())
    
    # Demand exceeds available stock
    demand = pd.Series([20, 30], index=simple_state.index)
    order = pd.Series([0, 0], index=simple_state.index)
    
    next_state, costs = sim.step(simple_state, demand, order)
    
    # Available: (10+5=15, 20+0=20)
    # Lost: (20-15=5, 30-20=10)
    expected_shortage = (5 + 10) * 1.0
    
    assert costs["shortage"] == pytest.approx(expected_shortage)
    assert next_state.loc[(0, 100), "on_hand"] == 0
    assert next_state.loc[(0, 101), "on_hand"] == 0


def test_holding_cost(simple_state):
    """Test holding cost on end-of-week inventory"""
    sim = Simulator(Costs(holding=0.2, shortage=1.0), LeadTime())
    
    # Low demand, high inventory remaining
    demand = pd.Series([2, 3], index=simple_state.index)
    order = pd.Series([0, 0], index=simple_state.index)
    
    next_state, costs = sim.step(simple_state, demand, order)
    
    # On hand after: (10+5-2=13, 20+0-3=17)
    expected_holding = (13 + 17) * 0.2
    
    assert costs["holding"] == pytest.approx(expected_holding)
    assert costs["shortage"] == 0.0

