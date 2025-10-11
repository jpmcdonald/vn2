"""Core simulation engine for inventory dynamics with lead time"""

from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd


@dataclass
class Costs:
    """Cost structure for inventory optimization"""
    holding: float = 0.2        # per unit per week
    shortage: float = 1.0        # per unit of lost sales


@dataclass
class LeadTime:
    """Lead time configuration"""
    lead_weeks: int = 2          # orders arrive at start of week W+3
    review_weeks: int = 1        # weekly review period


class Simulator:
    """
    Weekly discrete-event simulator with 2-week lead time.
    
    State columns: on_hand, intransit_1, intransit_2
    - intransit_1: arrives next week
    - intransit_2: arrives in 2 weeks
    """
    
    def __init__(self, costs: Costs, lt: LeadTime):
        self.costs = costs
        self.lt = lt

    def step(
        self, 
        state: pd.DataFrame, 
        demand: pd.Series, 
        order: pd.Series
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Execute one week of simulation.
        
        Args:
            state: Current state with columns [on_hand, intransit_1, intransit_2]
            demand: True demand this week (may exceed on_hand)
            order: Order quantity placed at end of week
            
        Returns:
            (next_state, costs_dict)
        """
        # Receive goods from intransit_1
        received = state["intransit_1"]
        on_hand = state["on_hand"] + received
        
        # Satisfy demand up to available stock
        sold = demand.clip(upper=on_hand)
        lost = demand - sold
        
        # Update on_hand after sales
        on_hand = on_hand - sold
        
        # Roll pipeline: intransit_2 -> intransit_1, order -> intransit_2
        next_state = pd.DataFrame(index=state.index)
        next_state["on_hand"] = on_hand
        next_state["intransit_1"] = state["intransit_2"]
        next_state["intransit_2"] = order
        
        # Calculate costs
        costs_dict = {
            "holding": float(on_hand.sum() * self.costs.holding),
            "shortage": float(lost.sum() * self.costs.shortage),
            "total": float(on_hand.sum() * self.costs.holding + lost.sum() * self.costs.shortage)
        }
        
        return next_state, costs_dict

    def run_episode(
        self,
        initial_state: pd.DataFrame,
        demands: pd.DataFrame,  # columns = weeks, index = (Store, Product)
        orders: pd.DataFrame,   # columns = weeks, index = (Store, Product)
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run full simulation episode.
        
        Returns:
            (states_history, costs_history)
        """
        state = initial_state.copy()
        states = []
        costs = []
        
        for week in demands.columns:
            demand = demands[week]
            order = orders[week] if week in orders.columns else pd.Series(0, index=state.index)
            
            state, cost = self.step(state, demand, order)
            states.append(state.copy())
            costs.append(cost)
        
        states_df = pd.DataFrame(states)
        costs_df = pd.DataFrame(costs)
        
        return states_df, costs_df

