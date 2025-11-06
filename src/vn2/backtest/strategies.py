"""
Forecasting strategy definitions for backtesting.

This module defines different forecasting strategies that can be compared
in backtests to evaluate their cost performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..analyze.sequential_planner import Costs
from .temporal_data import DataSnapshot


@dataclass
class StrategyResult:
    """Result of applying a strategy for one week."""
    week: int
    store: int
    product: int
    order_placed: int
    expected_cost: float
    model_used: str
    confidence: float
    strategy_name: str


class ForecastStrategy(ABC):
    """
    Base class for forecasting strategies.
    
    A strategy defines how to:
    1. Select models for each SKU
    2. Load and potentially adjust forecasts
    3. Make ordering decisions
    """
    
    def __init__(self, name: str, costs: Costs):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
            costs: Cost parameters (holding, shortage)
        """
        self.name = name
        self.costs = costs
    
    @abstractmethod
    def select_model_for_sku(
        self, 
        store: int, 
        product: int, 
        snapshot: DataSnapshot,
        available_models: List[str]
    ) -> str:
        """
        Select the best model for a SKU at decision time.
        
        Args:
            store: Store ID
            product: Product ID
            snapshot: Data available at decision time
            available_models: List of available model names
        
        Returns:
            Selected model name
        """
        pass
    
    @abstractmethod
    def adjust_forecast(
        self, 
        forecast_quantiles: np.ndarray,
        store: int,
        product: int, 
        model_name: str,
        snapshot: DataSnapshot
    ) -> np.ndarray:
        """
        Optionally adjust forecast based on strategy.
        
        Args:
            forecast_quantiles: Raw forecast quantiles
            store: Store ID
            product: Product ID
            model_name: Model name
            snapshot: Data available at decision time
        
        Returns:
            Adjusted forecast quantiles
        """
        pass
    
    def make_order_decision(
        self,
        store: int,
        product: int,
        adjusted_forecast_h1: np.ndarray,
        adjusted_forecast_h2: np.ndarray, 
        current_state: Dict,
        snapshot: DataSnapshot
    ) -> Tuple[int, float]:
        """
        Make order decision using adjusted forecasts.
        
        This uses the existing choose_order_L2 function.
        
        Args:
            store: Store ID
            product: Product ID
            adjusted_forecast_h1: Adjusted h1 forecast PMF
            adjusted_forecast_h2: Adjusted h2 forecast PMF
            current_state: Current inventory state
            snapshot: Data snapshot
        
        Returns:
            (order_quantity, expected_cost)
        """
        from ..analyze.sequential_planner import choose_order_L2
        
        order_qty, expected_cost = choose_order_L2(
            adjusted_forecast_h1,
            adjusted_forecast_h2,
            current_state['on_hand'],
            current_state['intransit_1'],
            current_state['intransit_2'],
            self.costs,
            micro_refine=True
        )
        
        return order_qty, expected_cost


class OriginalStrategy(ForecastStrategy):
    """
    Original strategy using the selector map as-is.
    
    This represents what we actually did in the competition.
    """
    
    def __init__(self, costs: Costs, selector_map_path: Path):
        """
        Initialize with original selector map.
        
        Args:
            costs: Cost parameters
            selector_map_path: Path to original selector map
        """
        super().__init__("Original", costs)
        self.selector_map = pd.read_parquet(selector_map_path)
        self.selector_dict = {}
        for _, row in self.selector_map.iterrows():
            key = (int(row['store']), int(row['product']))
            self.selector_dict[key] = row['model_name']
    
    def select_model_for_sku(
        self, 
        store: int, 
        product: int, 
        snapshot: DataSnapshot,
        available_models: List[str]
    ) -> str:
        """Use original selector map."""
        selected = self.selector_dict.get((store, product))
        if selected and selected in available_models:
            return selected
        # Fallback to first available model
        return available_models[0] if available_models else 'slurp_bootstrap'
    
    def adjust_forecast(
        self, 
        forecast_quantiles: np.ndarray,
        store: int,
        product: int, 
        model_name: str,
        snapshot: DataSnapshot
    ) -> np.ndarray:
        """No adjustments in original strategy."""
        return forecast_quantiles


class ImprovedStrategy(ForecastStrategy):
    """
    Improved strategy with bias corrections and specialized model selection.
    
    This represents what we could have done with our improvements.
    """
    
    def __init__(
        self, 
        costs: Costs, 
        bias_corrections_path: Path,
        specialized_assignments_path: Path
    ):
        """
        Initialize with improvement data.
        
        Args:
            costs: Cost parameters
            bias_corrections_path: Path to bias corrections
            specialized_assignments_path: Path to specialized model assignments
        """
        super().__init__("Improved", costs)
        
        # Load bias corrections
        self.bias_corrections = pd.read_csv(bias_corrections_path)
        self.bias_dict = self.bias_corrections.set_index('model')[
            ['cost_multiplier', 'variance_multiplier', 'min_cost_floor']
        ].to_dict('index')
        
        # Load specialized assignments
        self.specialized_assignments = pd.read_parquet(specialized_assignments_path)
        self.assignments_dict = {}
        for _, row in self.specialized_assignments.iterrows():
            key = (int(row['store']), int(row['product']))
            self.assignments_dict[key] = {
                'stockout_model': row['stockout_model'],
                'overstock_model': row['overstock_model'], 
                'density_model': row['density_model'],
                'sku_type': row['sku_type']
            }
    
    def select_model_for_sku(
        self, 
        store: int, 
        product: int, 
        snapshot: DataSnapshot,
        available_models: List[str]
    ) -> str:
        """Use specialized model selection."""
        assignment = self.assignments_dict.get((store, product))
        if assignment:
            # Choose based on SKU characteristics
            sku_type = assignment['sku_type']
            
            if sku_type == 'shortage_prone':
                preferred = assignment['stockout_model']
            elif sku_type == 'overstock_prone':
                preferred = assignment['overstock_model']
            else:
                preferred = assignment['density_model']
            
            if preferred in available_models:
                return preferred
        
        # Fallback to best available model (prefer slurp_bootstrap)
        for fallback in ['slurp_bootstrap', 'zinb', 'slurp_stockout_aware']:
            if fallback in available_models:
                return fallback
        
        return available_models[0] if available_models else 'slurp_bootstrap'
    
    def adjust_forecast(
        self, 
        forecast_quantiles: np.ndarray,
        store: int,
        product: int, 
        model_name: str,
        snapshot: DataSnapshot
    ) -> np.ndarray:
        """Apply bias corrections to forecasts."""
        if model_name in self.bias_dict:
            corrections = self.bias_dict[model_name]
            
            # Apply cost multiplier
            adjusted = forecast_quantiles * corrections['cost_multiplier']
            
            # Apply minimum floor
            adjusted = np.maximum(adjusted, corrections['min_cost_floor'])
            
            # Apply variance multiplier (spread around median)
            median_val = np.median(adjusted)
            deviations = adjusted - median_val
            adjusted = median_val + deviations * corrections['variance_multiplier']
            
            # Ensure non-negative and monotonic
            adjusted = np.maximum(adjusted, 0)
            adjusted = np.sort(adjusted)  # Ensure monotonicity
            
            return adjusted
        
        return forecast_quantiles


class BiasOnlyStrategy(ForecastStrategy):
    """
    Strategy with only bias corrections, no specialized model selection.
    
    For ablation studies to isolate the impact of bias corrections.
    """
    
    def __init__(self, costs: Costs, bias_corrections_path: Path, selector_map_path: Path):
        super().__init__("BiasOnly", costs)
        
        # Load original selector
        self.selector_map = pd.read_parquet(selector_map_path)
        self.selector_dict = {}
        for _, row in self.selector_map.iterrows():
            key = (int(row['store']), int(row['product']))
            self.selector_dict[key] = row['model_name']
        
        # Load bias corrections
        self.bias_corrections = pd.read_csv(bias_corrections_path)
        self.bias_dict = self.bias_corrections.set_index('model')[
            ['cost_multiplier', 'variance_multiplier', 'min_cost_floor']
        ].to_dict('index')
    
    def select_model_for_sku(
        self, 
        store: int, 
        product: int, 
        snapshot: DataSnapshot,
        available_models: List[str]
    ) -> str:
        """Use original selector map (no specialized selection)."""
        selected = self.selector_dict.get((store, product))
        if selected and selected in available_models:
            return selected
        return available_models[0] if available_models else 'slurp_bootstrap'
    
    def adjust_forecast(
        self, 
        forecast_quantiles: np.ndarray,
        store: int,
        product: int, 
        model_name: str,
        snapshot: DataSnapshot
    ) -> np.ndarray:
        """Apply bias corrections only."""
        if model_name in self.bias_dict:
            corrections = self.bias_dict[model_name]
            
            adjusted = forecast_quantiles * corrections['cost_multiplier']
            adjusted = np.maximum(adjusted, corrections['min_cost_floor'])
            
            median_val = np.median(adjusted)
            deviations = adjusted - median_val
            adjusted = median_val + deviations * corrections['variance_multiplier']
            
            adjusted = np.maximum(adjusted, 0)
            adjusted = np.sort(adjusted)
            
            return adjusted
        
        return forecast_quantiles


class SpecializedOnlyStrategy(ForecastStrategy):
    """
    Strategy with only specialized model selection, no bias corrections.
    
    For ablation studies to isolate the impact of specialized ensembles.
    """
    
    def __init__(self, costs: Costs, specialized_assignments_path: Path):
        super().__init__("SpecializedOnly", costs)
        
        # Load specialized assignments
        self.specialized_assignments = pd.read_parquet(specialized_assignments_path)
        self.assignments_dict = {}
        for _, row in self.specialized_assignments.iterrows():
            key = (int(row['store']), int(row['product']))
            self.assignments_dict[key] = {
                'stockout_model': row['stockout_model'],
                'overstock_model': row['overstock_model'], 
                'density_model': row['density_model'],
                'sku_type': row['sku_type']
            }
    
    def select_model_for_sku(
        self, 
        store: int, 
        product: int, 
        snapshot: DataSnapshot,
        available_models: List[str]
    ) -> str:
        """Use specialized model selection (no bias corrections)."""
        assignment = self.assignments_dict.get((store, product))
        if assignment:
            sku_type = assignment['sku_type']
            
            if sku_type == 'shortage_prone':
                preferred = assignment['stockout_model']
            elif sku_type == 'overstock_prone':
                preferred = assignment['overstock_model']
            else:
                preferred = assignment['density_model']
            
            if preferred in available_models:
                return preferred
        
        # Fallback
        for fallback in ['slurp_bootstrap', 'zinb', 'slurp_stockout_aware']:
            if fallback in available_models:
                return fallback
        
        return available_models[0] if available_models else 'slurp_bootstrap'
    
    def adjust_forecast(
        self, 
        forecast_quantiles: np.ndarray,
        store: int,
        product: int, 
        model_name: str,
        snapshot: DataSnapshot
    ) -> np.ndarray:
        """No bias corrections in this strategy."""
        return forecast_quantiles


class PerfectForesightStrategy(ForecastStrategy):
    """
    Theoretical upper bound: perfect foresight of actual demand.
    
    For research purposes to establish theoretical maximum performance.
    """
    
    def __init__(self, costs: Costs, actual_sales_data: Dict[int, pd.DataFrame]):
        super().__init__("PerfectForesight", costs)
        self.actual_sales = actual_sales_data
    
    def select_model_for_sku(
        self, 
        store: int, 
        product: int, 
        snapshot: DataSnapshot,
        available_models: List[str]
    ) -> str:
        """Model doesn't matter with perfect foresight."""
        return 'perfect_foresight'
    
    def adjust_forecast(
        self, 
        forecast_quantiles: np.ndarray,
        store: int,
        product: int, 
        model_name: str,
        snapshot: DataSnapshot
    ) -> np.ndarray:
        """Replace forecast with perfect foresight."""
        # Get actual demand for the weeks when orders will arrive
        week = snapshot.decision_week
        
        # Orders placed in week N arrive in week N+2
        arrival_weeks = [week + 2, week + 3]
        
        perfect_demands = []
        for arrival_week in arrival_weeks:
            if arrival_week in self.actual_sales:
                sales_df = self.actual_sales[arrival_week]
                sku_sales = sales_df[
                    (sales_df['Store'] == store) & (sales_df['Product'] == product)
                ]
                if len(sku_sales) > 0:
                    actual_demand = sku_sales.iloc[0]['Sales']  # Assuming 'Sales' column
                    perfect_demands.append(actual_demand)
                else:
                    perfect_demands.append(0)
            else:
                perfect_demands.append(0)
        
        # Create perfect forecast (all quantiles equal to actual)
        if len(perfect_demands) >= 1:
            return np.full(len(forecast_quantiles), perfect_demands[0])
        else:
            return forecast_quantiles


def create_strategy(
    strategy_name: str,
    costs: Costs,
    **kwargs
) -> ForecastStrategy:
    """
    Factory function to create strategies.
    
    Args:
        strategy_name: Name of strategy to create
        costs: Cost parameters
        **kwargs: Strategy-specific parameters
    
    Returns:
        ForecastStrategy instance
    """
    if strategy_name == "original":
        return OriginalStrategy(costs, kwargs['selector_map_path'])
    
    elif strategy_name == "improved":
        return ImprovedStrategy(
            costs, 
            kwargs['bias_corrections_path'],
            kwargs['specialized_assignments_path']
        )
    
    elif strategy_name == "bias_only":
        return BiasOnlyStrategy(
            costs,
            kwargs['bias_corrections_path'],
            kwargs['selector_map_path']
        )
    
    elif strategy_name == "specialized_only":
        return SpecializedOnlyStrategy(
            costs,
            kwargs['specialized_assignments_path']
        )
    
    elif strategy_name == "perfect_foresight":
        return PerfectForesightStrategy(costs, kwargs['actual_sales_data'])
    
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


# Strategy configurations for common research scenarios
STRATEGY_CONFIGS = {
    'original': {
        'description': 'Original competition strategy',
        'required_files': ['selector_map_path']
    },
    'improved': {
        'description': 'Full improvements (bias + specialized)',
        'required_files': ['bias_corrections_path', 'specialized_assignments_path']
    },
    'bias_only': {
        'description': 'Bias corrections only (ablation)',
        'required_files': ['bias_corrections_path', 'selector_map_path']
    },
    'specialized_only': {
        'description': 'Specialized selection only (ablation)',
        'required_files': ['specialized_assignments_path']
    },
    'perfect_foresight': {
        'description': 'Theoretical upper bound',
        'required_files': ['actual_sales_data']
    }
}
