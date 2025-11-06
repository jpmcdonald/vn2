"""
Strategy backtester for evaluating forecasting strategies with temporal constraints.

This module implements the main backtesting engine that can evaluate different
forecasting strategies while respecting strict temporal data availability.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .temporal_data import TemporalDataManager, DataSnapshot
from .strategies import ForecastStrategy, StrategyResult
from ..analyze.sequential_planner import Costs
from ..analyze.sequential_backtest import quantiles_to_pmf, BacktestState


@dataclass
class WeeklyResult:
    """Result for one week of backtesting."""
    week: int
    strategy_name: str
    total_expected_cost: float
    total_realized_cost: float
    n_skus_processed: int
    n_skus_with_forecasts: int
    model_usage: Dict[str, int]
    sku_results: List[StrategyResult]


@dataclass 
class BacktestResults:
    """Complete backtest results for a strategy."""
    strategy_name: str
    weekly_results: List[WeeklyResult]
    total_expected_cost: float
    total_realized_cost: float
    cumulative_costs: Dict[int, float]  # week -> cumulative cost
    model_attribution: Dict[str, float]  # model -> total cost contribution
    validation_metrics: Dict[str, float]


class StrategyBacktester:
    """
    Main backtesting engine for strategy evaluation.
    
    Evaluates forecasting strategies while respecting temporal constraints
    and calculating both expected and realized costs week by week.
    """
    
    def __init__(
        self,
        data_manager: TemporalDataManager,
        checkpoints_dir: Path,
        costs: Costs,
        quantile_levels: Optional[np.ndarray] = None,
        pmf_grain: int = 500
    ):
        """
        Initialize backtester.
        
        Args:
            data_manager: Temporal data manager
            checkpoints_dir: Path to forecast checkpoints
            costs: Cost parameters
            quantile_levels: Quantile levels for PMF conversion
            pmf_grain: PMF support size
        """
        self.data_manager = data_manager
        self.checkpoints_dir = checkpoints_dir
        self.costs = costs
        self.pmf_grain = pmf_grain
        
        if quantile_levels is None:
            self.quantile_levels = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 
                                           0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
        else:
            self.quantile_levels = quantile_levels
    
    def run_strategy_backtest(
        self,
        strategy: ForecastStrategy,
        weeks: List[int] = [1, 2, 3, 4]
    ) -> BacktestResults:
        """
        Run complete strategy backtest.
        
        Args:
            strategy: Forecasting strategy to evaluate
            weeks: List of weeks to backtest
        
        Returns:
            BacktestResults with complete evaluation
        """
        print(f"Running backtest for strategy: {strategy.name}")
        print(f"Weeks: {weeks}")
        print()
        
        weekly_results = []
        cumulative_costs = {}
        model_attribution = {}
        cumulative_cost = 0.0
        
        for week in weeks:
            print(f"Processing Week {week}...")
            
            # Get data snapshot for this week
            snapshot = self.data_manager.get_data_snapshot(week)
            
            # Run week evaluation
            week_result = self._evaluate_week(strategy, snapshot)
            weekly_results.append(week_result)
            
            # Update cumulative tracking
            cumulative_cost += week_result.total_realized_cost
            cumulative_costs[week] = cumulative_cost
            
            # Update model attribution
            for sku_result in week_result.sku_results:
                model = sku_result.model_used
                if model not in model_attribution:
                    model_attribution[model] = 0.0
                # Use actual cost for attribution (what really happened)
                actual_cost = self._get_actual_cost_for_sku(
                    sku_result.store, sku_result.product, week, snapshot
                )
                model_attribution[model] += actual_cost
            
            print(f"  Expected cost: {week_result.total_expected_cost:.2f}")
            print(f"  Realized cost: {week_result.total_realized_cost:.2f}")
            print(f"  SKUs processed: {week_result.n_skus_processed}")
            print()
        
        # Calculate validation metrics
        validation_metrics = self._calculate_validation_metrics(weekly_results)
        
        return BacktestResults(
            strategy_name=strategy.name,
            weekly_results=weekly_results,
            total_expected_cost=sum(w.total_expected_cost for w in weekly_results),
            total_realized_cost=sum(w.total_realized_cost for w in weekly_results),
            cumulative_costs=cumulative_costs,
            model_attribution=model_attribution,
            validation_metrics=validation_metrics
        )
    
    def _evaluate_week(
        self,
        strategy: ForecastStrategy, 
        snapshot: DataSnapshot
    ) -> WeeklyResult:
        """
        Evaluate strategy for one week.
        
        Args:
            strategy: Forecasting strategy
            snapshot: Data available at decision time
        
        Returns:
            WeeklyResult for this week
        """
        week = snapshot.decision_week
        sku_results = []
        model_usage = {}
        total_expected = 0.0
        total_realized = 0.0
        n_with_forecasts = 0
        
        # Get available models for this week
        available_models = self._get_available_models(week)
        
        # Process each SKU
        for (store, product), state in snapshot.available_state.items():
            
            # Select model for this SKU
            selected_model = strategy.select_model_for_sku(
                store, product, snapshot, available_models
            )
            
            # Track model usage
            model_usage[selected_model] = model_usage.get(selected_model, 0) + 1
            
            # Load forecast for this SKU and model
            forecast_h1, forecast_h2 = self._load_forecast(
                store, product, selected_model, week
            )
            
            if forecast_h1 is not None and forecast_h2 is not None:
                # Apply strategy adjustments
                adjusted_h1 = strategy.adjust_forecast(
                    forecast_h1, store, product, selected_model, snapshot
                )
                adjusted_h2 = strategy.adjust_forecast(
                    forecast_h2, store, product, selected_model, snapshot
                )
                
                # Convert to PMFs
                h1_pmf = quantiles_to_pmf(adjusted_h1, self.quantile_levels, self.pmf_grain)
                h2_pmf = quantiles_to_pmf(adjusted_h2, self.quantile_levels, self.pmf_grain)
                
                # Make order decision
                order_qty, expected_cost = strategy.make_order_decision(
                    store, product, h1_pmf, h2_pmf, state, snapshot
                )
                
                n_with_forecasts += 1
            else:
                # No forecast available
                order_qty = 0
                expected_cost = 0.0
            
            # Calculate realized cost for this week
            realized_cost = self._get_actual_cost_for_sku(store, product, week, snapshot)
            
            total_expected += expected_cost
            total_realized += realized_cost
            
            sku_results.append(StrategyResult(
                week=week,
                store=store,
                product=product,
                order_placed=order_qty,
                expected_cost=expected_cost,
                model_used=selected_model,
                confidence=1.0,  # TODO: Calculate actual confidence
                strategy_name=strategy.name
            ))
        
        return WeeklyResult(
            week=week,
            strategy_name=strategy.name,
            total_expected_cost=total_expected,
            total_realized_cost=total_realized,
            n_skus_processed=len(sku_results),
            n_skus_with_forecasts=n_with_forecasts,
            model_usage=model_usage,
            sku_results=sku_results
        )
    
    def _get_available_models(self, week: int) -> List[str]:
        """
        Get models available for a specific week.
        
        Args:
            week: Week number
        
        Returns:
            List of available model names
        """
        # Check which models have checkpoints for this week
        available_models = []
        
        fold_idx = week - 1  # fold_0 = week 1, etc.
        
        for model_dir in self.checkpoints_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            
            # Check if this model has any forecasts for this fold
            has_forecasts = False
            for sku_dir in model_dir.iterdir():
                if sku_dir.is_dir():
                    fold_file = sku_dir / f'fold_{fold_idx}.pkl'
                    if fold_file.exists():
                        has_forecasts = True
                        break
            
            if has_forecasts:
                available_models.append(model_name)
        
        return sorted(available_models)
    
    def _load_forecast(
        self,
        store: int,
        product: int, 
        model_name: str,
        week: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load forecast for SKU, model, and week.
        
        Args:
            store: Store ID
            product: Product ID
            model_name: Model name
            week: Decision week
        
        Returns:
            (h1_quantiles, h2_quantiles) or (None, None) if not available
        """
        fold_idx = week - 1  # fold_0 = week 1, etc.
        
        checkpoint_path = (self.checkpoints_dir / model_name / 
                          f'{store}_{product}' / f'fold_{fold_idx}.pkl')
        
        if not checkpoint_path.exists():
            return None, None
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            quantiles_df = checkpoint.get('quantiles')
            if quantiles_df is None or len(quantiles_df) < 2:
                return None, None
            
            # Get h1 and h2 quantiles
            h1_quantiles = quantiles_df.loc[1].values if 1 in quantiles_df.index else quantiles_df.iloc[0].values
            h2_quantiles = quantiles_df.loc[2].values if 2 in quantiles_df.index else quantiles_df.iloc[1].values
            
            return h1_quantiles, h2_quantiles
            
        except Exception as e:
            print(f"Warning: Failed to load forecast for ({store}, {product}, {model_name}, week {week}): {e}")
            return None, None
    
    def _get_actual_cost_for_sku(
        self,
        store: int,
        product: int,
        week: int,
        snapshot: DataSnapshot
    ) -> float:
        """
        Get actual realized cost for SKU in a specific week.
        
        This uses the state files to get the actual costs that were incurred.
        """
        # Try to get cost from state file
        state_file = self.data_manager.available_files.get(f'state{week}')
        
        if state_file and state_file.exists():
            state_df = pd.read_csv(state_file)
            sku_row = state_df[
                (state_df['Store'] == store) & (state_df['Product'] == product)
            ]
            
            if len(sku_row) > 0:
                holding_cost = sku_row.iloc[0].get('Holding Cost', 0.0)
                shortage_cost = sku_row.iloc[0].get('Shortage Cost', 0.0)
                return float(holding_cost + shortage_cost)
        
        # Fallback: estimate cost from available sales data
        sales_data = snapshot.available_sales.get(week)
        if sales_data is not None:
            sku_sales = sales_data[
                (sales_data['Store'] == store) & (sales_data['Product'] == product)
            ]
            if len(sku_sales) > 0:
                # Simple cost estimation (would need more sophisticated logic)
                return 0.0  # Placeholder
        
        return 0.0
    
    def _calculate_validation_metrics(self, weekly_results: List[WeeklyResult]) -> Dict[str, float]:
        """
        Calculate validation metrics for the backtest.
        
        Args:
            weekly_results: List of weekly results
        
        Returns:
            Dict with validation metrics
        """
        if not weekly_results:
            return {}
        
        # Aggregate across weeks
        total_expected = sum(w.total_expected_cost for w in weekly_results)
        total_realized = sum(w.total_realized_cost for w in weekly_results)
        
        # Model diversity
        all_models = set()
        for week_result in weekly_results:
            all_models.update(week_result.model_usage.keys())
        
        # Coverage (simplified - would need actual CI data)
        forecast_accuracy = 1.0 - abs(total_realized - total_expected) / max(total_realized, 1.0)
        
        return {
            'total_expected_cost': total_expected,
            'total_realized_cost': total_realized,
            'cost_error': total_realized - total_expected,
            'cost_error_percentage': (total_realized - total_expected) / max(abs(total_expected), 1) * 100,
            'forecast_accuracy': forecast_accuracy,
            'n_models_used': len(all_models),
            'n_weeks_evaluated': len(weekly_results),
            'avg_skus_per_week': np.mean([w.n_skus_processed for w in weekly_results]),
            'forecast_coverage': np.mean([w.n_skus_with_forecasts / max(w.n_skus_processed, 1) for w in weekly_results])
        }
    
    def compare_strategies(
        self,
        strategies: List[ForecastStrategy],
        weeks: List[int] = [1, 2, 3, 4]
    ) -> pd.DataFrame:
        """
        Compare multiple strategies across weeks.
        
        Args:
            strategies: List of strategies to compare
            weeks: Weeks to evaluate
        
        Returns:
            DataFrame with strategy comparison
        """
        all_results = []
        
        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"Evaluating Strategy: {strategy.name}")
            print('='*60)
            
            result = self.run_strategy_backtest(strategy, weeks)
            
            # Add to comparison
            for week_result in result.weekly_results:
                all_results.append({
                    'strategy': strategy.name,
                    'week': week_result.week,
                    'expected_cost': week_result.total_expected_cost,
                    'realized_cost': week_result.total_realized_cost,
                    'cost_difference': week_result.total_realized_cost - week_result.total_expected_cost,
                    'n_skus': week_result.n_skus_processed,
                    'forecast_coverage': week_result.n_skus_with_forecasts / max(week_result.n_skus_processed, 1)
                })
        
        comparison_df = pd.DataFrame(all_results)
        
        # Add cumulative columns
        comparison_df['cumulative_realized'] = comparison_df.groupby('strategy')['realized_cost'].cumsum()
        comparison_df['cumulative_expected'] = comparison_df.groupby('strategy')['expected_cost'].cumsum()
        
        return comparison_df
    
    def generate_research_summary(
        self,
        comparison_df: pd.DataFrame,
        baseline_strategy: str = 'Original'
    ) -> Dict:
        """
        Generate research-quality summary of strategy comparison.
        
        Args:
            comparison_df: Strategy comparison DataFrame
            baseline_strategy: Name of baseline strategy
        
        Returns:
            Dict with research summary
        """
        strategies = comparison_df['strategy'].unique()
        weeks = sorted(comparison_df['week'].unique())
        
        summary = {
            'strategies_evaluated': list(strategies),
            'weeks_evaluated': weeks,
            'baseline_strategy': baseline_strategy
        }
        
        # Calculate final cumulative costs
        final_week = max(weeks)
        final_costs = comparison_df[comparison_df['week'] == final_week].set_index('strategy')['cumulative_realized']
        
        if baseline_strategy in final_costs.index:
            baseline_cost = final_costs[baseline_strategy]
            
            # Calculate improvements vs baseline
            improvements = {}
            for strategy in strategies:
                if strategy != baseline_strategy and strategy in final_costs.index:
                    strategy_cost = final_costs[strategy]
                    improvement = baseline_cost - strategy_cost
                    improvement_pct = improvement / baseline_cost * 100 if baseline_cost > 0 else 0
                    
                    improvements[strategy] = {
                        'absolute_improvement': improvement,
                        'percentage_improvement': improvement_pct,
                        'final_cost': strategy_cost
                    }
            
            summary['baseline_final_cost'] = baseline_cost
            summary['improvements'] = improvements
        
        # Week-by-week progression
        summary['weekly_progression'] = {}
        for week in weeks:
            week_data = comparison_df[comparison_df['week'] == week]
            week_summary = {}
            for strategy in strategies:
                strategy_data = week_data[week_data['strategy'] == strategy]
                if len(strategy_data) > 0:
                    week_summary[strategy] = {
                        'expected': strategy_data.iloc[0]['expected_cost'],
                        'realized': strategy_data.iloc[0]['realized_cost'],
                        'cumulative': strategy_data.iloc[0]['cumulative_realized']
                    }
            summary['weekly_progression'][week] = week_summary
        
        return summary
    
    def validate_temporal_constraints(self, weeks: List[int]) -> Tuple[bool, List[str]]:
        """
        Validate that backtesting respects temporal constraints.
        
        Args:
            weeks: Weeks being backtested
        
        Returns:
            (is_valid, violations)
        """
        violations = []
        
        for week in weeks:
            snapshot = self.data_manager.get_data_snapshot(week)
            
            # Check that we're not using future data
            decision_date = snapshot.decision_date
            
            # Validate demand data cutoff
            if not snapshot.available_demand.empty:
                max_demand_date = snapshot.available_demand['week'].max()
                if max_demand_date >= decision_date:
                    violations.append(f"Week {week}: Using demand data from {max_demand_date} >= decision date {decision_date}")
            
            # Validate sales data
            for sales_week, sales_df in snapshot.available_sales.items():
                sales_date = self.data_manager.week_dates.get(sales_week)
                if sales_date and sales_date >= decision_date:
                    violations.append(f"Week {week}: Using sales data from Week {sales_week} ({sales_date}) >= decision date {decision_date}")
        
        return len(violations) == 0, violations
