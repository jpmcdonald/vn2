"""
Runner for 12-week sequential backtest across models and SKUs.

Orchestrates parallel evaluation and aggregation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from joblib import Parallel, delayed
import json
from datetime import datetime

from .sequential_backtest import (
    BacktestResult,
    BacktestState,
    Costs,
    run_12week_backtest,
    reconstruct_initial_state,
    load_actual_demand
)
from .forecast_loader import (
    load_forecasts_for_sku,
    get_available_models,
    get_available_skus
)


def run_single_backtest(
    store: int,
    product: int,
    model_name: str,
    checkpoints_dir: Path,
    initial_state_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    costs: Costs,
    pmf_grain: int = 500,
    n_weeks: int = 12
) -> Optional[BacktestResult]:
    """
    Run backtest for one (model, SKU) combination.
    
    Args:
        store: Store ID
        product: Product ID
        model_name: Model name
        checkpoints_dir: Path to checkpoints
        initial_state_df: Initial state DataFrame
        sales_df: Sales history DataFrame
        costs: Cost parameters
        pmf_grain: PMF support size
        n_weeks: Number of weeks to backtest
    
    Returns:
        BacktestResult or None if failed
    """
    try:
        # Load forecasts
        h1_pmfs, h2_pmfs = load_forecasts_for_sku(
            store, product, model_name,
            checkpoints_dir,
            n_folds=n_weeks,
            pmf_grain=pmf_grain
        )
        
        # Load actual demand
        actuals = load_actual_demand(store, product, sales_df, n_weeks)
        
        # Reconstruct initial state
        initial_state = reconstruct_initial_state(
            store, product,
            initial_state_df,
            sales_df,
            backtest_start_week=1
        )
        
        # Run backtest
        result = run_12week_backtest(
            store, product, model_name,
            h1_pmfs, h2_pmfs, actuals,
            initial_state, costs, pmf_grain
        )
        
        return result
        
    except Exception as e:
        print(f"Error processing {model_name} ({store}, {product}): {e}")
        return None


def run_all_backtests(
    checkpoints_dir: Path,
    data_dir: Path,
    output_dir: Path,
    costs: Costs,
    models: Optional[List[str]] = None,
    skus: Optional[List[Tuple[int, int]]] = None,
    n_jobs: int = 12,
    pmf_grain: int = 500,
    n_weeks: int = 12
) -> pd.DataFrame:
    """
    Run backtests for all models and SKUs.
    
    Args:
        checkpoints_dir: Path to forecast checkpoints
        data_dir: Path to raw data directory
        output_dir: Path to output directory
        costs: Cost parameters
        models: List of model names (None = all available)
        skus: List of (store, product) tuples (None = all available)
        n_jobs: Number of parallel jobs
        pmf_grain: PMF support size
        n_weeks: Number of weeks to backtest
    
    Returns:
        DataFrame with aggregated results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    initial_state_df = pd.read_csv(data_dir / "Week 0 - 2024-04-08 - Initial State.csv")
    sales_df = pd.read_csv(data_dir / "Week 1 - 2024-04-15 - Sales.csv")
    
    # Get models
    if models is None:
        models = get_available_models(checkpoints_dir)
    print(f"Models: {models}")
    
    # Get SKUs (use first model to determine universe)
    if skus is None:
        skus = get_available_skus(checkpoints_dir, models[0])
    print(f"SKUs: {len(skus)}")
    
    # Generate tasks
    tasks = []
    for model in models:
        for store, product in skus:
            tasks.append((store, product, model))
    
    print(f"Total tasks: {len(tasks)}")
    
    # Run in parallel
    print(f"Running backtests with {n_jobs} workers...")
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_backtest)(
            store, product, model,
            checkpoints_dir, initial_state_df, sales_df,
            costs, pmf_grain, n_weeks
        )
        for store, product, model in tasks
    )
    
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    print(f"Valid results: {len(valid_results)} / {len(tasks)}")
    
    # Convert to DataFrame
    records = []
    for result in valid_results:
        records.append({
            'store': result.store,
            'product': result.product,
            'model': result.model_name,
            'total_expected_cost': result.total_expected_cost,
            'total_realized_cost': result.total_realized_cost,
            'total_expected_cost_excl_w1': result.total_expected_cost_excl_w1,
            'total_realized_cost_excl_w1': result.total_realized_cost_excl_w1,
            'n_weeks': result.n_weeks,
            'n_missing_forecasts': result.n_missing_forecasts,
            'max_pmf_residual': result.diagnostics.get('max_pmf_residual', 0.0)
        })
    
    df = pd.DataFrame(records)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"backtest_results_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")
    
    # Save detailed per-week results
    detailed_records = []
    for result in valid_results:
        for week in result.weeks:
            detailed_records.append({
                'store': result.store,
                'product': result.product,
                'model': result.model_name,
                'week': week.week,
                'order_placed': week.order_placed,
                'demand_actual': week.demand_actual,
                'expected_cost': week.expected_cost,
                'realized_cost': week.realized_cost,
                'on_hand_before': week.state_before.on_hand,
                'intransit_1_before': week.state_before.intransit_1,
                'intransit_2_before': week.state_before.intransit_2,
                'pmf_residual': week.pmf_residual
            })
    
    detailed_df = pd.DataFrame(detailed_records)
    detailed_file = output_dir / f"backtest_detailed_{timestamp}.csv"
    detailed_df.to_csv(detailed_file, index=False)
    print(f"Saved detailed results to {detailed_file}")
    
    return df


def compute_selector(
    results_df: pd.DataFrame,
    output_dir: Path,
    cost_column: str = 'total_realized_cost'
) -> pd.DataFrame:
    """
    Compute per-SKU model selector based on minimum cost.
    
    Args:
        results_df: DataFrame with backtest results
        output_dir: Path to output directory
        cost_column: Column to use for selection
    
    Returns:
        DataFrame with selected models per SKU
    """
    # Find best model per SKU
    idx = results_df.groupby(['store', 'product'])[cost_column].idxmin()
    selector_df = results_df.loc[idx, ['store', 'product', 'model', cost_column]].copy()
    selector_df = selector_df.rename(columns={'model': 'selected_model', cost_column: 'best_cost'})
    
    # Save selector
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    selector_file = output_dir / f"selector_{timestamp}.csv"
    selector_df.to_csv(selector_file, index=False)
    print(f"Saved selector to {selector_file}")
    
    # Print summary
    print("\nSelector Summary:")
    print(selector_df['selected_model'].value_counts())
    print(f"\nTotal SKUs: {len(selector_df)}")
    print(f"Mean best cost: {selector_df['best_cost'].mean():.2f}")
    print(f"Median best cost: {selector_df['best_cost'].median():.2f}")
    
    return selector_df


def generate_leaderboard(
    results_df: pd.DataFrame,
    output_dir: Path,
    cost_column: str = 'total_realized_cost'
) -> pd.DataFrame:
    """
    Generate model leaderboard by portfolio total cost.
    
    Args:
        results_df: DataFrame with backtest results
        output_dir: Path to output directory
        cost_column: Column to use for ranking
    
    Returns:
        DataFrame with leaderboard
    """
    # Aggregate by model
    leaderboard = results_df.groupby('model').agg({
        cost_column: ['sum', 'mean', 'median', 'std'],
        'store': 'count'
    }).reset_index()
    
    leaderboard.columns = ['model', 'total_cost', 'mean_cost', 'median_cost', 'std_cost', 'n_skus']
    leaderboard = leaderboard.sort_values('total_cost')
    
    # Save leaderboard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    leaderboard_file = output_dir / f"leaderboard_{timestamp}.csv"
    leaderboard.to_csv(leaderboard_file, index=False)
    print(f"Saved leaderboard to {leaderboard_file}")
    
    # Print leaderboard
    print("\nModel Leaderboard:")
    print(leaderboard.to_string(index=False))
    
    return leaderboard

