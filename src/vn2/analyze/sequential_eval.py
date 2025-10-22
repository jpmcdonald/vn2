"""
Sequential L=2 evaluation with per-SKU cost tracking.

Evaluates forecast models by running exact sequential planning over H=12 epochs,
computing realized costs only for decision-affected periods (t+2 for each decision at t).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from joblib import Parallel, delayed
from rich import print as rprint
import pickle

from vn2.analyze.sequential_planner import Costs, _safe_pmf
from vn2.analyze.sequential_backtest import (
    run_12week_backtest,
    reconstruct_initial_state,
    load_actual_demand,
    quantiles_to_pmf,
    BacktestState
)
from vn2.analyze.forecast_loader import load_forecasts_for_sku as load_pmfs_for_sku


@dataclass
class SequentialConfig:
    """Configuration for sequential evaluation"""
    checkpoints_dir: Path
    demand_path: Path
    state_path: Path
    output_dir: Path
    run_tag: str = "seq12"
    n_jobs: int = 12
    holding_cost: float = 0.2
    shortage_cost: float = 1.0
    sip_grain: int = 500
    holdout_weeks: int = 12  # H epochs


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict]:
    """Load a model checkpoint."""
    if not checkpoint_path.exists():
        return None
    try:
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        rprint(f"[yellow]Warning: Failed to load {checkpoint_path}: {e}[/yellow]")
        return None


def get_sku_universe(checkpoints_dir: Path) -> List[Tuple[int, int]]:
    """Get list of all SKUs with checkpoints."""
    skus = set()
    for model_dir in checkpoints_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for sku_dir in model_dir.iterdir():
            if sku_dir.is_dir():
                # Parse store_product directory name
                parts = sku_dir.name.split('_')
                if len(parts) == 2:
                    try:
                        store = int(parts[0])
                        product = int(parts[1])
                        skus.add((store, product))
                    except ValueError:
                        continue
    return sorted(list(skus))


def get_models(checkpoints_dir: Path) -> List[str]:
    """Get list of all models with checkpoints."""
    models = []
    for model_dir in checkpoints_dir.iterdir():
        if model_dir.is_dir():
            models.append(model_dir.name)
    return sorted(models)


def load_forecasts_for_sku(
    store: int,
    product: int,
    model_name: str,
    checkpoints_dir: Path,
    H: int,
    quantile_levels: np.ndarray,
    sip_grain: int
) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
    """
    Load h+1 and h+2 PMFs for all epochs (folds) for a SKU.
    
    Args:
        store: Store ID
        product: Product ID
        model_name: Model name
        checkpoints_dir: Path to checkpoints
        H: Number of epochs (folds)
        quantile_levels: Quantile levels to expect
        sip_grain: PMF grain (max support)
    
    Returns:
        (forecasts_h1, forecasts_h2): Lists of PMFs (or None if missing)
    """
    forecasts_h1 = []
    forecasts_h2 = []
    
    for fold_idx in range(H):
        checkpoint_path = checkpoints_dir / model_name / f"{store}_{product}" / f"fold_{fold_idx}.pkl"
        checkpoint = load_checkpoint(checkpoint_path)
        
        if checkpoint is None or 'quantiles' not in checkpoint:
            forecasts_h1.append(None)
            forecasts_h2.append(None)
            continue
        
        quantiles_df = checkpoint['quantiles']
        
        # Extract h=1 and h=2 quantiles
        h1_quantiles = None
        h2_quantiles = None
        
        if 1 in quantiles_df.index:
            h1_quantiles = quantiles_df.loc[1].values
        if 2 in quantiles_df.index:
            h2_quantiles = quantiles_df.loc[2].values
        
        # Convert to PMFs
        if h1_quantiles is not None and len(h1_quantiles) == len(quantile_levels):
            try:
                h1_pmf = quantiles_to_pmf(h1_quantiles, quantile_levels, grain=sip_grain)
                forecasts_h1.append(h1_pmf)
            except Exception:
                forecasts_h1.append(None)
        else:
            forecasts_h1.append(None)
        
        if h2_quantiles is not None and len(h2_quantiles) == len(quantile_levels):
            try:
                h2_pmf = quantiles_to_pmf(h2_quantiles, quantile_levels, grain=sip_grain)
                forecasts_h2.append(h2_pmf)
            except Exception:
                forecasts_h2.append(None)
        else:
            forecasts_h2.append(None)
    
    return forecasts_h1, forecasts_h2


def get_actuals_for_sku(
    store: int,
    product: int,
    demand_df: pd.DataFrame
) -> np.ndarray:
    """
    Get actual demand time series for a SKU.
    
    Args:
        store: Store ID
        product: Product ID
        demand_df: Demand dataframe
    
    Returns:
        Array of actual demands (sorted by week)
    """
    sku_data = demand_df[
        (demand_df['Store'] == store) & (demand_df['Product'] == product)
    ].sort_values('week').reset_index(drop=True)
    
    if len(sku_data) == 0:
        return np.array([])
    
    actuals = sku_data['sales'].values
    return np.round(actuals).astype(int)


def get_initial_state(
    store: int,
    product: int,
    state_df: pd.DataFrame
) -> Tuple[int, int, int]:
    """
    Get initial inventory state (I0, Q1, Q2) for a SKU.
    
    Args:
        store: Store ID
        product: Product ID
        state_df: State dataframe indexed by (Store, Product)
    
    Returns:
        (I0, Q1, Q2)
    """
    try:
        state_row = state_df.loc[(store, product)]
        I0 = int(state_row['on_hand'])
        Q1 = int(state_row['intransit_1'])
        Q2 = int(state_row['intransit_2'])
    except KeyError:
        # Fallback to zero state
        I0, Q1, Q2 = 0, 0, 0
    
    return I0, Q1, Q2


def evaluate_sku_model(
    store: int,
    product: int,
    model_name: str,
    checkpoints_dir: Path,
    demand_df: pd.DataFrame,
    state_df: pd.DataFrame,
    config: SequentialConfig,
    quantile_levels: np.ndarray,
    fallback_pmf: Optional[np.ndarray] = None
) -> Optional[Dict]:
    """
    Evaluate a single (SKU, model) combination over H epochs.
    
    Args:
        store: Store ID
        product: Product ID
        model_name: Model name
        checkpoints_dir: Path to checkpoints
        demand_df: Demand dataframe
        state_df: State dataframe
        config: Configuration
        quantile_levels: Quantile levels
        fallback_pmf: PMF to use for missing forecasts
    
    Returns:
        Dict with evaluation results or None if failed
    """
    # Load forecasts using new loader
    forecasts_h1, forecasts_h2 = load_pmfs_for_sku(
        store, product, model_name, checkpoints_dir,
        n_folds=config.holdout_weeks,
        quantile_levels=quantile_levels,
        pmf_grain=config.sip_grain
    )
    
    # Get actuals
    actuals = get_actuals_for_sku(store, product, demand_df)
    
    if len(actuals) < config.holdout_weeks:
        return None  # Not enough actuals
    
    # Truncate to holdout_weeks
    actuals = actuals[:config.holdout_weeks]
    
    # Get initial state
    I0, Q1, Q2 = get_initial_state(store, product, state_df)
    
    # Create BacktestState
    initial_state = BacktestState(
        week=1,
        on_hand=I0,
        intransit_1=Q1,
        intransit_2=Q2
    )
    
    # Setup costs
    costs = Costs(holding=config.holding_cost, shortage=config.shortage_cost)
    
    # Run 12-week backtest
    try:
        result = run_12week_backtest(
            store=store,
            product=product,
            model_name=model_name,
            forecasts_h1=forecasts_h1,
            forecasts_h2=forecasts_h2,
            actuals=actuals,
            initial_state=initial_state,
            costs=costs,
            pmf_grain=config.sip_grain
        )
        
        return {
            'model_name': model_name,
            'store': store,
            'product': product,
            'total_cost': result.total_realized_cost,
            'total_cost_excl_w1': result.total_realized_cost_excl_w1,
            'total_expected_cost': result.total_expected_cost,
            'total_expected_cost_excl_w1': result.total_expected_cost_excl_w1,
            'n_missing': result.n_missing_forecasts,
            'n_weeks': result.n_weeks,
            'orders': [w.order_placed for w in result.weeks],
            'costs_by_week': [w.realized_cost for w in result.weeks],
            'max_pmf_residual': result.diagnostics.get('max_pmf_residual', 0.0)
        }
    except Exception as e:
        rprint(f"[yellow]Warning: Failed to evaluate {model_name} for ({store}, {product}): {e}[/yellow]")
        return None


def run_sequential_evaluation(config: SequentialConfig) -> pd.DataFrame:
    """
    Run sequential evaluation for all models and SKUs.
    
    Returns:
        DataFrame with per-SKU, per-model results
    """
    rprint(f"[bold blue]🚀 Sequential L=2 Evaluation: H={config.holdout_weeks} epochs[/bold blue]")
    rprint(f"  Checkpoints: {config.checkpoints_dir}")
    rprint(f"  Output: {config.output_dir}")
    rprint(f"  Costs: cu={config.shortage_cost}, co={config.holding_cost}")
    rprint(f"  PMF grain: {config.sip_grain}")
    
    # Load data
    rprint("[cyan]Loading demand and state data...[/cyan]")
    demand_df = pd.read_parquet(config.demand_path)
    state_df = pd.read_parquet(config.state_path)
    
    # Ensure state is indexed
    if not isinstance(state_df.index, pd.MultiIndex):
        if 'Store' in state_df.columns and 'Product' in state_df.columns:
            state_df = state_df.set_index(['Store', 'Product'])
    
    rprint(f"  Demand: {len(demand_df):,} rows")
    rprint(f"  State: {len(state_df):,} SKUs")
    
    # Get universe
    rprint("[cyan]Discovering SKUs and models...[/cyan]")
    models = get_models(config.checkpoints_dir)
    skus = get_sku_universe(config.checkpoints_dir)
    
    rprint(f"  Models: {len(models)}")
    rprint(f"  SKUs: {len(skus)}")
    
    # Quantile levels (from config or default)
    quantile_levels = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    
    # Optional: create fallback PMF (uniform or zero-inflated)
    # For now, use None (missing forecasts → q=0)
    fallback_pmf = None
    
    # Generate tasks
    tasks = []
    for model in models:
        for store, product in skus:
            tasks.append((store, product, model))
    
    rprint(f"[cyan]Running {len(tasks):,} evaluations with {config.n_jobs} workers...[/cyan]")
    
    # Parallel evaluation
    results = Parallel(n_jobs=config.n_jobs, backend='loky', verbose=5)(
        delayed(evaluate_sku_model)(
            store, product, model,
            config.checkpoints_dir, demand_df, state_df,
            config, quantile_levels, fallback_pmf
        ) for store, product, model in tasks
    )
    
    # Filter None results
    results = [r for r in results if r is not None]
    
    rprint(f"[green]✅ Completed {len(results):,} evaluations[/green]")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df


def aggregate_model_totals(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-model portfolio totals."""
    rprint("[cyan]Aggregating model totals...[/cyan]")
    
    agg = results_df.groupby('model_name').agg({
        'total_cost': ['sum', 'mean', 'std', 'count',
                       lambda x: np.quantile(x, 0.05),
                       lambda x: np.quantile(x, 0.50),
                       lambda x: np.quantile(x, 0.95)],
        'n_missing': 'sum'
    }).reset_index()
    
    agg.columns = [
        'model_name', 'portfolio_cost', 'mean_sku_cost', 'std_sku_cost', 'n_skus',
        'p05_sku', 'p50_sku', 'p95_sku', 'total_missing'
    ]
    
    # Sort by portfolio cost
    agg = agg.sort_values('portfolio_cost')
    
    return agg


def compute_selector(results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-SKU selector: pick model with lowest cost per SKU.
    
    Returns:
        (selector_summary, selector_map)
    """
    rprint("[cyan]Computing per-SKU selector...[/cyan]")
    
    # For each SKU, pick model with min cost
    selector_map = results_df.loc[
        results_df.groupby(['store', 'product'])['total_cost'].idxmin()
    ][['store', 'product', 'model_name', 'total_cost']].reset_index(drop=True)
    
    selector_summary = pd.DataFrame([{
        'model_name': 'SELECTOR',
        'portfolio_cost': selector_map['total_cost'].sum(),
        'n_skus': len(selector_map),
        'mean_sku_cost': selector_map['total_cost'].mean()
    }])
    
    rprint(f"[green]  Selector portfolio cost: {selector_summary['portfolio_cost'].iloc[0]:.2f}[/green]")
    
    return selector_summary, selector_map


def render_leaderboard(
    model_totals: pd.DataFrame,
    selector_summary: Optional[pd.DataFrame] = None
) -> str:
    """Render markdown leaderboard."""
    if selector_summary is not None:
        combined = pd.concat([model_totals, selector_summary], ignore_index=True)
    else:
        combined = model_totals
    
    combined = combined.sort_values('portfolio_cost')
    
    md_lines = []
    md_lines.append("# Sequential L=2 Leaderboard (H=12 epochs)")
    md_lines.append("")
    md_lines.append("Per-SKU sequential planning with exact realized costs at decision-affected periods.")
    md_lines.append("")
    
    # Table
    cols = ['model_name', 'portfolio_cost', 'n_skus', 'mean_sku_cost', 'p50_sku', 'total_missing']
    cols = [c for c in cols if c in combined.columns]
    
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    md_lines.append(header)
    md_lines.append(separator)
    
    for _, row in combined[cols].iterrows():
        values = []
        for col in cols:
            val = row[col]
            if pd.isna(val):
                values.append("N/A")
            elif col == 'model_name':
                values.append(str(val))
            elif col in ['n_skus']:
                values.append(str(int(val)))
            elif col in ['portfolio_cost', 'mean_sku_cost', 'p50_sku']:
                values.append(f"{val:.2f}")
            else:
                values.append(f"{val:.4f}")
        
        row_str = "| " + " | ".join(values) + " |"
        md_lines.append(row_str)
    
    return "\n".join(md_lines)


def run_full_sequential_eval(config: SequentialConfig):
    """Main entry point for sequential evaluation."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Run evaluation
    results_df = run_sequential_evaluation(config)
    
    # Save detailed results
    results_path = config.output_dir / f"sequential_results_{config.run_tag}.parquet"
    results_df.to_parquet(results_path, index=False)
    rprint(f"[green]✅ Saved results: {results_path}[/green]")
    
    # 2. Aggregate totals
    model_totals = aggregate_model_totals(results_df)
    totals_path = config.output_dir / f"model_totals_{config.run_tag}.parquet"
    model_totals.to_parquet(totals_path, index=False)
    rprint(f"[green]✅ Saved model totals: {totals_path}[/green]")
    
    # 3. Compute selector
    selector_summary, selector_map = compute_selector(results_df)
    selector_path = config.output_dir / f"selector_map_{config.run_tag}.parquet"
    selector_map.to_parquet(selector_path, index=False)
    rprint(f"[green]✅ Saved selector map: {selector_path}[/green]")
    
    # 4. Render leaderboard
    leaderboard_md = render_leaderboard(model_totals, selector_summary)
    leaderboard_path = config.output_dir / f"leaderboard_{config.run_tag}.md"
    with open(leaderboard_path, 'w') as f:
        f.write(leaderboard_md)
    rprint(f"[green]✅ Saved leaderboard: {leaderboard_path}[/green]")
    
    # Print summary
    rprint("\n" + "="*80)
    rprint(leaderboard_md)
    rprint("="*80 + "\n")
    
    rprint(f"[bold green]✅ Sequential Evaluation Complete![/bold green]")
    rprint(f"  Best model: {model_totals.iloc[0]['model_name']}")
    rprint(f"  Best cost: {model_totals.iloc[0]['portfolio_cost']:.2f}")
    rprint(f"  Selector cost: {selector_summary.iloc[0]['portfolio_cost']:.2f}")

