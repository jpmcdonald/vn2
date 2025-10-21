"""
W8 Evaluation: Exact per-SKU 8-fold costs with Q=0 fallback.

Computes discrete, realized h2 costs for the latest 8 chronological folds,
with integer Q decisions and Q=0 fallback for missing predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from joblib import Parallel, delayed
from rich import print as rprint

from vn2.analyze.sip_opt import Costs as SIPCosts, compute_realized_w2


@dataclass
class W8Config:
    """Configuration for W8 evaluation"""
    folds_path: Path
    demand_path: Path
    state_path: Path
    output_dir: Path
    run_tag: str = "v4full"
    n_jobs: int = 12
    holding_cost: float = 0.2
    shortage_cost: float = 1.0
    critical_fractile: float = 0.8333333333  # shortage / (shortage + holding)


def load_data(config: W8Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load folds, demand, and state data"""
    rprint(f"[cyan]Loading data...[/cyan]")
    
    folds_df = pd.read_parquet(config.folds_path)
    demand_df = pd.read_parquet(config.demand_path)
    state_df = pd.read_parquet(config.state_path)
    
    # Ensure state is indexed properly
    if not isinstance(state_df.index, pd.MultiIndex):
        if 'Store' in state_df.columns and 'Product' in state_df.columns:
            state_df = state_df.set_index(['Store', 'Product'])
    
    rprint(f"  Folds: {len(folds_df):,} rows")
    rprint(f"  Demand: {len(demand_df):,} rows")
    rprint(f"  State: {len(state_df):,} SKUs")
    
    return folds_df, demand_df, state_df


def identify_8_folds(folds_df: pd.DataFrame) -> List[int]:
    """
    Identify the latest 8 chronological fold origins (global, same for all SKUs).
    
    Returns:
        List of 8 fold_idx values in ascending order
    """
    all_folds = sorted(folds_df['fold_idx'].unique())
    
    if len(all_folds) < 8:
        raise ValueError(f"Need at least 8 folds, found {len(all_folds)}")
    
    # Take the last 8
    selected_folds = all_folds[-8:]
    
    rprint(f"[cyan]Selected 8 folds: {selected_folds}[/cyan]")
    
    return selected_folds


def get_sku_universe(folds_df: pd.DataFrame) -> List[Tuple[int, int]]:
    """Get list of all (store, product) SKUs"""
    skus = folds_df[['store', 'product']].drop_duplicates().values
    sku_list = [tuple(row) for row in skus]
    return sorted(sku_list)


def reconstruct_y_for_fold(
    store: int,
    product: int,
    fold_idx: int,
    demand_df: pd.DataFrame,
    holdout_weeks: int = 12
) -> Tuple[Optional[int], Optional[int]]:
    """
    Reconstruct actual demand (y1, y2) for a given fold.
    
    Fold logic matches prepare_train_test_split:
    split_idx = len(sku_df) - (holdout_weeks - fold_idx)
    test covers [split_idx, split_idx+2)
    
    Args:
        store: Store ID
        product: Product ID
        fold_idx: Fold index (0-based, 0=most recent, 11=earliest)
        demand_df: Demand dataframe with columns [Store, Product, week, sales]
        holdout_weeks: Number of holdout weeks (default 12, matching training)
    
    Returns:
        (y1, y2): Tuple of actual demands for h=1 and h=2, or (None, None) if not found
    """
    # Filter to SKU
    sku_data = demand_df[
        (demand_df['Store'] == store) & (demand_df['Product'] == product)
    ].sort_values('week').reset_index(drop=True)
    
    if len(sku_data) == 0:
        return None, None
    
    # Compute test start index (matches prepare_train_test_split)
    total_weeks = len(sku_data)
    split_idx = total_weeks - (holdout_weeks - fold_idx)
    test_start_idx = split_idx
    test_end_idx = split_idx + 2
    
    if test_start_idx < 0 or test_end_idx > len(sku_data):
        return None, None
    
    y1 = int(np.round(sku_data.iloc[test_start_idx]['sales']))
    y2 = int(np.round(sku_data.iloc[test_start_idx + 1]['sales']))
    
    return y1, y2


def get_initial_state_for_fold(
    store: int,
    product: int,
    fold_idx: int,
    state_df: pd.DataFrame
) -> Tuple[int, int, int]:
    """
    Get initial inventory state for a fold.
    
    For simplicity, we use the same initial state from state.parquet for all folds.
    In a full simulation, this would track inventory evolution, but for evaluation
    we use a consistent baseline.
    
    Args:
        store: Store ID
        product: Product ID
        fold_idx: Fold index
        state_df: State dataframe indexed by (Store, Product)
    
    Returns:
        (I0, Q1, Q2): Initial on-hand, intransit_1, intransit_2
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


def compute_sku_fold_cost(
    store: int,
    product: int,
    fold_idx: int,
    model_name: str,
    folds_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    state_df: pd.DataFrame,
    costs: SIPCosts,
    holdout_weeks: int = 8
) -> Dict:
    """
    Compute h2 realized cost for a single (model, SKU, fold) combination.
    
    If the model has a prediction row, use sip_order_qty and recompute cost.
    If missing, use Q=0 and compute cost (penalty).
    
    Returns:
        Dict with keys: model_name, store, product, fold_idx, Q, cost_w2, is_missing
    """
    # Look for existing row
    row_mask = (
        (folds_df['model_name'] == model_name) &
        (folds_df['store'] == store) &
        (folds_df['product'] == product) &
        (folds_df['fold_idx'] == fold_idx)
    )
    
    existing_rows = folds_df[row_mask]
    
    # Reconstruct actual demand
    y1, y2 = reconstruct_y_for_fold(store, product, fold_idx, demand_df, holdout_weeks)
    
    if y1 is None or y2 is None:
        # Cannot reconstruct demand; skip this fold
        return None
    
    # Get initial state
    I0, Q1, Q2 = get_initial_state_for_fold(store, product, fold_idx, state_df)
    
    if len(existing_rows) > 0:
        # Use existing prediction and stored cost
        row = existing_rows.iloc[0]
        Q = int(row['sip_order_qty']) if pd.notna(row['sip_order_qty']) else 0
        cost_w2 = float(row['sip_realized_cost_w2']) if pd.notna(row['sip_realized_cost_w2']) else 0.0
        is_missing = False
    else:
        # Missing prediction: Q=0 fallback, compute cost
        Q = 0
        cost_w2 = compute_realized_w2(Q, I0, Q1, Q2, y1, y2, costs)
        is_missing = True
    
    return {
        'model_name': model_name,
        'store': store,
        'product': product,
        'fold_idx': fold_idx,
        'Q': Q,
        'cost_w2': cost_w2,
        'is_missing': is_missing,
        'y1': y1,
        'y2': y2,
        'I0': I0,
        'Q1': Q1,
        'Q2': Q2
    }


def compute_per_sku_totals(
    folds_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    state_df: pd.DataFrame,
    selected_folds: List[int],
    sku_list: List[Tuple[int, int]],
    costs: SIPCosts,
    n_jobs: int = 12,
    holdout_weeks: int = 12
) -> pd.DataFrame:
    """
    Compute per-SKU 8-fold totals for all models.
    
    Returns:
        DataFrame with columns: model_name, store, product, total_cost_w8, 
                                folds_covered, folds_missing
    """
    rprint(f"[cyan]Computing per-SKU 8-fold totals...[/cyan]")
    rprint(f"  SKUs: {len(sku_list)}")
    rprint(f"  Folds: {len(selected_folds)}")
    rprint(f"  Models: {folds_df['model_name'].nunique()}")
    rprint(f"  Parallel jobs: {n_jobs}")
    
    models = folds_df['model_name'].unique()
    
    # Generate all tasks
    tasks = []
    for model in models:
        for store, product in sku_list:
            for fold_idx in selected_folds:
                tasks.append((store, product, fold_idx, model))
    
    rprint(f"  Total tasks: {len(tasks):,}")
    
    # Parallel computation
    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
        delayed(compute_sku_fold_cost)(
            store, product, fold_idx, model,
            folds_df, demand_df, state_df, costs, holdout_weeks
        ) for store, product, fold_idx, model in tasks
    )
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    rprint(f"[green]  Computed {len(results):,} fold-level costs[/green]")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Aggregate per (model, SKU)
    agg = results_df.groupby(['model_name', 'store', 'product']).agg({
        'cost_w2': 'sum',
        'is_missing': 'sum',
        'fold_idx': 'count'
    }).reset_index()
    
    agg.rename(columns={
        'cost_w2': 'total_cost_w8',
        'is_missing': 'folds_missing',
        'fold_idx': 'folds_covered'
    }, inplace=True)
    
    # Sanity check: folds_covered should be 8 for all
    if not (agg['folds_covered'] == 8).all():
        rprint("[yellow]⚠️  Warning: Some SKUs have != 8 folds covered[/yellow]")
    
    return agg


def aggregate_model_totals(per_sku_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-SKU totals to model-level portfolio totals.
    
    Returns:
        DataFrame with columns: model_name, total_cost_w8, n_skus, 
                                p05, p50, p95 (quantiles of per-SKU totals)
    """
    rprint(f"[cyan]Aggregating model portfolio totals...[/cyan]")
    
    agg = per_sku_df.groupby('model_name').agg({
        'total_cost_w8': ['sum', 'count', lambda x: np.quantile(x, 0.05), 
                          lambda x: np.quantile(x, 0.50), lambda x: np.quantile(x, 0.95)]
    }).reset_index()
    
    agg.columns = ['model_name', 'total_cost_w8', 'n_skus', 'p05_sku', 'p50_sku', 'p95_sku']
    
    # Sort by total cost ascending
    agg = agg.sort_values('total_cost_w8')
    
    return agg


def compute_metrics_on_folds(
    folds_df: pd.DataFrame,
    selected_folds: List[int],
    critical_fractile: float
) -> pd.DataFrame:
    """
    Compute accuracy and density metrics on the same fold universe.
    
    Metrics are computed only on rows where predictions exist (no Q=0 fallback).
    
    Returns:
        DataFrame with model-level metrics: MAPE, MASE, BIAS, MAE, pinball, CRPS,
                                            coverage, width, CF metrics
    """
    rprint(f"[cyan]Computing metrics on fold universe...[/cyan]")
    
    # Filter to selected folds
    df = folds_df[folds_df['fold_idx'].isin(selected_folds)].copy()
    
    rprint(f"  Rows with predictions: {len(df):,}")
    
    # Aggregate by model
    metrics = []
    
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        
        metric_row = {'model_name': model}
        
        # Accuracy metrics
        for col in ['mae', 'mape', 'mase', 'bias']:
            if col in model_df.columns:
                metric_row[col] = model_df[col].mean()
        
        # Density metrics
        for col in ['pinball_loss', 'crps']:
            if col in model_df.columns:
                metric_row[col] = model_df[col].mean()
        
        # Coverage and width at different levels
        for level in [80, 90, 95]:
            cov_col = f'coverage_{level}'
            width_col = f'width_{level}'
            if cov_col in model_df.columns:
                metric_row[cov_col] = model_df[cov_col].mean()
            if width_col in model_df.columns:
                metric_row[width_col] = model_df[width_col].mean()
        
        # CF metrics (h1 and h2)
        for h in [1, 2]:
            for metric in ['pinball_cf', 'hit_cf', 'local_width']:
                col = f'{metric}_h{h}'
                if col in model_df.columns:
                    metric_row[col] = model_df[col].mean()
        
        metrics.append(metric_row)
    
    metrics_df = pd.DataFrame(metrics)
    
    return metrics_df


def compute_per_sku_selector(
    per_sku_df: pd.DataFrame,
    metrics_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-SKU selector: for each SKU, pick the model with lowest total_cost_w8.
    
    Tie-breakers: lower pinball_cf_h2, then lower crps.
    
    Returns:
        (selector_totals_df, selector_map_df)
    """
    rprint(f"[cyan]Computing per-SKU selector...[/cyan]")
    
    # Merge with metrics for tie-breaking
    merged = per_sku_df.merge(
        metrics_df[['model_name', 'pinball_cf_h2', 'crps']], 
        on='model_name', 
        how='left'
    )
    
    # Sort by cost, then tie-breakers
    merged = merged.sort_values(
        ['store', 'product', 'total_cost_w8', 'pinball_cf_h2', 'crps']
    )
    
    # Select best per SKU
    selector_map = merged.groupby(['store', 'product']).first().reset_index()
    selector_map = selector_map[['store', 'product', 'model_name', 'total_cost_w8']]
    
    # Aggregate selector total
    selector_total = selector_map['total_cost_w8'].sum()
    
    selector_totals = pd.DataFrame([{
        'model_name': 'selector',
        'total_cost_w8': selector_total,
        'n_skus': len(selector_map)
    }])
    
    rprint(f"[green]  Selector total: {selector_total:.2f}[/green]")
    
    return selector_totals, selector_map


def render_leaderboard(
    model_totals_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    selector_totals_df: Optional[pd.DataFrame] = None
) -> str:
    """
    Render Markdown leaderboard table.
    
    Returns:
        Markdown string
    """
    # Merge totals with metrics
    merged = model_totals_df.merge(metrics_df, on='model_name', how='left')
    
    # Append selector if provided
    if selector_totals_df is not None:
        # Merge selector with metrics (will be NaN for most metrics)
        selector_with_metrics = selector_totals_df.merge(
            metrics_df, on='model_name', how='left'
        )
        merged = pd.concat([merged, selector_with_metrics], ignore_index=True)
    
    # Sort by total cost
    merged = merged.sort_values('total_cost_w8')
    
    # Select display columns
    display_cols = ['model_name', 'total_cost_w8', 'n_skus']
    
    # Add metrics if available
    for col in ['mape', 'mase', 'bias', 'mae', 'pinball_loss', 'crps', 
                'coverage_80', 'coverage_90', 'coverage_95',
                'width_80', 'width_90', 'width_95',
                'pinball_cf_h2', 'hit_cf_h2', 'local_width_h2']:
        if col in merged.columns:
            display_cols.append(col)
    
    display_cols = [c for c in display_cols if c in merged.columns]
    
    # Format as Markdown
    md_lines = []
    md_lines.append("# W8 Leaderboard: 8-Fold Exact Per-SKU Costs")
    md_lines.append("")
    md_lines.append("Costs computed as discrete h2 realized costs with Q=0 fallback for missing predictions.")
    md_lines.append("")
    
    # Header
    header = "| " + " | ".join(display_cols) + " |"
    separator = "| " + " | ".join(["---"] * len(display_cols)) + " |"
    md_lines.append(header)
    md_lines.append(separator)
    
    # Rows
    for _, row in merged[display_cols].iterrows():
        values = []
        for col in display_cols:
            val = row[col]
            if pd.isna(val):
                values.append("N/A")
            elif col == 'model_name':
                values.append(str(val))
            elif col in ['n_skus']:
                values.append(str(int(val)))
            elif col in ['total_cost_w8']:
                values.append(f"{val:.2f}")
            else:
                values.append(f"{val:.4f}")
        
        row_str = "| " + " | ".join(values) + " |"
        md_lines.append(row_str)
    
    return "\n".join(md_lines)


def run_w8_eval(config: W8Config):
    """Main entry point for W8 evaluation"""
    rprint(f"[bold blue]🚀 W8 Evaluation: {config.run_tag}[/bold blue]")
    rprint(f"  Folds: {config.folds_path}")
    rprint(f"  Output: {config.output_dir}")
    rprint(f"  Costs: holding={config.holding_cost}, shortage={config.shortage_cost}")
    rprint(f"  Critical fractile: {config.critical_fractile:.6f}")
    
    # Load data
    folds_df, demand_df, state_df = load_data(config)
    
    # Identify 8 folds
    selected_folds = identify_8_folds(folds_df)
    
    # Get SKU universe
    sku_list = get_sku_universe(folds_df)
    rprint(f"[cyan]SKU universe: {len(sku_list)} SKUs[/cyan]")
    
    # Setup costs
    costs = SIPCosts(holding=config.holding_cost, shortage=config.shortage_cost)
    
    # 1. Compute per-SKU totals
    per_sku_df = compute_per_sku_totals(
        folds_df, demand_df, state_df, selected_folds, sku_list, costs, 
        n_jobs=config.n_jobs, holdout_weeks=12
    )
    
    # Save per-SKU totals
    per_sku_path = config.output_dir / f"per_sku_totals_w8_{config.run_tag}.parquet"
    per_sku_df.to_parquet(per_sku_path, index=False)
    rprint(f"[green]✅ Saved per-SKU totals: {per_sku_path}[/green]")
    
    # 2. Aggregate model totals
    model_totals_df = aggregate_model_totals(per_sku_df)
    
    # Save model totals
    model_totals_path = config.output_dir / f"model_totals_w8_{config.run_tag}.parquet"
    model_totals_df.to_parquet(model_totals_path, index=False)
    rprint(f"[green]✅ Saved model totals: {model_totals_path}[/green]")
    
    # 3. Compute metrics
    metrics_df = compute_metrics_on_folds(folds_df, selected_folds, config.critical_fractile)
    
    # Save metrics
    metrics_path = config.output_dir / f"metrics_w8_{config.run_tag}.parquet"
    metrics_df.to_parquet(metrics_path, index=False)
    rprint(f"[green]✅ Saved metrics: {metrics_path}[/green]")
    
    # 4. Compute selector
    selector_totals_df, selector_map_df = compute_per_sku_selector(per_sku_df, metrics_df)
    
    # Save selector
    selector_totals_path = config.output_dir / f"selector_totals_w8_{config.run_tag}.parquet"
    selector_totals_df.to_parquet(selector_totals_path, index=False)
    rprint(f"[green]✅ Saved selector totals: {selector_totals_path}[/green]")
    
    selector_map_path = config.output_dir / f"selector_map_w8_{config.run_tag}.parquet"
    selector_map_df.to_parquet(selector_map_path, index=False)
    rprint(f"[green]✅ Saved selector map: {selector_map_path}[/green]")
    
    # 5. Render leaderboard
    leaderboard_md = render_leaderboard(model_totals_df, metrics_df, selector_totals_df)
    
    # Save leaderboard
    leaderboard_path = config.output_dir / f"leaderboard_w8_{config.run_tag}.md"
    with open(leaderboard_path, 'w') as f:
        f.write(leaderboard_md)
    rprint(f"[green]✅ Saved leaderboard: {leaderboard_path}[/green]")
    
    # Print leaderboard to console
    rprint("\n" + "="*80)
    rprint(leaderboard_md)
    rprint("="*80 + "\n")
    
    # Print summary
    rprint(f"[bold green]✅ W8 Evaluation Complete![/bold green]")
    rprint(f"  Best model: {model_totals_df.iloc[0]['model_name']}")
    rprint(f"  Best cost: {model_totals_df.iloc[0]['total_cost_w8']:.2f}")
    rprint(f"  Selector cost: {selector_totals_df.iloc[0]['total_cost_w8']:.2f}")

