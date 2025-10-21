"""
Per-SKU model selector: Choose best model for each SKU based on newsvendor metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal, Optional, Dict, List


def select_per_sku_from_folds(
    eval_folds_path: Path,
    cost_col: str = 'sip_realized_cost_w2',
    fold_window: int = 8,
    tie_breakers: List[str] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Per-SKU selector using last-N decision-affected folds with CF tie-breaking.
    
    For each SKU, aggregate cost across last-N folds per model, rank by cost + CF metrics.
    
    Args:
        eval_folds_path: Path to eval_folds parquet (e.g., eval_folds_v4_sip.parquet)
        cost_col: Cost column to minimize (default: sip_realized_cost_w2)
        fold_window: Number of last folds to use per SKU (default: 8)
        tie_breakers: List of CF metrics for tie-breaking (default: pinball_cf_h2, hit_cf_h2, local_width_h2)
        output_path: Optional path to save selector map
    
    Returns:
        DataFrame with (store, product, selected_model, total_cost_w8, model_rank)
    """
    if tie_breakers is None:
        tie_breakers = ['pinball_cf_h2', 'hit_cf_h2', 'local_width_h2']
    
    df = pd.read_parquet(eval_folds_path)
    
    if cost_col not in df.columns:
        raise ValueError(f"Cost column {cost_col} not found. Available: {df.columns.tolist()}")
    
    # Select last-N folds per SKU
    df['_fold_rank'] = df.groupby(['store', 'product'])['fold_idx'].rank(method='first', ascending=False)
    dfN = df[df['_fold_rank'] <= fold_window].copy()
    
    # Primary score: sum of cost across last-N folds
    agg = (dfN.groupby(['store', 'product', 'model_name'], as_index=False)[cost_col]
             .sum().rename(columns={cost_col: 'total_cost_w8'}))

    # Coverage per SKU per model across last-N folds
    cov = (dfN.groupby(['store','product','model_name'])['fold_idx']
              .nunique().reset_index(name='n_folds'))
    agg = agg.merge(cov, on=['store','product','model_name'], how='left')
    
    # Tie-breakers: aggregate CF metrics (mean across folds)
    for tb in tie_breakers:
        if tb in dfN.columns:
            tb_agg = dfN.groupby(['store', 'product', 'model_name'])[tb].mean().reset_index()
            tb_agg = tb_agg.rename(columns={tb: f'{tb}_mean'})
            agg = agg.merge(tb_agg, on=['store', 'product', 'model_name'], how='left')
    
    # Build SKU-level ranking
    def rank_one(g):
        # Enforce full coverage for this SKU
        g = g[g['n_folds'] >= fold_window].copy()
        if g.empty:
            return g.assign(model_rank=pd.Series(dtype=int))
        # Sort by total_cost_w8, then tie-breakers in order
        sort_cols = ['total_cost_w8']
        ascending = [True]
        
        for tb in tie_breakers:
            col = f'{tb}_mean'
            if col in g.columns:
                sort_cols.append(col)
                # Smaller is better for pinball/local_width; larger is better for hit_cf
                ascending.append(False if 'hit_cf' in tb else True)
        
        return g.sort_values(sort_cols, ascending=ascending).assign(model_rank=np.arange(1, len(g)+1))
    
    ranked = agg.groupby(['store', 'product'], group_keys=False).apply(rank_one)
    
    # Top-1 per SKU
    top1 = ranked[ranked['model_rank'] == 1][['store', 'product', 'model_name', 'total_cost_w8']].rename(
        columns={'model_name': 'selected_model'})
    
    # Save full ranking for coverage fallback
    if output_path:
        ranked.to_parquet(output_path.with_name(output_path.stem + '_ranking.parquet'), index=False)
        top1.to_parquet(output_path, index=False)
        print(f"\n✅ Saved selector map to {output_path}")
        print(f"✅ Saved full ranking to {output_path.with_name(output_path.stem + '_ranking.parquet')}")
    
    # Summary stats
    print("="*70)
    print(f"PER-SKU SELECTOR (last {fold_window} folds, CF tie-breaking)")
    print("="*70)
    num_skus = agg[['store', 'product']].drop_duplicates().shape[0]
    print(f"\nTotal SKUs: {num_skus}")
    
    # Count selections
    sel_counts = top1['selected_model'].value_counts().sort_values(ascending=False)
    print(f"\nModel selection counts:")
    for model, count in sel_counts.items():
        pct = 100.0 * count / num_skus
        print(f"  {model:25s}: {count:3d} / {num_skus} ({pct:5.1f}%)")
    
    return top1


def select_best_models(
    eval_agg_path: Path,
    metric: Literal['pinball_cf_h1', 'crps', 'asymmetric_loss_h1', 'expected_cost'] = 'pinball_cf_h1',
    output_path: Path = None
) -> pd.DataFrame:
    """
    For each SKU, select the best model based on specified metric.
    
    Args:
        eval_agg_path: Path to eval_agg parquet file
        metric: Metric to optimize ('pinball_cf_h1', 'crps', 'asymmetric_loss_h1', 'expected_cost')
        output_path: Optional path to save results
    
    Returns:
        DataFrame with (store, product, best_model, metric_value, n_candidates)
    """
    # Load aggregated results
    df = pd.read_parquet(eval_agg_path)
    
    if metric not in df.columns:
        raise ValueError(f"Metric {metric} not found in data. Available: {df.columns.tolist()}")
    
    # For each SKU, find model with minimum metric value
    best_models = []
    
    for (store, product), group in df.groupby(['store', 'product']):
        # Filter to models with valid metric
        valid = group[group[metric].notna()].copy()
        
        if len(valid) == 0:
            continue
        
        # Find best (minimum metric)
        best_idx = valid[metric].idxmin()
        best_row = valid.loc[best_idx]
        
        best_models.append({
            'store': store,
            'product': product,
            'best_model': best_row['model_name'],
            metric: best_row[metric],
            'n_candidates': len(valid),
            'mae': best_row.get('mae', np.nan),
            'coverage_90': best_row.get('coverage_90', np.nan),
            'service_level': best_row.get('service_level', np.nan)
        })
    
    result_df = pd.DataFrame(best_models)
    
    # Summary statistics
    print("="*70)
    print(f"PER-SKU MODEL SELECTION (metric: {metric})")
    print("="*70)
    print(f"\nTotal SKUs: {len(result_df)}")
    print(f"\nModel selection counts:")
    counts = result_df['best_model'].value_counts()
    for model, count in counts.items():
        pct = 100 * count / len(result_df)
        print(f"  {model:20s}: {count:3d} / {len(result_df)} ({pct:5.1f}%)")
    
    # Save if requested
    if output_path:
        result_df.to_parquet(output_path)
        print(f"\n✅ Saved to {output_path}")
    
    return result_df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Select best model per SKU")
    parser.add_argument('--eval-agg', type=Path, default=Path('models/results/eval_agg_v3.parquet'))
    parser.add_argument('--metric', type=str, default='pinball_cf_h1',
                       choices=['pinball_cf_h1', 'crps', 'asymmetric_loss_h1', 'expected_cost'])
    parser.add_argument('--output', type=Path, default=None)
    
    args = parser.parse_args()
    
    select_best_models(args.eval_agg, args.metric, args.output)


if __name__ == '__main__':
    main()

