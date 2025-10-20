"""
Per-SKU model selector: Choose best model for each SKU based on newsvendor metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal, Optional, Dict


def select_per_sku_from_folds(
    eval_folds_path: Path,
    cost_col: str = 'sip_realized_cost_w2',
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Per-SKU selector using fold-level costs (tie-aware).
    
    For each SKU, aggregate cost across folds per model, find min, handle ties.
    
    Args:
        eval_folds_path: Path to eval_folds parquet (e.g., eval_folds_v4_sip.parquet)
        cost_col: Cost column to minimize (default: sip_realized_cost_w2)
        output_path: Optional path to save selector map
    
    Returns:
        DataFrame with (store, product, best_model, total_cost, n_ties)
    """
    df = pd.read_parquet(eval_folds_path)
    
    if cost_col not in df.columns:
        raise ValueError(f"Cost column {cost_col} not found. Available: {df.columns.tolist()}")
    
    # Aggregate cost per SKU per model across folds
    agg = (df.groupby(['store', 'product', 'model_name'], as_index=False)[cost_col]
             .sum().rename(columns={cost_col: 'total_cost'}))
    
    # Find min per SKU
    min_cost = agg.groupby(['store', 'product'])['total_cost'].transform('min')
    champions = agg[agg['total_cost'] == min_cost].copy()
    
    # Count ties
    n_ties = champions.groupby(['store', 'product'])['model_name'].transform('count')
    champions['n_ties'] = n_ties
    champions['win_share'] = 1.0 / n_ties
    
    # For single-champion SKUs, keep one row; for ties, keep all with win_share
    result = champions.copy()
    
    print("="*70)
    print(f"PER-SKU SELECTOR (from folds, cost_col={cost_col})")
    print("="*70)
    num_skus = agg[['store', 'product']].drop_duplicates().shape[0]
    print(f"\nTotal SKUs: {num_skus}")
    print(f"SKUs with ties: {(n_ties > 1).sum()} / {len(champions)}")
    
    # Aggregate win shares
    win_counts = champions.groupby('model_name')['win_share'].sum().sort_values(ascending=False)
    print(f"\nModel win shares (tie-adjusted):")
    for model, share in win_counts.items():
        pct = 100.0 * share / num_skus
        print(f"  {model:25s}: {share:6.2f} / {num_skus} ({pct:5.1f}%)")
    
    if output_path:
        result.to_parquet(output_path, index=False)
        print(f"\n✅ Saved to {output_path}")
    
    return result


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

