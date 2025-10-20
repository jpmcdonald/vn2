"""
Per-SKU model selector: Choose best model for each SKU based on newsvendor metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal


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

