#!/usr/bin/env python3
"""
Compare actual Week 1 submission with selector recommendations.

This script validates the selector's recommendations by comparing them against
the actual orders that were submitted for Week 1.

Usage:
    python scripts/compare_week1_orders.py

Requires:
    - Actual submission CSV in data/submissions/
    - Sequential backtest results in models/results/
    - Selector map in models/results/

Output:
    - Detailed comparison CSV in models/results/
    - Summary statistics printed to console
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path


def load_actual_submission(submission_path: Path, week_col: str = '2024-04-15') -> pd.DataFrame:
    """
    Load actual Week 1 submission.
    
    Args:
        submission_path: Path to submission CSV
        week_col: Column name for Week 1 orders
    
    Returns:
        DataFrame with Store, Product, actual_order
    """
    actual = pd.read_csv(submission_path)
    actual_week1 = actual[['Store', 'Product', week_col]].rename(
        columns={week_col: 'actual_order'}
    )
    return actual_week1


def get_selector_orders(results_path: Path, selector_path: Path) -> pd.DataFrame:
    """
    Extract Week 1 orders from selector recommendations.
    
    Args:
        results_path: Path to sequential_results parquet
        selector_path: Path to selector_map parquet
    
    Returns:
        DataFrame with Store, Product, selector_order, selected_model
    """
    # Load results and selector map
    results = pd.read_parquet(results_path)
    selector_map = pd.read_parquet(selector_path)
    
    # Get week 1 orders from selector models
    selector_orders = []
    for _, row in selector_map.iterrows():
        store, product, model = row['store'], row['product'], row['model_name']
        
        # Find this SKU's result for this model
        mask = (results['store'] == store) & \
               (results['product'] == product) & \
               (results['model_name'] == model)
        
        if mask.any():
            orders = results[mask].iloc[0]['orders']
            week1_order = orders[0] if len(orders) > 0 else 0
            selector_orders.append({
                'Store': store,
                'Product': product,
                'selector_order': week1_order,
                'selected_model': model
            })
    
    return pd.DataFrame(selector_orders)


def compute_agreement_metrics(comparison: pd.DataFrame) -> dict:
    """
    Compute agreement metrics between actual and selector orders.
    
    Args:
        comparison: DataFrame with actual_order and selector_order columns
    
    Returns:
        Dictionary of metrics
    """
    comparison['diff'] = comparison['selector_order'] - comparison['actual_order']
    comparison['abs_diff'] = comparison['diff'].abs()
    
    total = len(comparison)
    
    return {
        'total_skus': total,
        'exact_matches': (comparison['diff'] == 0).sum(),
        'within_1': (comparison['abs_diff'] <= 1).sum(),
        'within_5': (comparison['abs_diff'] <= 5).sum(),
        'within_10': (comparison['abs_diff'] <= 10).sum(),
        'mean_diff': comparison['diff'].mean(),
        'median_diff': comparison['diff'].median(),
        'mean_abs_diff': comparison['abs_diff'].mean(),
        'median_abs_diff': comparison['abs_diff'].median(),
        'actual_mean': comparison['actual_order'].mean(),
        'actual_median': comparison['actual_order'].median(),
        'selector_mean': comparison['selector_order'].mean(),
        'selector_median': comparison['selector_order'].median(),
    }


def main():
    """Main entry point."""
    print('='*80)
    print('Week 1 Order Comparison: Actual Submission vs Selector')
    print('='*80)
    print()
    
    # Paths
    submission_path = Path('data/submissions/orders_selector_wide_2024-04-15.csv')
    results_dir = Path('models/results')
    
    # Find most recent results
    results_files = sorted(results_dir.glob('sequential_results_seq12_*.parquet'))
    selector_files = sorted(results_dir.glob('selector_map_seq12_*.parquet'))
    
    if not results_files or not selector_files:
        print('ERROR: Results files not found in models/results/')
        sys.exit(1)
    
    results_path = results_files[-1]
    selector_path = selector_files[-1]
    
    print(f'Using results: {results_path.name}')
    print(f'Using selector: {selector_path.name}')
    print(f'Using submission: {submission_path.name}')
    print()
    
    # Load data
    actual = load_actual_submission(submission_path)
    selector = get_selector_orders(results_path, selector_path)
    
    # Merge
    comparison = actual.merge(selector, on=['Store', 'Product'], how='inner')
    
    print(f'Total SKUs compared: {len(comparison)}')
    print()
    
    # Compute metrics
    metrics = compute_agreement_metrics(comparison)
    
    # Display statistics
    print('Order Statistics:')
    print('-'*80)
    print(f"Actual orders - Mean: {metrics['actual_mean']:.2f}, "
          f"Median: {metrics['actual_median']:.2f}")
    print(f"Selector orders - Mean: {metrics['selector_mean']:.2f}, "
          f"Median: {metrics['selector_median']:.2f}")
    print()
    print(f"Difference - Mean: {metrics['mean_diff']:.2f}, "
          f"Median: {metrics['median_diff']:.2f}")
    print(f"Absolute difference - Mean: {metrics['mean_abs_diff']:.2f}, "
          f"Median: {metrics['median_abs_diff']:.2f}")
    print()
    
    # Agreement metrics
    print('Agreement Metrics:')
    print('-'*80)
    print(f"Exact matches: {metrics['exact_matches']} "
          f"({metrics['exact_matches']/metrics['total_skus']*100:.1f}%)")
    print(f"Within ±1 unit: {metrics['within_1']} "
          f"({metrics['within_1']/metrics['total_skus']*100:.1f}%)")
    print(f"Within ±5 units: {metrics['within_5']} "
          f"({metrics['within_5']/metrics['total_skus']*100:.1f}%)")
    print(f"Within ±10 units: {metrics['within_10']} "
          f"({metrics['within_10']/metrics['total_skus']*100:.1f}%)")
    print()
    
    # Top differences
    print('Top 10 Largest Differences (Selector - Actual):')
    print('-'*80)
    top_diffs = comparison.nlargest(10, 'abs_diff')[
        ['Store', 'Product', 'actual_order', 'selector_order', 'diff', 'selected_model']
    ]
    print(top_diffs.to_string(index=False))
    print()
    
    # Assessment
    exact_pct = metrics['exact_matches'] / metrics['total_skus'] * 100
    within1_pct = metrics['within_1'] / metrics['total_skus'] * 100
    
    print('Assessment:')
    print('-'*80)
    if exact_pct >= 70 and within1_pct >= 85:
        print('✅ Excellent agreement - selector is highly aligned with actual submission')
    elif exact_pct >= 50 and within1_pct >= 70:
        print('⚠️  Good agreement - selector shows reasonable alignment')
    else:
        print('❌ Poor agreement - investigate differences')
    print()
    
    # Save comparison
    output_path = results_dir / 'week1_order_comparison.csv'
    comparison.to_csv(output_path, index=False)
    print(f'Saved detailed comparison to: {output_path}')


if __name__ == '__main__':
    main()

