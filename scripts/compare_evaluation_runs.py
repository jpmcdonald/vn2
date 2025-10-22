#!/usr/bin/env python3
"""
Compare sequential backtest evaluation runs.

This script compares portfolio costs between two evaluation runs to validate
implementation consistency.

Usage:
    python scripts/compare_evaluation_runs.py

Requires:
    - Previous results in backups/ or models/results/
    - Current results in models/results/
"""

import pandas as pd
import sys
from pathlib import Path


def compare_runs(prev_results: dict, curr_results_path: Path):
    """
    Compare portfolio costs between previous and current runs.
    
    Args:
        prev_results: Dictionary of {model_name: portfolio_cost} from previous run
        curr_results_path: Path to current model_totals parquet file
    
    Returns:
        DataFrame with comparison results
    """
    # Load current results
    curr_totals = pd.read_parquet(curr_results_path)
    curr = dict(zip(curr_totals.model_name, curr_totals.portfolio_cost))
    
    # Add selector cost (from separate file)
    selector_path = curr_results_path.parent / curr_results_path.name.replace('model_totals', 'selector_map')
    if selector_path.exists():
        selector_df = pd.read_parquet(selector_path)
        curr['SELECTOR'] = selector_df.total_cost.sum()
    
    # Compare
    comparison = []
    for model in prev_results.keys():
        if model in curr:
            p = prev_results[model]
            c = curr[model]
            diff = c - p
            pct = (diff / p) * 100
            comparison.append({
                'model': model,
                'previous': p,
                'current': c,
                'diff': diff,
                'pct_change': pct
            })
    
    return pd.DataFrame(comparison)


def main():
    """Main entry point."""
    print('='*80)
    print('Sequential Backtest Evaluation Comparison')
    print('='*80)
    print()
    
    # Previous results (from seq12_v1)
    # These can be loaded from backup files or hardcoded
    prev = {
        'SELECTOR': 5593.00,
        'zinb': 8905.20,
        'slurp_bootstrap': 9769.80,
        'slurp_stockout_aware': 10049.40,
        'knn_profile': 10662.60,
        'lightgbm_quantile': 14092.60,
    }
    
    # Find most recent results
    results_dir = Path('models/results')
    model_totals_files = sorted(results_dir.glob('model_totals_seq12_*.parquet'))
    
    if not model_totals_files:
        print('ERROR: No model_totals files found in models/results/')
        sys.exit(1)
    
    curr_file = model_totals_files[-1]
    print(f'Comparing against: {curr_file.name}')
    print()
    
    # Compare
    comparison = compare_runs(prev, curr_file)
    
    # Display results
    print('Model Comparison (Portfolio Cost):')
    print('-'*80)
    print(f"{'Model':<25} {'Previous':>12} {'Current':>12} {'Diff':>12} {'% Change':>12}")
    print('-'*80)
    
    for _, row in comparison.iterrows():
        sign = '+' if row['diff'] > 0 else ''
        print(f"{row['model']:<25} {row['previous']:>12.2f} {row['current']:>12.2f} "
              f"{sign}{row['diff']:>11.2f} {sign}{row['pct_change']:>10.1f}%")
    
    print()
    print('Summary:')
    print('-'*80)
    
    # Overall statistics
    avg_pct_change = comparison['pct_change'].abs().mean()
    max_pct_change = comparison['pct_change'].abs().max()
    
    print(f'Average absolute % change: {avg_pct_change:.2f}%')
    print(f'Maximum absolute % change: {max_pct_change:.2f}%')
    print()
    
    if avg_pct_change < 5.0:
        print('✅ Results are highly consistent (avg change < 5%)')
    elif avg_pct_change < 10.0:
        print('⚠️  Results show moderate variation (5-10% change)')
    else:
        print('❌ Results show significant variation (>10% change)')
    
    # Save comparison
    output_path = results_dir / 'run_comparison.csv'
    comparison.to_csv(output_path, index=False)
    print(f'\nSaved comparison to: {output_path}')


if __name__ == '__main__':
    main()

