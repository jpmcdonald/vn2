#!/usr/bin/env python3
"""
Identify worst-performing SKUs from Week 4 analysis for targeted fixes.

This script analyzes the Week 4 expected vs realized cost results to find
SKUs with the largest cost differences, focusing on those that could benefit
from improved forecasting models or bias corrections.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

def identify_problem_skus(results_path: Path, top_n: int = 20) -> pd.DataFrame:
    """
    Identify the worst-performing SKUs from Week 4 analysis.
    
    Args:
        results_path: Path to Week 4 expected vs realized results CSV
        top_n: Number of worst SKUs to return
    
    Returns:
        DataFrame with worst-performing SKUs and their issues
    """
    # Load Week 4 results
    df = pd.read_csv(results_path)
    
    # Calculate severity metrics
    df['cost_ratio'] = df['realized_cost'] / np.maximum(df['expected_cost'], 0.01)
    df['absolute_error'] = np.abs(df['cost_difference'])
    df['severity_score'] = df['cost_difference'] * np.log1p(df['realized_cost'])
    
    # Identify different types of problems
    df['zero_prediction'] = (df['expected_cost'] < 0.01) & (df['realized_cost'] > 1.0)
    df['massive_underestimate'] = df['cost_ratio'] > 5.0
    df['outside_ci'] = ~df['within_ci']
    
    # Sort by severity (largest positive cost differences first)
    problem_skus = df[df['cost_difference'] > 0].nlargest(top_n, 'severity_score')
    
    # Add problem categorization
    def categorize_problem(row):
        issues = []
        if row['zero_prediction']:
            issues.append('ZERO_PREDICTION')
        if row['massive_underestimate']:
            issues.append('MASSIVE_UNDERESTIMATE') 
        if row['outside_ci']:
            issues.append('OUTSIDE_CI')
        if row['realized_cost'] > 20:
            issues.append('HIGH_COST')
        if row['shortage_cost'] > 0 and row['holding_cost'] == 0:
            issues.append('PURE_SHORTAGE')
        return ','.join(issues) if issues else 'MODERATE_ERROR'
    
    problem_skus['problem_type'] = problem_skus.apply(categorize_problem, axis=1)
    
    return problem_skus[['store', 'product', 'model', 'expected_cost', 'realized_cost', 
                        'cost_difference', 'cost_ratio', 'severity_score', 'problem_type',
                        'actual_demand', 'week4_start_inventory', 'within_ci']]


def analyze_model_failures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze which models are failing most frequently.
    
    Args:
        df: Full Week 4 results DataFrame
    
    Returns:
        DataFrame with model failure statistics
    """
    model_stats = []
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        stats = {
            'model': model,
            'n_skus': len(model_df),
            'mean_cost_diff': model_df['cost_difference'].mean(),
            'median_cost_diff': model_df['cost_difference'].median(),
            'pct_zero_predictions': (model_df['expected_cost'] < 0.01).mean() * 100,
            'pct_outside_ci': (~model_df['within_ci']).mean() * 100,
            'pct_massive_underestimate': (model_df['realized_cost'] / np.maximum(model_df['expected_cost'], 0.01) > 5.0).mean() * 100,
            'total_cost_error': model_df['cost_difference'].sum()
        }
        model_stats.append(stats)
    
    return pd.DataFrame(model_stats).sort_values('total_cost_error', ascending=False)


def main():
    """Main entry point."""
    results_path = Path('models/results/week4_expected_vs_realized.csv')
    
    if not results_path.exists():
        print(f"Error: {results_path} not found. Run Week 4 analysis first.")
        return
    
    print('='*80)
    print('WORST-PERFORMING SKUS ANALYSIS')
    print('='*80)
    print()
    
    # Load full results
    df = pd.read_csv(results_path)
    
    # Get top 20 problem SKUs
    problem_skus = identify_problem_skus(results_path, top_n=20)
    
    print('TOP 20 WORST-PERFORMING SKUS:')
    print('-'*80)
    print(problem_skus.to_string(index=False))
    print()
    
    # Model failure analysis
    model_failures = analyze_model_failures(df)
    
    print('MODEL FAILURE ANALYSIS:')
    print('-'*80)
    print(model_failures.to_string(index=False))
    print()
    
    # Summary statistics
    print('SUMMARY STATISTICS:')
    print('-'*80)
    total_skus = len(df)
    problem_skus_count = len(df[df['cost_difference'] > 5.0])
    zero_pred_count = len(df[(df['expected_cost'] < 0.01) & (df['realized_cost'] > 1.0)])
    outside_ci_count = len(df[~df['within_ci']])
    
    print(f'Total SKUs: {total_skus}')
    print(f'SKUs with cost difference > 5: {problem_skus_count} ({problem_skus_count/total_skus*100:.1f}%)')
    print(f'SKUs with zero predictions but realized cost > 1: {zero_pred_count} ({zero_pred_count/total_skus*100:.1f}%)')
    print(f'SKUs outside 90% CI: {outside_ci_count} ({outside_ci_count/total_skus*100:.1f}%)')
    print()
    
    # Save results
    output_path = Path('models/results/problem_skus_analysis.csv')
    problem_skus.to_csv(output_path, index=False)
    
    model_output_path = Path('models/results/model_failure_analysis.csv') 
    model_failures.to_csv(model_output_path, index=False)
    
    print(f'Saved problem SKUs to: {output_path}')
    print(f'Saved model analysis to: {model_output_path}')


if __name__ == '__main__':
    main()
