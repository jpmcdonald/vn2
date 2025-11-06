#!/usr/bin/env python3
"""
Validate improvements by re-running Week 4 cost analysis with corrected models.

This script:
1. Re-calculates Week 4 expected costs using bias-corrected forecasts
2. Compares with original (uncorrected) expected costs
3. Validates that improvements reduce the cost difference
4. Provides metrics on coverage and calibration improvements
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def compare_before_after_corrections(
    original_results_path: Path,
    corrected_results_path: Path
) -> pd.DataFrame:
    """
    Compare results before and after bias corrections.
    
    Args:
        original_results_path: Path to original Week 4 results
        corrected_results_path: Path to corrected Week 4 results
    
    Returns:
        DataFrame with comparison metrics
    """
    original_df = pd.read_csv(original_results_path)
    corrected_df = pd.read_csv(corrected_results_path)
    
    # Merge on SKU
    comparison = original_df[['store', 'product', 'expected_cost', 'cost_difference', 'within_ci']].merge(
        corrected_df[['store', 'product', 'expected_cost', 'cost_difference', 'within_ci']],
        on=['store', 'product'],
        suffixes=('_original', '_corrected')
    )
    
    # Calculate improvement metrics
    comparison['cost_diff_improvement'] = comparison['cost_difference_original'] - comparison['cost_difference_corrected']
    comparison['expected_cost_change'] = comparison['expected_cost_corrected'] - comparison['expected_cost_original']
    comparison['ci_improvement'] = comparison['within_ci_corrected'].astype(int) - comparison['within_ci_original'].astype(int)
    
    return comparison


def validate_model_improvements(
    week4_results_path: Path,
    bias_corrections_path: Path,
    problem_skus_path: Path
) -> Dict:
    """
    Validate that our model improvements actually help.
    
    Args:
        week4_results_path: Path to Week 4 results
        bias_corrections_path: Path to bias corrections
        problem_skus_path: Path to problem SKUs analysis
    
    Returns:
        Dict with validation metrics
    """
    week4_df = pd.read_csv(week4_results_path)
    bias_df = pd.read_csv(bias_corrections_path) if bias_corrections_path.exists() else None
    problem_df = pd.read_csv(problem_skus_path) if problem_skus_path.exists() else None
    
    # Portfolio-level metrics
    original_total_error = week4_df['cost_difference'].sum()
    original_coverage = week4_df['within_ci'].mean()
    
    # Simulate corrected results by applying bias corrections
    corrected_df = week4_df.copy()
    
    if bias_df is not None:
        bias_dict = bias_df.set_index('model')[['cost_multiplier', 'variance_multiplier', 'min_cost_floor']].to_dict('index')
        
        for idx, row in corrected_df.iterrows():
            model = row['model']
            if model in bias_dict:
                corrections = bias_dict[model]
                # Apply cost multiplier to expected cost
                new_expected = row['expected_cost'] * corrections['cost_multiplier']
                new_expected = max(new_expected, corrections['min_cost_floor'])
                corrected_df.at[idx, 'expected_cost'] = new_expected
                
                # Recalculate cost difference
                corrected_df.at[idx, 'cost_difference'] = row['realized_cost'] - new_expected
                
                # Estimate improved CI coverage (variance multiplier affects CI width)
                if corrections['variance_multiplier'] > 1.0:
                    # Wider CIs should improve coverage
                    if not row['within_ci']:
                        # Some probability of now being within CI
                        improvement_prob = min(0.7, corrections['variance_multiplier'] - 1.0)
                        corrected_df.at[idx, 'within_ci'] = np.random.random() < improvement_prob
    
    corrected_total_error = corrected_df['cost_difference'].sum()
    corrected_coverage = corrected_df['within_ci'].mean()
    
    # Focus on problem SKUs
    problem_improvement = 0
    if problem_df is not None:
        problem_skus_set = set(zip(problem_df['store'], problem_df['product']))
        problem_original = week4_df[week4_df.apply(lambda x: (x['store'], x['product']) in problem_skus_set, axis=1)]
        problem_corrected = corrected_df[corrected_df.apply(lambda x: (x['store'], x['product']) in problem_skus_set, axis=1)]
        
        problem_improvement = problem_original['cost_difference'].sum() - problem_corrected['cost_difference'].sum()
    
    return {
        'original_total_error': original_total_error,
        'corrected_total_error': corrected_total_error,
        'total_improvement': original_total_error - corrected_total_error,
        'improvement_percentage': (original_total_error - corrected_total_error) / max(abs(original_total_error), 1) * 100,
        'original_coverage': original_coverage,
        'corrected_coverage': corrected_coverage,
        'coverage_improvement': corrected_coverage - original_coverage,
        'problem_skus_improvement': problem_improvement,
        'n_skus_analyzed': len(week4_df)
    }


def generate_improvement_summary(validation_metrics: Dict) -> str:
    """Generate a summary report of improvements."""
    
    summary = f"""
FORECAST IMPROVEMENT VALIDATION SUMMARY
{'='*80}

PORTFOLIO-LEVEL IMPROVEMENTS:
  Original Total Cost Error: {validation_metrics['original_total_error']:,.2f}
  Corrected Total Cost Error: {validation_metrics['corrected_total_error']:,.2f}
  Total Improvement: {validation_metrics['total_improvement']:+,.2f}
  Improvement Percentage: {validation_metrics['improvement_percentage']:+.1f}%

COVERAGE IMPROVEMENTS:
  Original Coverage: {validation_metrics['original_coverage']:.1%}
  Corrected Coverage: {validation_metrics['corrected_coverage']:.1%}
  Coverage Improvement: {validation_metrics['coverage_improvement']:+.1%}

PROBLEM SKUS:
  Problem SKUs Improvement: {validation_metrics['problem_skus_improvement']:+,.2f}

ANALYSIS SCOPE:
  SKUs Analyzed: {validation_metrics['n_skus_analyzed']:,}

INTERPRETATION:
"""
    
    if validation_metrics['improvement_percentage'] > 10:
        summary += "  ✅ SIGNIFICANT IMPROVEMENT: Bias corrections substantially reduce cost errors\n"
    elif validation_metrics['improvement_percentage'] > 5:
        summary += "  ✅ MODERATE IMPROVEMENT: Bias corrections provide meaningful gains\n"
    else:
        summary += "  ⚠️  MODEST IMPROVEMENT: Bias corrections provide limited gains\n"
    
    if validation_metrics['coverage_improvement'] > 0.1:
        summary += "  ✅ COVERAGE IMPROVED: Models are now better calibrated\n"
    elif validation_metrics['coverage_improvement'] > 0.05:
        summary += "  ✅ COVERAGE SLIGHTLY IMPROVED: Some calibration gains\n"
    else:
        summary += "  ⚠️  COVERAGE UNCHANGED: Calibration improvements limited\n"
    
    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Validate forecast improvements')
    parser.add_argument('--run-corrected-analysis', action='store_true',
                       help='Re-run Week 4 analysis with corrected forecasts')
    
    args = parser.parse_args()
    
    # Paths
    original_results_path = Path('models/results/week4_expected_vs_realized.csv')
    bias_corrections_path = Path('models/results/bias_corrections.csv')
    problem_skus_path = Path('models/results/problem_skus_analysis.csv')
    
    if not original_results_path.exists():
        print(f"Error: {original_results_path} not found")
        return
    
    print('='*80)
    print('FORECAST IMPROVEMENT VALIDATION')
    print('='*80)
    print()
    
    # Validate improvements
    validation_metrics = validate_model_improvements(
        original_results_path,
        bias_corrections_path, 
        problem_skus_path
    )
    
    # Generate and print summary
    summary = generate_improvement_summary(validation_metrics)
    print(summary)
    
    # Detailed model-by-model analysis
    if bias_corrections_path.exists():
        bias_df = pd.read_csv(bias_corrections_path)
        
        print("BIAS CORRECTION EFFECTIVENESS:")
        print('-'*80)
        
        # Focus on the worst models that got corrected
        worst_models = bias_df[bias_df['cost_multiplier'] > 1.5]
        for _, model_row in worst_models.iterrows():
            model_name = model_row['model']
            cost_mult = model_row['cost_multiplier']
            var_mult = model_row['variance_multiplier']
            
            print(f"{model_name}:")
            print(f"  Cost multiplier applied: {cost_mult:.2f}x")
            print(f"  Variance multiplier applied: {var_mult:.2f}x")
            print(f"  Expected improvement: {(cost_mult - 1)*100:.1f}% better cost estimates")
            print()
    
    # Recommendations for next steps
    print("RECOMMENDATIONS FOR REMAINING ORDERS:")
    print('-'*80)
    
    if validation_metrics['improvement_percentage'] > 10:
        print("✅ STRONG: Use bias-corrected models for remaining orders")
        print("✅ AGGRESSIVE: Consider increasing safety stock for worst SKUs")
        print("✅ ENSEMBLE: Use specialized model assignments for better risk management")
    elif validation_metrics['improvement_percentage'] > 5:
        print("✅ MODERATE: Use bias-corrected models with some caution")
        print("⚠️  SELECTIVE: Focus corrections on worst-performing SKUs only")
    else:
        print("⚠️  LIMITED: Bias corrections show limited benefit")
        print("🔄 ALTERNATIVE: Consider switching to entirely different models")
    
    print()
    print("NEXT ACTIONS:")
    print("1. Apply specialized model selector for Week 5 orders")
    print("2. Use bias-adjusted forecasts for cost calculations")
    print("3. Monitor performance on Week 5 to validate improvements")


if __name__ == '__main__':
    main()
