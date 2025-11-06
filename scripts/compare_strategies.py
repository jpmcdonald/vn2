#!/usr/bin/env python3
"""
Compare different forecasting strategies using rigorous backtesting.

This script generates research-quality comparisons of forecasting strategies,
showing cumulative cost improvements and validating the thesis that
forecast accuracy ≠ financial performance.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vn2.backtest import TemporalDataManager, StrategyBacktester, create_strategy
from vn2.analyze.sequential_planner import Costs


def create_strategy_comparison_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create publication-ready strategy comparison table.
    
    Args:
        comparison_df: Raw comparison data
    
    Returns:
        Formatted comparison table
    """
    # Pivot to get strategies as columns
    pivot_realized = comparison_df.pivot(index='week', columns='strategy', values='cumulative_realized')
    pivot_expected = comparison_df.pivot(index='week', columns='strategy', values='cumulative_expected')
    
    # Calculate improvements vs baseline (Original)
    if 'Original' in pivot_realized.columns:
        baseline = pivot_realized['Original']
        
        improvement_table = []
        for week in pivot_realized.index:
            row = {'Week': week, 'Original_Cost': baseline[week]}
            
            for strategy in pivot_realized.columns:
                if strategy != 'Original':
                    strategy_cost = pivot_realized.loc[week, strategy]
                    improvement = baseline[week] - strategy_cost
                    improvement_pct = improvement / baseline[week] * 100 if baseline[week] > 0 else 0
                    
                    row[f'{strategy}_Cost'] = strategy_cost
                    row[f'{strategy}_Improvement'] = improvement
                    row[f'{strategy}_Improvement_Pct'] = improvement_pct
            
            improvement_table.append(row)
        
        return pd.DataFrame(improvement_table)
    
    return comparison_df


def generate_ablation_analysis(comparison_df: pd.DataFrame) -> Dict:
    """
    Generate ablation study analysis.
    
    Args:
        comparison_df: Strategy comparison data
    
    Returns:
        Dict with ablation analysis
    """
    strategies = comparison_df['strategy'].unique()
    final_week = comparison_df['week'].max()
    
    # Get final cumulative costs
    final_costs = comparison_df[comparison_df['week'] == final_week].set_index('strategy')['cumulative_realized']
    
    ablation = {}
    
    if all(s in final_costs.index for s in ['Original', 'BiasOnly', 'SpecializedOnly', 'Improved']):
        original_cost = final_costs['Original']
        bias_only_cost = final_costs['BiasOnly'] 
        specialized_only_cost = final_costs['SpecializedOnly']
        improved_cost = final_costs['Improved']
        
        # Calculate individual contributions
        bias_contribution = original_cost - bias_only_cost
        specialized_contribution = original_cost - specialized_only_cost
        total_improvement = original_cost - improved_cost
        
        # Check for interaction effects
        expected_combined = bias_contribution + specialized_contribution
        actual_combined = total_improvement
        interaction_effect = actual_combined - expected_combined
        
        ablation = {
            'original_cost': original_cost,
            'bias_only_improvement': bias_contribution,
            'specialized_only_improvement': specialized_contribution,
            'total_improvement': total_improvement,
            'expected_combined_improvement': expected_combined,
            'actual_combined_improvement': actual_combined,
            'interaction_effect': interaction_effect,
            'bias_contribution_pct': bias_contribution / total_improvement * 100 if total_improvement != 0 else 0,
            'specialized_contribution_pct': specialized_contribution / total_improvement * 100 if total_improvement != 0 else 0
        }
    
    return ablation


def generate_model_attribution_analysis(
    backtester: StrategyBacktester,
    strategies: List,
    comparison_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze which models contributed most to improvements.
    
    Args:
        backtester: Strategy backtester instance
        strategies: List of strategies evaluated
        comparison_df: Strategy comparison data
    
    Returns:
        DataFrame with model attribution analysis
    """
    # This would require running the backtester and collecting model attribution
    # For now, return placeholder
    return pd.DataFrame({
        'model': ['slurp_bootstrap', 'zinb', 'lightgbm_quantile'],
        'cost_contribution': [100, 200, 300],
        'usage_count': [50, 75, 100]
    })


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Compare forecasting strategies')
    parser.add_argument('--weeks', nargs='+', type=int, default=[1, 2, 3, 4],
                       help='Weeks to include in backtest')
    parser.add_argument('--strategies', nargs='+', default=['original', 'improved', 'bias_only', 'specialized_only'],
                       help='Strategies to compare')
    parser.add_argument('--output-dir', type=Path, default=Path('models/results/strategy_comparison'),
                       help='Output directory for results')
    
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True)
    
    print('='*80)
    print('STRATEGY COMPARISON BACKTEST')
    print('='*80)
    print()
    print(f"Weeks: {args.weeks}")
    print(f"Strategies: {args.strategies}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Initialize components
    data_manager = TemporalDataManager(Path('data'))
    costs = Costs(holding=0.2, shortage=1.0)
    
    # Validate data availability
    print("Data availability check...")
    data_manager.print_data_availability()
    print()
    
    # Initialize backtester
    backtester = StrategyBacktester(
        data_manager=data_manager,
        checkpoints_dir=Path('models/checkpoints'),
        costs=costs
    )
    
    # Validate temporal constraints
    is_valid, violations = backtester.validate_temporal_constraints(args.weeks)
    if not is_valid:
        print("❌ TEMPORAL CONSTRAINT VIOLATIONS:")
        for violation in violations:
            print(f"   {violation}")
        print()
        return
    else:
        print("✅ Temporal constraints validated")
        print()
    
    # Create strategies
    strategies = []
    for strategy_name in args.strategies:
        try:
            if strategy_name == 'original':
                strategy = create_strategy(
                    'original', costs,
                    selector_map_path=Path('models/results/selector_map_seq12_v1.parquet')
                )
            elif strategy_name == 'improved':
                strategy = create_strategy(
                    'improved', costs,
                    bias_corrections_path=Path('models/results/bias_corrections.csv'),
                    specialized_assignments_path=Path('models/results/specialized_model_assignments.parquet')
                )
            elif strategy_name == 'bias_only':
                strategy = create_strategy(
                    'bias_only', costs,
                    bias_corrections_path=Path('models/results/bias_corrections.csv'),
                    selector_map_path=Path('models/results/selector_map_seq12_v1.parquet')
                )
            elif strategy_name == 'specialized_only':
                strategy = create_strategy(
                    'specialized_only', costs,
                    specialized_assignments_path=Path('models/results/specialized_model_assignments.parquet')
                )
            else:
                print(f"Warning: Unknown strategy {strategy_name}, skipping")
                continue
            
            strategies.append(strategy)
            
        except Exception as e:
            print(f"Warning: Failed to create strategy {strategy_name}: {e}")
            continue
    
    if not strategies:
        print("Error: No valid strategies created")
        return
    
    # Run comparison
    print("Running strategy comparison...")
    comparison_df = backtester.compare_strategies(strategies, args.weeks)
    
    # Generate analysis
    print("\n" + "="*80)
    print("STRATEGY COMPARISON RESULTS")
    print("="*80)
    
    # Strategy comparison table
    comparison_table = create_strategy_comparison_table(comparison_df)
    print("\nSTRATEGY COMPARISON TABLE:")
    print("-"*80)
    print(comparison_table.to_string(index=False, float_format='%.2f'))
    
    # Ablation analysis
    ablation = generate_ablation_analysis(comparison_df)
    if ablation:
        print("\nABLATION STUDY:")
        print("-"*80)
        print(f"Original cost: {ablation['original_cost']:,.2f}")
        print(f"Bias correction contribution: {ablation['bias_contribution_pct']:.1f}%")
        print(f"Specialized ensemble contribution: {ablation['specialized_contribution_pct']:.1f}%")
        print(f"Total improvement: {ablation['total_improvement']:+,.2f}")
        print(f"Interaction effect: {ablation['interaction_effect']:+,.2f}")
    
    # Research summary
    research_summary = backtester.generate_research_summary(comparison_df)
    
    print("\nRESEARCH SUMMARY:")
    print("-"*80)
    if 'improvements' in research_summary:
        for strategy, metrics in research_summary['improvements'].items():
            print(f"{strategy}:")
            print(f"  Final cost: {metrics['final_cost']:,.2f}")
            print(f"  Improvement: {metrics['absolute_improvement']:+,.2f} ({metrics['percentage_improvement']:+.1f}%)")
    
    # Save results
    comparison_df.to_csv(args.output_dir / 'strategy_comparison.csv', index=False)
    comparison_table.to_csv(args.output_dir / 'strategy_comparison_table.csv', index=False)
    
    # Save research summary
    import json
    with open(args.output_dir / 'research_summary.json', 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(research_summary, f, indent=2, default=convert_numpy)
    
    print(f"\n✅ Results saved to: {args.output_dir}")
    print(f"   - strategy_comparison.csv: Raw comparison data")
    print(f"   - strategy_comparison_table.csv: Publication-ready table")
    print(f"   - research_summary.json: Research summary metrics")


if __name__ == '__main__':
    main()
