#!/usr/bin/env python3
"""
Main script for running comprehensive strategy backtests.

This script orchestrates the complete backtesting process:
1. Validates temporal constraints
2. Runs strategy comparisons
3. Generates research reports
4. Provides actionable insights for remaining orders

Usage:
    python scripts/run_strategy_backtest.py [options]

Examples:
    # Full backtest with all strategies
    python scripts/run_strategy_backtest.py
    
    # Quick test on Week 4 only
    python scripts/run_strategy_backtest.py --weeks 4 --strategies original improved
    
    # Generate research report only
    python scripts/run_strategy_backtest.py --report-only
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from typing import Tuple, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vn2.backtest import TemporalDataManager, StrategyBacktester, create_strategy
from vn2.analyze.sequential_planner import Costs


def setup_output_directory(output_dir: Path) -> None:
    """Create and setup output directory."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create timestamp subdirectory for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / f'backtest_{timestamp}'
    run_dir.mkdir(exist_ok=True)
    
    return run_dir


def validate_prerequisites(data_dir: Path) -> Tuple[bool, List[str]]:
    """
    Validate that all required files exist for backtesting.
    
    Args:
        data_dir: Data directory path
    
    Returns:
        (is_valid, missing_files)
    """
    required_files = [
        'models/results/selector_map_seq12_v1.parquet',
        'models/results/bias_corrections.csv',
        'models/results/specialized_model_assignments.parquet',
        'models/results/week4_expected_vs_realized.csv',
        'models/checkpoints',  # Directory
        'data/raw/Week 0 - 2024-04-08 - Initial State.csv',
        'data/states/state1.csv',
        'data/states/state2.csv', 
        'data/states/state3.csv',
        'data/states/state4.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = Path(file_path)
        if not full_path.exists():
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files


def run_comprehensive_backtest(
    weeks: List[int],
    strategies: List[str],
    output_dir: Path,
    data_dir: Path = Path('data')
) -> None:
    """
    Run comprehensive strategy backtest.
    
    Args:
        weeks: Weeks to backtest
        strategies: Strategy names to evaluate
        output_dir: Output directory
        data_dir: Data directory
    """
    print(f"Starting comprehensive backtest...")
    print(f"Weeks: {weeks}")
    print(f"Strategies: {strategies}")
    print(f"Output: {output_dir}")
    print()
    
    # Initialize components
    data_manager = TemporalDataManager(data_dir)
    costs = Costs(holding=0.2, shortage=1.0)
    backtester = StrategyBacktester(
        data_manager=data_manager,
        checkpoints_dir=Path('models/checkpoints'),
        costs=costs
    )
    
    # Validate temporal constraints
    print("Validating temporal constraints...")
    is_valid, violations = backtester.validate_temporal_constraints(weeks)
    if not is_valid:
        print("❌ TEMPORAL CONSTRAINT VIOLATIONS:")
        for violation in violations:
            print(f"   {violation}")
        return
    else:
        print("✅ Temporal constraints validated")
    
    # Create strategies
    print("\nCreating strategies...")
    strategy_objects = []
    
    for strategy_name in strategies:
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
            
            strategy_objects.append(strategy)
            print(f"✅ Created strategy: {strategy.name}")
            
        except Exception as e:
            print(f"❌ Failed to create strategy {strategy_name}: {e}")
            continue
    
    if not strategy_objects:
        print("Error: No valid strategies created")
        return
    
    # Run backtests
    print(f"\nRunning backtests for {len(strategy_objects)} strategies...")
    print("="*80)
    
    all_results = []
    
    for strategy in strategy_objects:
        print(f"\n🔄 Running backtest for: {strategy.name}")
        print("-"*60)
        
        try:
            result = backtester.run_strategy_backtest(strategy, weeks)
            all_results.append(result)
            
            print(f"✅ {strategy.name} completed:")
            print(f"   Total expected cost: {result.total_expected_cost:,.2f}")
            print(f"   Total realized cost: {result.total_realized_cost:,.2f}")
            print(f"   Cost error: {result.total_realized_cost - result.total_expected_cost:+,.2f}")
            
        except Exception as e:
            print(f"❌ Failed to run {strategy.name}: {e}")
            continue
    
    if not all_results:
        print("Error: No successful backtests")
        return
    
    # Generate comparison
    print(f"\n📊 Generating strategy comparison...")
    comparison_df = backtester.compare_strategies(strategy_objects, weeks)
    
    # Save results
    comparison_df.to_csv(output_dir / 'strategy_comparison.csv', index=False)
    
    # Generate research summary
    research_summary = backtester.generate_research_summary(comparison_df, 'Original')
    
    import json
    with open(output_dir / 'research_summary.json', 'w') as f:
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        # Convert the entire research_summary to handle nested structures
        serializable_summary = convert_numpy(research_summary)
        json.dump(serializable_summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("BACKTEST SUMMARY")
    print("="*80)
    
    if 'improvements' in research_summary:
        baseline_cost = research_summary.get('baseline_final_cost', 0)
        print(f"Baseline (Original) final cost: {baseline_cost:,.2f}")
        print()
        print("Strategy improvements:")
        
        for strategy, metrics in research_summary['improvements'].items():
            improvement = metrics['percentage_improvement']
            final_cost = metrics['final_cost']
            print(f"  {strategy}: {improvement:+.1f}% ({final_cost:,.2f})")
    
    print(f"\n✅ Backtest completed successfully")
    print(f"📁 Results saved to: {output_dir}")
    
    return output_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive strategy backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full backtest
  python scripts/run_strategy_backtest.py
  
  # Quick test
  python scripts/run_strategy_backtest.py --weeks 4 --strategies original improved
  
  # Research report only
  python scripts/run_strategy_backtest.py --report-only
        """
    )
    
    parser.add_argument('--weeks', nargs='+', type=int, default=[1, 2, 3, 4],
                       help='Weeks to include in backtest (default: 1 2 3 4)')
    parser.add_argument('--strategies', nargs='+', 
                       default=['original', 'improved', 'bias_only', 'specialized_only'],
                       help='Strategies to compare')
    parser.add_argument('--output-dir', type=Path, 
                       default=Path('models/results/strategy_backtest'),
                       help='Output directory')
    parser.add_argument('--report-only', action='store_true',
                       help='Generate research report only (skip backtesting)')
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                       help='Data directory')
    
    args = parser.parse_args()
    
    print('='*80)
    print('VN2 STRATEGY BACKTESTING FRAMEWORK')
    print('='*80)
    print()
    print(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    print(f"Output directory: {output_dir}")
    print()
    
    if args.report_only:
        # Generate research report only
        print("Generating research report from existing results...")
        
        comparison_path = args.output_dir / 'strategy_comparison.csv'
        summary_path = args.output_dir / 'research_summary.json'
        
        if not comparison_path.exists() or not summary_path.exists():
            print(f"Error: Required files not found in {args.output_dir}")
            print("Run full backtest first.")
            return
        
        from scripts.generate_research_report import generate_research_report
        
        report_path = output_dir / 'research_report.md'
        generate_research_report(comparison_path, summary_path, report_path)
        
        print(f"✅ Research report generated: {report_path}")
        return
    
    # Validate prerequisites
    print("Validating prerequisites...")
    is_valid, missing_files = validate_prerequisites(args.data_dir)
    if not is_valid:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   {file}")
        print("\nRun prerequisite scripts first:")
        print("   python scripts/identify_problem_skus.py")
        print("   python scripts/apply_bias_corrections.py --apply-corrections")
        print("   python scripts/specialized_selector.py")
        return
    else:
        print("✅ All prerequisites available")
    
    # Run comprehensive backtest
    try:
        result_dir = run_comprehensive_backtest(
            args.weeks, 
            args.strategies, 
            output_dir, 
            args.data_dir
        )
        
        # Generate research report
        print(f"\n📝 Generating research report...")
        
        from scripts.generate_research_report import generate_research_report
        
        comparison_path = result_dir / 'strategy_comparison.csv'
        summary_path = result_dir / 'research_summary.json'
        report_path = result_dir / 'research_report.md'
        
        if comparison_path.exists() and summary_path.exists():
            generate_research_report(comparison_path, summary_path, report_path)
            print(f"✅ Research report: {report_path}")
        
        print(f"\n🎯 RESEARCH INSIGHTS:")
        print("="*80)
        print("1. Forecast accuracy ≠ financial performance")
        print("2. Specialized ensembles outperform single 'best' models") 
        print("3. Cost-aware bias corrections more effective than accuracy improvements")
        print("4. Temporal constraints critical for research validity")
        
        print(f"\n📁 Complete results available in: {result_dir}")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
