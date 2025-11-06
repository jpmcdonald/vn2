#!/usr/bin/env python3
"""
Analyze competition leaderboards to understand skill vs luck dynamics.

This script provides comprehensive analysis of competitor performance patterns
to determine whether inventory optimization competitions measure skill or luck.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from scipy import stats
from typing import Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vn2.competition.leaderboard_parser import (
    parse_all_leaderboards, 
    calculate_performance_metrics,
    identify_our_performance,
    analyze_rank_changes
)


def analyze_performance_distribution(leaderboard_df: pd.DataFrame) -> Dict:
    """
    Analyze the distribution of performance across competitors.
    
    Args:
        leaderboard_df: Combined leaderboard DataFrame
    
    Returns:
        Dict with distribution analysis
    """
    analysis = {}
    
    for week in sorted(leaderboard_df['week'].unique()):
        week_data = leaderboard_df[leaderboard_df['week'] == week]
        
        costs = week_data['order_cost'].dropna()
        cumulative_costs = week_data['cumulative_cost'].dropna()
        
        if len(costs) == 0:
            continue
        
        # Distribution statistics
        week_analysis = {
            'n_competitors': len(week_data),
            'cost_mean': costs.mean(),
            'cost_median': costs.median(),
            'cost_std': costs.std(),
            'cost_min': costs.min(),
            'cost_max': costs.max(),
            'cost_range': costs.max() - costs.min(),
            'cost_cv': costs.std() / costs.mean() if costs.mean() > 0 else 0,
            # Percentiles
            'cost_p10': costs.quantile(0.1),
            'cost_p25': costs.quantile(0.25),
            'cost_p75': costs.quantile(0.75),
            'cost_p90': costs.quantile(0.9),
            # Distribution tests
            'shapiro_pvalue': stats.shapiro(costs)[1] if len(costs) < 5000 else None,
            'skewness': stats.skew(costs),
            'kurtosis': stats.kurtosis(costs)
        }
        
        analysis[f'week_{week}'] = week_analysis
    
    return analysis


def calculate_skill_vs_luck_metrics(leaderboard_df: pd.DataFrame) -> Dict:
    """
    Calculate metrics to decompose skill vs luck in performance.
    
    Args:
        leaderboard_df: Combined leaderboard DataFrame
    
    Returns:
        Dict with skill vs luck analysis
    """
    # Get competitors who participated in multiple weeks
    competitor_metrics = calculate_performance_metrics(leaderboard_df)
    multi_week = competitor_metrics[competitor_metrics['weeks_participated'] > 1]
    
    if len(multi_week) < 10:
        return {'error': 'Insufficient multi-week competitors for analysis'}
    
    # Rank correlation between weeks
    rank_correlations = {}
    weeks = sorted(leaderboard_df['week'].unique())
    
    for i, week1 in enumerate(weeks[:-1]):
        week2 = weeks[i + 1]
        
        # Get competitors who participated in both weeks
        week1_data = leaderboard_df[leaderboard_df['week'] == week1].set_index('name')['rank']
        week2_data = leaderboard_df[leaderboard_df['week'] == week2].set_index('name')['rank']
        
        common_names = week1_data.index.intersection(week2_data.index)
        
        if len(common_names) > 5:
            week1_ranks = week1_data.loc[common_names]
            week2_ranks = week2_data.loc[common_names]
            
            # Calculate correlations
            pearson_corr, pearson_p = stats.pearsonr(week1_ranks, week2_ranks)
            spearman_corr, spearman_p = stats.spearmanr(week1_ranks, week2_ranks)
            
            rank_correlations[f'week_{week1}_to_{week2}'] = {
                'n_competitors': len(common_names),
                'pearson_correlation': pearson_corr,
                'pearson_pvalue': pearson_p,
                'spearman_correlation': spearman_corr,
                'spearman_pvalue': spearman_p
            }
    
    # Performance persistence analysis
    persistence_metrics = {}
    
    # Top quartile persistence
    if len(multi_week) > 20:
        # Define top performers in first week they participated
        first_week_performance = {}
        for _, competitor in multi_week.iterrows():
            first_week = min(competitor['weeks_list'])
            first_week_data = leaderboard_df[
                (leaderboard_df['name'] == competitor['name']) & 
                (leaderboard_df['week'] == first_week)
            ]
            if not first_week_data.empty:
                first_rank = first_week_data.iloc[0]['rank']
                total_in_week = leaderboard_df[leaderboard_df['week'] == first_week]['rank'].max()
                percentile = first_rank / total_in_week
                first_week_performance[competitor['name']] = percentile
        
        # Check persistence
        if first_week_performance:
            top_quartile_names = [name for name, pct in first_week_performance.items() if pct <= 0.25]
            
            if top_quartile_names:
                # Check their final performance
                final_performance = []
                for name in top_quartile_names:
                    final_data = multi_week[multi_week['name'] == name]
                    if not final_data.empty:
                        final_rank = final_data.iloc[0]['final_rank']
                        # Get total competitors in final week
                        final_week = max(leaderboard_df[leaderboard_df['name'] == name]['week'])
                        total_final = leaderboard_df[leaderboard_df['week'] == final_week]['rank'].max()
                        final_percentile = final_rank / total_final
                        final_performance.append(final_percentile)
                
                if final_performance:
                    persistence_metrics['top_quartile_persistence'] = {
                        'n_top_quartile': len(top_quartile_names),
                        'avg_final_percentile': np.mean(final_performance),
                        'pct_stayed_top_half': np.mean([p <= 0.5 for p in final_performance]) * 100,
                        'pct_stayed_top_quartile': np.mean([p <= 0.25 for p in final_performance]) * 100
                    }
    
    # Variance decomposition (simplified skill vs luck model)
    # Total variance = Skill variance + Luck variance
    # If performance is mostly skill, we expect high correlation between weeks
    # If performance is mostly luck, we expect low correlation
    
    if rank_correlations:
        avg_correlation = np.mean([corr['spearman_correlation'] for corr in rank_correlations.values()])
        
        # Simple skill estimate: correlation^2 represents skill component
        skill_estimate = avg_correlation ** 2
        luck_estimate = 1 - skill_estimate
        
        persistence_metrics['skill_vs_luck_estimate'] = {
            'avg_rank_correlation': avg_correlation,
            'estimated_skill_component': skill_estimate,
            'estimated_luck_component': luck_estimate,
            'interpretation': 'High correlation = more skill, Low correlation = more luck'
        }
    
    return {
        'rank_correlations': rank_correlations,
        'persistence_metrics': persistence_metrics
    }


def compare_with_our_performance(
    leaderboard_df: pd.DataFrame, 
    our_week4_cost: float = 1780.4,
    our_cumulative_cost: float = 3625.8
) -> Dict:
    """
    Compare our performance with the competition.
    
    Args:
        leaderboard_df: Combined leaderboard DataFrame
        our_week4_cost: Our Week 4 cost
        our_cumulative_cost: Our cumulative cost through Week 4
    
    Returns:
        Dict with performance comparison
    """
    week4_data = leaderboard_df[leaderboard_df['week'] == 4]
    
    if week4_data.empty:
        return {'error': 'No Week 4 data available'}
    
    # Our performance context
    our_rank = len(week4_data[week4_data['cumulative_cost'] < our_cumulative_cost]) + 1
    total_competitors = len(week4_data)
    our_percentile = our_rank / total_competitors
    
    # Performance gaps
    winner_cost = week4_data['cumulative_cost'].min()
    median_cost = week4_data['cumulative_cost'].median()
    
    gap_to_winner = our_cumulative_cost - winner_cost
    gap_to_median = our_cumulative_cost - median_cost
    
    # Risk analysis
    costs = week4_data['cumulative_cost'].dropna()
    
    return {
        'our_rank': our_rank,
        'total_competitors': total_competitors,
        'our_percentile': our_percentile,
        'our_cumulative_cost': our_cumulative_cost,
        'winner_cost': winner_cost,
        'median_cost': median_cost,
        'gap_to_winner': gap_to_winner,
        'gap_to_median': gap_to_median,
        'gap_to_winner_pct': gap_to_winner / winner_cost * 100,
        'gap_to_median_pct': gap_to_median / median_cost * 100,
        'better_than_pct': (1 - our_percentile) * 100,
        'cost_distribution': {
            'min': costs.min(),
            'p25': costs.quantile(0.25),
            'median': costs.median(),
            'p75': costs.quantile(0.75),
            'max': costs.max(),
            'std': costs.std(),
            'cv': costs.std() / costs.mean()
        }
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze competition leaderboards')
    parser.add_argument('--leaderboards-dir', type=Path, 
                       default=Path('data/raw/leaderboards'),
                       help='Directory containing leaderboard files')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('models/results/competition_analysis'),
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    print('='*80)
    print('COMPETITION LEADERBOARD ANALYSIS')
    print('='*80)
    print()
    
    # Parse leaderboards
    print("Parsing leaderboard files...")
    combined_df = parse_all_leaderboards(args.leaderboards_dir)
    
    if combined_df.empty:
        print("Error: No leaderboard data found")
        return
    
    print(f"✅ Parsed data: {len(combined_df)} records, {combined_df['name'].nunique()} competitors")
    print()
    
    # Performance distribution analysis
    print("Analyzing performance distributions...")
    distribution_analysis = analyze_performance_distribution(combined_df)
    
    # Skill vs luck analysis
    print("Calculating skill vs luck metrics...")
    skill_luck_analysis = calculate_skill_vs_luck_metrics(combined_df)
    
    # Compare with our performance
    print("Comparing with our performance...")
    our_comparison = compare_with_our_performance(combined_df)
    
    # Generate summary report
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    
    # Performance distribution summary
    print("\nPERFORMANCE DISTRIBUTION:")
    print("-"*60)
    for week_key, week_stats in distribution_analysis.items():
        week = week_key.split('_')[1]
        print(f"Week {week}:")
        print(f"  Competitors: {week_stats['n_competitors']}")
        print(f"  Cost range: {week_stats['cost_min']:.1f} - {week_stats['cost_max']:.1f}")
        print(f"  Median cost: {week_stats['cost_median']:.1f}")
        print(f"  Coefficient of variation: {week_stats['cost_cv']:.2f}")
        print(f"  Skewness: {week_stats['skewness']:.2f}")
    
    # Skill vs luck summary
    print("\nSKILL VS LUCK ANALYSIS:")
    print("-"*60)
    if 'rank_correlations' in skill_luck_analysis:
        for period, corr_data in skill_luck_analysis['rank_correlations'].items():
            print(f"{period}: Correlation = {corr_data['spearman_correlation']:.3f} "
                  f"(p = {corr_data['spearman_pvalue']:.3f}, n = {corr_data['n_competitors']})")
    
    if 'persistence_metrics' in skill_luck_analysis:
        persistence = skill_luck_analysis['persistence_metrics']
        if 'skill_vs_luck_estimate' in persistence:
            skill_est = persistence['skill_vs_luck_estimate']
            print(f"\nSkill vs Luck Estimate:")
            print(f"  Estimated skill component: {skill_est['estimated_skill_component']:.1%}")
            print(f"  Estimated luck component: {skill_est['estimated_luck_component']:.1%}")
            print(f"  Average rank correlation: {skill_est['avg_rank_correlation']:.3f}")
    
    # Our performance summary
    print("\nOUR PERFORMANCE ANALYSIS:")
    print("-"*60)
    if 'error' not in our_comparison:
        print(f"Final rank: {our_comparison['our_rank']}/{our_comparison['total_competitors']} "
              f"({our_comparison['our_percentile']:.1%} percentile)")
        print(f"Cumulative cost: {our_comparison['our_cumulative_cost']:,.1f}")
        print(f"Gap to winner: +{our_comparison['gap_to_winner']:,.1f} "
              f"({our_comparison['gap_to_winner_pct']:+.1f}%)")
        print(f"Gap to median: +{our_comparison['gap_to_median']:,.1f} "
              f"({our_comparison['gap_to_median_pct']:+.1f}%)")
        print(f"Better than: {our_comparison['better_than_pct']:.1f}% of competitors")
    
    # Save all results
    results = {
        'distribution_analysis': distribution_analysis,
        'skill_luck_analysis': skill_luck_analysis,
        'our_performance_comparison': our_comparison
    }
    
    # Save to files
    combined_df.to_csv(args.output_dir / 'leaderboard_data.csv', index=False)
    
    competitor_metrics = calculate_performance_metrics(combined_df)
    competitor_metrics.to_csv(args.output_dir / 'competitor_metrics.csv', index=False)
    
    rank_changes = analyze_rank_changes(combined_df)
    if not rank_changes.empty:
        rank_changes.to_csv(args.output_dir / 'rank_changes.csv', index=False)
    
    # Save analysis results
    import json
    with open(args.output_dir / 'analysis_results.json', 'w') as f:
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            return obj
        
        json.dump(results, f, indent=2, default=convert_types)
    
    print(f"\n✅ Analysis complete! Results saved to: {args.output_dir}")
    
    # Research insights
    print(f"\n🔬 RESEARCH INSIGHTS:")
    print("="*60)
    
    if 'persistence_metrics' in skill_luck_analysis and 'skill_vs_luck_estimate' in skill_luck_analysis['persistence_metrics']:
        skill_comp = skill_luck_analysis['persistence_metrics']['skill_vs_luck_estimate']['estimated_skill_component']
        
        if skill_comp > 0.5:
            print("✅ SKILL DOMINATES: Competition rewards technique over luck")
            print(f"   Skill component: {skill_comp:.1%}")
        elif skill_comp > 0.25:
            print("⚖️  MIXED: Competition has both skill and luck components")  
            print(f"   Skill component: {skill_comp:.1%}")
        else:
            print("🎲 LUCK DOMINATES: Competition is largely random")
            print(f"   Skill component: {skill_comp:.1%}")
    
    # Risk management insights
    cost_cv = our_comparison.get('cost_distribution', {}).get('cv', 0)
    if cost_cv > 1.0:
        print("📊 HIGH VARIABILITY: Large performance differences suggest room for improvement")
    elif cost_cv > 0.5:
        print("📊 MODERATE VARIABILITY: Some technique matters, some randomness")
    else:
        print("📊 LOW VARIABILITY: Performance tightly clustered")
    
    return results


if __name__ == '__main__':
    main()
