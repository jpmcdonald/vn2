#!/usr/bin/env python3
"""
Statistical analysis to decompose skill vs luck in inventory optimization competitions.

This script implements rigorous statistical methods to determine whether
competition performance is driven by skill or randomness.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from scipy import stats
from typing import Dict, Tuple
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_competition_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all competition analysis data."""
    leaderboard_data = pd.read_csv(data_dir / 'leaderboard_data.csv')
    competitor_metrics = pd.read_csv(data_dir / 'competitor_metrics.csv')
    rank_changes = pd.read_csv(data_dir / 'rank_changes.csv')
    
    return leaderboard_data, competitor_metrics, rank_changes


def analyze_performance_persistence(leaderboard_df: pd.DataFrame) -> Dict:
    """
    Analyze performance persistence across weeks.
    
    High persistence = more skill
    Low persistence = more luck
    """
    # Get Week 3 and Week 4 data
    week3_data = leaderboard_df[leaderboard_df['week'] == 3].set_index('name')
    week4_data = leaderboard_df[leaderboard_df['week'] == 4].set_index('name')
    
    # Find competitors in both weeks
    common_competitors = week3_data.index.intersection(week4_data.index)
    
    if len(common_competitors) < 5:
        return {'error': 'Insufficient overlapping competitors'}
    
    # Extract ranks and costs
    week3_ranks = week3_data.loc[common_competitors]['rank']
    week4_ranks = week4_data.loc[common_competitors]['rank']
    week3_costs = week3_data.loc[common_competitors]['cumulative_cost']
    week4_costs = week4_data.loc[common_competitors]['cumulative_cost']
    
    # Calculate correlations
    rank_corr, rank_p = stats.spearmanr(week3_ranks, week4_ranks)
    cost_corr, cost_p = stats.pearsonr(week3_costs, week4_costs)
    
    # Performance quartile persistence
    week3_total = len(week3_data)
    week4_total = len(week4_data)
    
    # Define quartiles in Week 3
    week3_top_quartile = week3_ranks <= week3_total * 0.25
    week3_bottom_quartile = week3_ranks >= week3_total * 0.75
    
    # Check persistence in Week 4
    week4_percentiles = week4_ranks / week4_total
    
    top_quartile_persistence = (week4_percentiles[week3_top_quartile] <= 0.5).mean() if week3_top_quartile.sum() > 0 else 0
    bottom_quartile_persistence = (week4_percentiles[week3_bottom_quartile] >= 0.5).mean() if week3_bottom_quartile.sum() > 0 else 0
    
    return {
        'n_competitors': len(common_competitors),
        'rank_correlation': rank_corr,
        'rank_correlation_pvalue': rank_p,
        'cost_correlation': cost_corr,
        'cost_correlation_pvalue': cost_p,
        'top_quartile_persistence': top_quartile_persistence,
        'bottom_quartile_persistence': bottom_quartile_persistence,
        'overall_persistence': (top_quartile_persistence + bottom_quartile_persistence) / 2
    }


def variance_decomposition_analysis(leaderboard_df: pd.DataFrame) -> Dict:
    """
    Decompose performance variance into skill and luck components.
    
    Uses a simple model: Total Variance = Skill Variance + Luck Variance
    """
    analysis = {}
    
    for week in sorted(leaderboard_df['week'].unique()):
        week_data = leaderboard_df[leaderboard_df['week'] == week]
        
        if len(week_data) < 10:
            continue
        
        costs = week_data['order_cost'].dropna()
        cumulative_costs = week_data['cumulative_cost'].dropna()
        
        # Calculate variance components
        cost_var = costs.var()
        cost_mean = costs.mean()
        
        # Coefficient of variation as skill proxy
        # High CV = high skill differences, Low CV = mostly luck
        cost_cv = costs.std() / cost_mean if cost_mean > 0 else 0
        
        # Distribution shape analysis
        _, shapiro_p = stats.shapiro(costs) if len(costs) < 5000 else (None, None)
        skewness = stats.skew(costs)
        kurtosis = stats.kurtosis(costs)
        
        # Performance spread analysis
        winner_cost = costs.min()
        loser_cost = costs.max()
        median_cost = costs.median()
        
        # Skill indicator: large spread suggests skill matters
        performance_spread = (loser_cost - winner_cost) / median_cost
        
        analysis[f'week_{week}'] = {
            'cost_variance': cost_var,
            'cost_cv': cost_cv,
            'performance_spread': performance_spread,
            'shapiro_pvalue': shapiro_p,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'n_competitors': len(costs),
            'winner_cost': winner_cost,
            'median_cost': median_cost,
            'loser_cost': loser_cost
        }
    
    return analysis


def random_walk_test(leaderboard_df: pd.DataFrame) -> Dict:
    """
    Test if performance follows a random walk (luck) vs persistent differences (skill).
    """
    # Get competitors who participated in both weeks
    week3_data = leaderboard_df[leaderboard_df['week'] == 3].set_index('name')
    week4_data = leaderboard_df[leaderboard_df['week'] == 4].set_index('name')
    
    common_competitors = week3_data.index.intersection(week4_data.index)
    
    if len(common_competitors) < 5:
        return {'error': 'Insufficient data for random walk test'}
    
    # Calculate rank changes
    week3_ranks = week3_data.loc[common_competitors]['rank']
    week4_ranks = week4_data.loc[common_competitors]['rank']
    rank_changes = week4_ranks - week3_ranks
    
    # Random walk hypothesis: rank changes should be normally distributed around 0
    # Skill hypothesis: rank changes should show persistence (correlation)
    
    # Test 1: Are rank changes normally distributed?
    _, normality_p = stats.normaltest(rank_changes)
    
    # Test 2: Is mean rank change significantly different from 0?
    _, mean_test_p = stats.ttest_1samp(rank_changes, 0)
    
    # Test 3: Autocorrelation test (if we had more weeks)
    # For now, just check if variance of rank changes is consistent with random walk
    
    # Expected variance under pure luck
    n_competitors_avg = (len(week3_data) + len(week4_data)) / 2
    expected_random_variance = n_competitors_avg ** 2 / 12  # Uniform distribution variance
    
    actual_variance = rank_changes.var()
    variance_ratio = actual_variance / expected_random_variance
    
    return {
        'n_competitors': len(common_competitors),
        'mean_rank_change': rank_changes.mean(),
        'rank_change_variance': actual_variance,
        'expected_random_variance': expected_random_variance,
        'variance_ratio': variance_ratio,
        'normality_pvalue': normality_p,
        'mean_test_pvalue': mean_test_p,
        'interpretation': {
            'variance_ratio_meaning': 'Ratio > 1 suggests skill (persistent differences), Ratio ≈ 1 suggests luck',
            'variance_ratio_value': variance_ratio,
            'likely_explanation': 'Skill-driven' if variance_ratio > 1.5 else 'Luck-driven' if variance_ratio < 0.7 else 'Mixed'
        }
    }


def calculate_skill_luck_ratio(persistence_metrics: Dict, variance_metrics: Dict) -> Dict:
    """
    Calculate overall skill vs luck ratio using multiple indicators.
    """
    indicators = {}
    
    # Indicator 1: Rank correlation (persistence)
    if 'rank_correlation' in persistence_metrics:
        rank_corr = abs(persistence_metrics['rank_correlation'])
        skill_from_correlation = rank_corr ** 2  # R-squared as skill proxy
        indicators['correlation_skill'] = skill_from_correlation
    
    # Indicator 2: Performance persistence
    if 'overall_persistence' in persistence_metrics:
        persistence = persistence_metrics['overall_persistence']
        # Perfect persistence = 1.0 (all skill), Random = 0.5 (no skill)
        skill_from_persistence = max(0, (persistence - 0.5) * 2)
        indicators['persistence_skill'] = skill_from_persistence
    
    # Indicator 3: Variance ratio
    if 'interpretation' in variance_metrics:
        variance_ratio = variance_metrics['variance_ratio']
        # Normalize variance ratio to 0-1 skill scale
        skill_from_variance = min(1.0, max(0, (variance_ratio - 0.5) / 2.0))
        indicators['variance_skill'] = skill_from_variance
    
    # Combined skill estimate (average of indicators)
    if indicators:
        combined_skill = np.mean(list(indicators.values()))
        combined_luck = 1 - combined_skill
        
        return {
            'individual_indicators': indicators,
            'combined_skill_estimate': combined_skill,
            'combined_luck_estimate': combined_luck,
            'confidence': len(indicators) / 3,  # How many indicators we have
            'interpretation': {
                'skill_percentage': combined_skill * 100,
                'luck_percentage': combined_luck * 100,
                'conclusion': (
                    'Skill-dominated competition' if combined_skill > 0.7 else
                    'Luck-dominated competition' if combined_skill < 0.3 else
                    'Mixed skill and luck competition'
                )
            }
        }
    
    return {'error': 'Insufficient data for skill/luck estimation'}


def assess_competition_efficiency(leaderboard_df: pd.DataFrame) -> Dict:
    """
    Assess how efficiently the competition separates skill from luck.
    """
    efficiency_metrics = {}
    
    for week in sorted(leaderboard_df['week'].unique()):
        week_data = leaderboard_df[leaderboard_df['week'] == week]
        costs = week_data['order_cost'].dropna()
        
        if len(costs) < 10:
            continue
        
        # Performance spread (signal)
        performance_spread = (costs.max() - costs.min()) / costs.median()
        
        # Noise level (CV of costs)
        noise_level = costs.std() / costs.mean()
        
        # Signal-to-noise ratio
        signal_to_noise = performance_spread / max(noise_level, 0.01)
        
        # Competition efficiency: how well does it separate performers?
        # High efficiency = large spread, low noise
        efficiency = signal_to_noise
        
        efficiency_metrics[f'week_{week}'] = {
            'performance_spread': performance_spread,
            'noise_level': noise_level,
            'signal_to_noise_ratio': signal_to_noise,
            'efficiency_score': efficiency,
            'n_competitors': len(costs)
        }
    
    return efficiency_metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze skill vs luck in competition')
    parser.add_argument('--data-dir', type=Path,
                       default=Path('models/results/competition_analysis'),
                       help='Directory with parsed competition data')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('models/results/competition_analysis'),
                       help='Output directory')
    
    args = parser.parse_args()
    
    print('='*80)
    print('SKILL VS LUCK STATISTICAL ANALYSIS')
    print('='*80)
    print()
    
    # Load data
    try:
        leaderboard_df, competitor_metrics, rank_changes = load_competition_data(args.data_dir)
        print(f"✅ Loaded data: {len(leaderboard_df)} records")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Performance persistence analysis
    print("Analyzing performance persistence...")
    persistence_metrics = analyze_performance_persistence(leaderboard_df)
    
    # Variance decomposition
    print("Analyzing variance components...")
    variance_metrics = variance_decomposition_analysis(leaderboard_df)
    
    # Random walk test
    print("Testing random walk hypothesis...")
    random_walk_results = random_walk_test(leaderboard_df)
    
    # Competition efficiency
    print("Assessing competition efficiency...")
    efficiency_metrics = assess_competition_efficiency(leaderboard_df)
    
    # Combined skill vs luck estimate
    print("Calculating combined skill vs luck ratio...")
    skill_luck_ratio = calculate_skill_luck_ratio(persistence_metrics, random_walk_results)
    
    # Print results
    print("\n" + "="*80)
    print("SKILL VS LUCK ANALYSIS RESULTS")
    print("="*80)
    
    # Performance persistence
    if 'error' not in persistence_metrics:
        print("\nPERFORMANCE PERSISTENCE:")
        print("-"*60)
        print(f"Rank correlation (Week 3→4): {persistence_metrics['rank_correlation']:.3f}")
        print(f"Statistical significance: p = {persistence_metrics['rank_correlation_pvalue']:.3f}")
        print(f"Top quartile persistence: {persistence_metrics['top_quartile_persistence']:.1%}")
        print(f"Bottom quartile persistence: {persistence_metrics['bottom_quartile_persistence']:.1%}")
    
    # Variance analysis
    print("\nVARIANCE DECOMPOSITION:")
    print("-"*60)
    for week_key, week_data in variance_metrics.items():
        week = week_key.split('_')[1]
        print(f"Week {week}:")
        print(f"  Performance spread: {week_data['performance_spread']:.2f}")
        print(f"  Coefficient of variation: {week_data['cost_cv']:.3f}")
        print(f"  Skewness: {week_data['skewness']:.2f}")
    
    # Random walk test
    if 'error' not in random_walk_results:
        print("\nRANDOM WALK TEST:")
        print("-"*60)
        print(f"Variance ratio (actual/expected): {random_walk_results['variance_ratio']:.2f}")
        print(f"Interpretation: {random_walk_results['interpretation']['likely_explanation']}")
        print(f"Mean rank change: {random_walk_results['mean_rank_change']:.1f}")
        print(f"Normality test p-value: {random_walk_results['normality_pvalue']:.3f}")
    
    # Competition efficiency
    print("\nCOMPETITION EFFICIENCY:")
    print("-"*60)
    for week_key, week_data in efficiency_metrics.items():
        week = week_key.split('_')[1]
        print(f"Week {week}: Signal/Noise = {week_data['signal_to_noise_ratio']:.2f}")
    
    # Combined skill vs luck estimate
    if 'error' not in skill_luck_ratio:
        print("\nCOMBINED SKILL VS LUCK ESTIMATE:")
        print("-"*60)
        print(f"Skill component: {skill_luck_ratio['interpretation']['skill_percentage']:.1f}%")
        print(f"Luck component: {skill_luck_ratio['interpretation']['luck_percentage']:.1f}%")
        print(f"Conclusion: {skill_luck_ratio['interpretation']['conclusion']}")
        print(f"Confidence: {skill_luck_ratio['confidence']:.1%} (based on {len(skill_luck_ratio['individual_indicators'])} indicators)")
    
    # Research implications
    print(f"\n🔬 RESEARCH IMPLICATIONS:")
    print("="*80)
    
    if 'error' not in skill_luck_ratio:
        skill_pct = skill_luck_ratio['interpretation']['skill_percentage']
        
        if skill_pct > 70:
            print("✅ STRONG EVIDENCE FOR SKILL:")
            print("   - Competition effectively separates good from bad techniques")
            print("   - Performance differences reflect genuine forecasting ability") 
            print("   - Investment in better methods should pay off")
        elif skill_pct > 40:
            print("⚖️  MIXED EVIDENCE:")
            print("   - Competition has both skill and luck components")
            print("   - Some techniques matter, but randomness also plays a role")
            print("   - Multiple submissions and risk management important")
        else:
            print("🎲 STRONG EVIDENCE FOR LUCK:")
            print("   - Competition performance largely random")
            print("   - Forecasting technique has limited impact")
            print("   - Focus should be on risk management and hedging")
    
    # Risk management implications
    print(f"\n📊 RISK MANAGEMENT IMPLICATIONS:")
    print("-"*80)
    
    # Analyze performance spread
    week4_data = leaderboard_df[leaderboard_df['week'] == 4]
    if not week4_data.empty:
        costs = week4_data['cumulative_cost']
        winner_cost = costs.min()
        median_cost = costs.median()
        our_cost = 3625.8  # Our actual cost
        
        print(f"Performance range: {costs.min():.0f} - {costs.max():.0f}")
        print(f"Our cost: {our_cost:.0f} (vs winner: +{our_cost - winner_cost:.0f})")
        print(f"Cost volatility: CV = {costs.std()/costs.mean():.2f}")
        
        if costs.std() / costs.mean() > 0.5:
            print("   → HIGH VOLATILITY: Large differences suggest room for improvement")
            print("   → RECOMMENDATION: Focus on technique and risk management")
        else:
            print("   → LOW VOLATILITY: Performance tightly clustered")
            print("   → RECOMMENDATION: Focus on risk management over technique")
    
    # Save results
    results = {
        'persistence_metrics': persistence_metrics,
        'variance_metrics': variance_metrics,
        'random_walk_results': random_walk_results,
        'efficiency_metrics': efficiency_metrics,
        'skill_luck_ratio': skill_luck_ratio
    }
    
    import json
    with open(args.output_dir / 'skill_vs_luck_analysis.json', 'w') as f:
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif pd.isna(obj):
                return None
            return obj
        
        json.dump(results, f, indent=2, default=convert_types)
    
    print(f"\n✅ Analysis saved to: {args.output_dir / 'skill_vs_luck_analysis.json'}")
    
    return results


if __name__ == '__main__':
    main()
