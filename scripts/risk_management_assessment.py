#!/usr/bin/env python3
"""
Risk management assessment for inventory optimization competition.

This script analyzes whether our risk management approach is optimal
and provides recommendations for the remaining orders.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def analyze_risk_profiles(leaderboard_df: pd.DataFrame) -> Dict:
    """
    Analyze risk profiles of different competitors.
    
    Args:
        leaderboard_df: Leaderboard data
    
    Returns:
        Dict with risk profile analysis
    """
    risk_profiles = {}
    
    # Get competitors with multiple weeks
    multi_week_competitors = []
    for name in leaderboard_df['name'].unique():
        competitor_data = leaderboard_df[leaderboard_df['name'] == name]
        if len(competitor_data) > 1:
            multi_week_competitors.append(name)
    
    for name in multi_week_competitors:
        competitor_data = leaderboard_df[leaderboard_df['name'] == name].sort_values('week')
        
        costs = competitor_data['order_cost'].values
        ranks = competitor_data['rank'].values
        
        if len(costs) < 2:
            continue
        
        # Risk metrics
        cost_volatility = np.std(costs) / np.mean(costs) if np.mean(costs) > 0 else 0
        rank_volatility = np.std(ranks)
        
        # Performance trajectory
        cost_trend = np.polyfit(range(len(costs)), costs, 1)[0] if len(costs) > 1 else 0
        rank_trend = np.polyfit(range(len(ranks)), ranks, 1)[0] if len(ranks) > 1 else 0
        
        # Risk classification
        if cost_volatility < 0.2:
            risk_profile = 'Conservative'
        elif cost_volatility > 0.5:
            risk_profile = 'Aggressive'
        else:
            risk_profile = 'Moderate'
        
        risk_profiles[name] = {
            'cost_volatility': cost_volatility,
            'rank_volatility': rank_volatility,
            'cost_trend': cost_trend,
            'rank_trend': rank_trend,
            'risk_profile': risk_profile,
            'final_cost': competitor_data['cumulative_cost'].iloc[-1],
            'final_rank': competitor_data['rank'].iloc[-1]
        }
    
    return risk_profiles


def evaluate_our_risk_management(
    leaderboard_df: pd.DataFrame,
    our_costs: List[float] = [380.6, 533.2, 931.6, 1780.4],
    our_cumulative: List[float] = [380.6, 913.8, 1845.4, 3625.8]
) -> Dict:
    """
    Evaluate our risk management approach.
    
    Args:
        leaderboard_df: Leaderboard data
        our_costs: Our weekly costs
        our_cumulative: Our cumulative costs
    
    Returns:
        Dict with risk management evaluation
    """
    evaluation = {}
    
    # Calculate our risk metrics
    our_cost_volatility = np.std(our_costs) / np.mean(our_costs)
    our_cost_trend = np.polyfit(range(len(our_costs)), our_costs, 1)[0]
    
    # Compare with competition
    week4_data = leaderboard_df[leaderboard_df['week'] == 4]
    
    if not week4_data.empty:
        competitor_costs = week4_data['cumulative_cost'].values
        
        # Percentile analysis
        our_percentile = (competitor_costs < our_cumulative[-1]).mean()
        
        # Risk-adjusted performance
        # Sharpe ratio equivalent: (return - risk_free) / volatility
        # Here: (negative_cost - worst_cost) / cost_volatility
        worst_cost = competitor_costs.max()
        best_cost = competitor_costs.min()
        
        our_risk_adjusted = (worst_cost - our_cumulative[-1]) / max(our_cost_volatility, 0.01)
        
        # Calculate same for all competitors (simplified)
        competitor_risk_adjusted = []
        for cost in competitor_costs:
            # Assume similar volatility for risk adjustment
            risk_adj = (worst_cost - cost) / max(our_cost_volatility, 0.01)
            competitor_risk_adjusted.append(risk_adj)
        
        our_risk_rank = (np.array(competitor_risk_adjusted) > our_risk_adjusted).sum() + 1
        
        evaluation = {
            'our_cost_volatility': our_cost_volatility,
            'our_cost_trend': our_cost_trend,
            'our_final_cost': our_cumulative[-1],
            'our_rank': len(competitor_costs) - (competitor_costs < our_cumulative[-1]).sum(),
            'our_percentile': our_percentile,
            'our_risk_adjusted_score': our_risk_adjusted,
            'our_risk_adjusted_rank': our_risk_rank,
            'competition_best_cost': best_cost,
            'competition_worst_cost': worst_cost,
            'competition_median_cost': np.median(competitor_costs),
            'gap_to_best': our_cumulative[-1] - best_cost,
            'gap_to_median': our_cumulative[-1] - np.median(competitor_costs),
            'cost_distribution_cv': np.std(competitor_costs) / np.mean(competitor_costs)
        }
    
    return evaluation


def identify_optimal_risk_strategies(leaderboard_df: pd.DataFrame, risk_profiles: Dict) -> Dict:
    """
    Identify optimal risk management strategies from top performers.
    
    Args:
        leaderboard_df: Leaderboard data
        risk_profiles: Risk profile analysis
    
    Returns:
        Dict with optimal strategy insights
    """
    # Get top performers (top 25%)
    week4_data = leaderboard_df[leaderboard_df['week'] == 4]
    total_competitors = len(week4_data)
    top_25pct_cutoff = int(total_competitors * 0.25)
    
    top_performers = week4_data[week4_data['rank'] <= top_25pct_cutoff]['name'].tolist()
    
    # Analyze risk profiles of top performers
    top_performer_profiles = {name: profile for name, profile in risk_profiles.items() 
                            if name in top_performers}
    
    if not top_performer_profiles:
        return {'error': 'No top performers with multiple weeks of data'}
    
    # Calculate average risk metrics for top performers
    top_volatilities = [profile['cost_volatility'] for profile in top_performer_profiles.values()]
    top_trends = [profile['cost_trend'] for profile in top_performer_profiles.values()]
    
    optimal_strategy = {
        'n_top_performers_analyzed': len(top_performer_profiles),
        'avg_cost_volatility': np.mean(top_volatilities),
        'avg_cost_trend': np.mean(top_trends),
        'risk_profile_distribution': {},
        'recommendations': []
    }
    
    # Risk profile distribution among top performers
    for profile in top_performer_profiles.values():
        risk_type = profile['risk_profile']
        optimal_strategy['risk_profile_distribution'][risk_type] = \
            optimal_strategy['risk_profile_distribution'].get(risk_type, 0) + 1
    
    # Generate recommendations based on analysis
    if np.mean(top_volatilities) > 0.3:
        optimal_strategy['recommendations'].append(
            "Top performers use AGGRESSIVE strategies - consider increasing risk tolerance"
        )
    elif np.mean(top_volatilities) < 0.15:
        optimal_strategy['recommendations'].append(
            "Top performers use CONSERVATIVE strategies - current approach may be optimal"
        )
    else:
        optimal_strategy['recommendations'].append(
            "Top performers use MODERATE risk - consider balanced approach"
        )
    
    if np.mean(top_trends) > 0:
        optimal_strategy['recommendations'].append(
            "Top performers increase costs over time - early aggressive ordering may be key"
        )
    else:
        optimal_strategy['recommendations'].append(
            "Top performers decrease costs over time - learning and adaptation is important"
        )
    
    return optimal_strategy


def assess_our_position_and_opportunities(
    our_evaluation: Dict,
    optimal_strategies: Dict,
    leaderboard_df: pd.DataFrame
) -> Dict:
    """
    Assess our current position and identify opportunities for improvement.
    """
    assessment = {}
    
    # Performance gap analysis
    if 'gap_to_best' in our_evaluation:
        gap_to_best = our_evaluation['gap_to_best']
        our_cost = our_evaluation['our_final_cost']
        
        # How much improvement is theoretically possible?
        max_improvement_pct = gap_to_best / our_cost * 100
        
        assessment['improvement_potential'] = {
            'max_possible_improvement': gap_to_best,
            'max_improvement_percentage': max_improvement_pct,
            'current_position': f"Rank {our_evaluation['our_rank']} of {len(leaderboard_df[leaderboard_df['week']==4])}"
        }
    
    # Risk profile comparison
    our_volatility = our_evaluation.get('our_cost_volatility', 0)
    optimal_volatility = optimal_strategies.get('avg_cost_volatility', 0)
    
    if abs(our_volatility - optimal_volatility) > 0.1:
        if our_volatility > optimal_volatility:
            assessment['risk_adjustment'] = "REDUCE RISK: You're more volatile than top performers"
        else:
            assessment['risk_adjustment'] = "INCREASE RISK: You're more conservative than top performers"
    else:
        assessment['risk_adjustment'] = "MAINTAIN CURRENT RISK LEVEL: Similar to top performers"
    
    # Specific recommendations for remaining orders
    recommendations = []
    
    # Based on performance gap
    if max_improvement_pct > 50:
        recommendations.append("MAJOR STRATEGY CHANGE NEEDED: Large performance gap suggests fundamental issues")
    elif max_improvement_pct > 25:
        recommendations.append("MODERATE ADJUSTMENTS: Significant room for improvement")
    else:
        recommendations.append("FINE TUNING: Small adjustments may be sufficient")
    
    # Based on risk profile
    if 'recommendations' in optimal_strategies:
        recommendations.extend(optimal_strategies['recommendations'])
    
    assessment['recommendations_for_remaining_orders'] = recommendations
    
    return assessment


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Assess risk management strategies')
    parser.add_argument('--data-dir', type=Path,
                       default=Path('models/results/competition_analysis'),
                       help='Directory with competition analysis data')
    
    args = parser.parse_args()
    
    print('='*80)
    print('RISK MANAGEMENT ASSESSMENT')
    print('='*80)
    print()
    
    # Load data
    try:
        leaderboard_df = pd.read_csv(args.data_dir / 'leaderboard_data.csv')
        print(f"✅ Loaded leaderboard data: {len(leaderboard_df)} records")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Analyze risk profiles
    print("Analyzing competitor risk profiles...")
    risk_profiles = analyze_risk_profiles(leaderboard_df)
    
    # Evaluate our risk management
    print("Evaluating our risk management approach...")
    our_evaluation = evaluate_our_risk_management(leaderboard_df)
    
    # Identify optimal strategies
    print("Identifying optimal risk strategies...")
    optimal_strategies = identify_optimal_risk_strategies(leaderboard_df, risk_profiles)
    
    # Assess our position
    print("Assessing our position and opportunities...")
    position_assessment = assess_our_position_and_opportunities(
        our_evaluation, optimal_strategies, leaderboard_df
    )
    
    # Print results
    print("\n" + "="*80)
    print("RISK MANAGEMENT ASSESSMENT RESULTS")
    print("="*80)
    
    # Our performance summary
    print("\nOUR PERFORMANCE SUMMARY:")
    print("-"*60)
    if 'our_final_cost' in our_evaluation:
        print(f"Final cumulative cost: {our_evaluation['our_final_cost']:,.1f}")
        print(f"Final rank: {our_evaluation['our_rank']}")
        print(f"Percentile: {our_evaluation['our_percentile']:.1%}")
        print(f"Cost volatility: {our_evaluation['our_cost_volatility']:.3f}")
        print(f"Gap to winner: +{our_evaluation['gap_to_best']:,.1f} (+{our_evaluation['gap_to_best']/our_evaluation['competition_best_cost']*100:.1f}%)")
        print(f"Gap to median: +{our_evaluation['gap_to_median']:,.1f}")
    
    # Optimal strategy insights
    print("\nOPTIMAL STRATEGY ANALYSIS:")
    print("-"*60)
    if 'error' not in optimal_strategies:
        print(f"Top performers analyzed: {optimal_strategies['n_top_performers_analyzed']}")
        print(f"Average volatility of top performers: {optimal_strategies['avg_cost_volatility']:.3f}")
        print("Risk profile distribution among top performers:")
        for profile, count in optimal_strategies['risk_profile_distribution'].items():
            print(f"  {profile}: {count}")
        
        print("\nTop performer recommendations:")
        for rec in optimal_strategies['recommendations']:
            print(f"  • {rec}")
    
    # Position assessment
    print("\nPOSITION ASSESSMENT:")
    print("-"*60)
    if 'improvement_potential' in position_assessment:
        potential = position_assessment['improvement_potential']
        print(f"Current position: {potential['current_position']}")
        print(f"Max possible improvement: {potential['max_improvement_percentage']:.1f}%")
        print(f"Risk adjustment: {position_assessment['risk_adjustment']}")
        
        print("\nRecommendations for remaining orders:")
        for rec in position_assessment['recommendations_for_remaining_orders']:
            print(f"  • {rec}")
    
    # Strategic insights
    print(f"\n🎯 STRATEGIC INSIGHTS:")
    print("="*80)
    
    # Competition structure insights
    week4_data = leaderboard_df[leaderboard_df['week'] == 4]
    if not week4_data.empty:
        costs = week4_data['cumulative_cost']
        cost_range = costs.max() - costs.min()
        median_cost = costs.median()
        range_pct = cost_range / median_cost * 100
        
        print(f"Competition cost range: {cost_range:,.0f} ({range_pct:.0f}% of median)")
        
        if range_pct > 100:
            print("✅ LARGE PERFORMANCE DIFFERENCES: Technique matters significantly")
            print("   → Focus on improving forecasting and optimization methods")
        elif range_pct > 50:
            print("⚖️  MODERATE PERFORMANCE DIFFERENCES: Both technique and luck matter")
            print("   → Balance between method improvement and risk management")
        else:
            print("🎲 SMALL PERFORMANCE DIFFERENCES: Luck dominates")
            print("   → Focus primarily on risk management and hedging")
    
    # Risk management implications
    our_vol = our_evaluation.get('our_cost_volatility', 0)
    comp_vol = our_evaluation.get('cost_distribution_cv', 0)
    
    if our_vol > comp_vol * 1.5:
        print(f"\n📊 HIGH INDIVIDUAL VOLATILITY: Your strategy is riskier than average")
        print("   → Consider more conservative ordering")
    elif our_vol < comp_vol * 0.5:
        print(f"\n📊 LOW INDIVIDUAL VOLATILITY: Your strategy is more conservative than average")
        print("   → Consider more aggressive ordering if performance gap is large")
    else:
        print(f"\n📊 MODERATE INDIVIDUAL VOLATILITY: Your risk level is typical")
    
    # Save results
    results = {
        'risk_profiles': risk_profiles,
        'our_evaluation': our_evaluation,
        'optimal_strategies': optimal_strategies,
        'position_assessment': position_assessment
    }
    
    import json
    with open(args.data_dir / 'risk_assessment.json', 'w') as f:
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif pd.isna(obj):
                return None
            return obj
        
        json.dump(results, f, indent=2, default=convert_types)
    
    print(f"\n✅ Risk assessment saved to: {args.data_dir / 'risk_assessment.json'}")
    
    return results


if __name__ == '__main__':
    main()
