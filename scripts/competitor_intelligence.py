#!/usr/bin/env python3
"""
Competitor intelligence analysis for inventory optimization competition.

This script classifies competitor strategies and identifies performance patterns
to understand what approaches are working in the competition.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def classify_competitor_strategies(leaderboard_df: pd.DataFrame) -> Dict:
    """
    Classify competitors into strategy archetypes based on performance patterns.
    
    Args:
        leaderboard_df: Leaderboard data
    
    Returns:
        Dict with strategy classifications
    """
    # Get competitors with data for analysis
    competitor_features = []
    competitor_names = []
    
    for name in leaderboard_df['name'].unique():
        competitor_data = leaderboard_df[leaderboard_df['name'] == name].sort_values('week')
        
        if len(competitor_data) < 1:
            continue
        
        # Extract features for clustering
        costs = competitor_data['order_cost'].values
        ranks = competitor_data['rank'].values
        cumulative_costs = competitor_data['cumulative_cost'].values
        
        # Feature engineering
        features = []
        
        # 1. Final performance
        final_rank = ranks[-1]
        final_cost = cumulative_costs[-1]
        features.extend([final_rank, final_cost])
        
        # 2. Performance volatility
        if len(costs) > 1:
            cost_volatility = np.std(costs) / np.mean(costs)
            rank_volatility = np.std(ranks)
        else:
            cost_volatility = 0
            rank_volatility = 0
        features.extend([cost_volatility, rank_volatility])
        
        # 3. Performance trend
        if len(costs) > 1:
            cost_trend = np.polyfit(range(len(costs)), costs, 1)[0]
            rank_trend = np.polyfit(range(len(ranks)), ranks, 1)[0]
        else:
            cost_trend = 0
            rank_trend = 0
        features.extend([cost_trend, rank_trend])
        
        # 4. Risk characteristics
        max_weekly_cost = max(costs)
        min_weekly_cost = min(costs)
        cost_range = max_weekly_cost - min_weekly_cost
        features.extend([max_weekly_cost, cost_range])
        
        # 5. Submission behavior
        total_entries = competitor_data['entries'].sum()
        avg_applies = competitor_data['apply_count'].mean()
        features.extend([total_entries, avg_applies])
        
        competitor_features.append(features)
        competitor_names.append(name)
    
    if len(competitor_features) < 3:
        return {'error': 'Insufficient competitors for clustering'}
    
    # Normalize features for clustering
    features_df = pd.DataFrame(competitor_features, index=competitor_names)
    features_normalized = (features_df - features_df.mean()) / features_df.std()
    features_normalized = features_normalized.fillna(0)
    
    # K-means clustering to identify strategy types
    n_clusters = min(4, len(competitor_features) // 2)  # Reasonable number of clusters
    
    if n_clusters >= 2:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_normalized)
        
        # Analyze clusters
        clusters = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_competitors = np.array(competitor_names)[cluster_mask]
            cluster_features = features_df.iloc[cluster_mask]
            
            # Characterize cluster
            cluster_analysis = {
                'competitors': cluster_competitors.tolist(),
                'n_competitors': len(cluster_competitors),
                'avg_final_rank': cluster_features.iloc[:, 0].mean(),
                'avg_final_cost': cluster_features.iloc[:, 1].mean(),
                'avg_cost_volatility': cluster_features.iloc[:, 2].mean(),
                'avg_rank_volatility': cluster_features.iloc[:, 3].mean(),
                'avg_cost_trend': cluster_features.iloc[:, 4].mean(),
                'avg_rank_trend': cluster_features.iloc[:, 5].mean()
            }
            
            # Classify strategy type
            if cluster_analysis['avg_final_rank'] <= len(competitor_names) * 0.3:
                strategy_type = 'Top Performers'
            elif cluster_analysis['avg_cost_volatility'] > 0.3:
                strategy_type = 'High Risk'
            elif cluster_analysis['avg_cost_volatility'] < 0.1:
                strategy_type = 'Conservative'
            else:
                strategy_type = 'Moderate'
            
            cluster_analysis['strategy_type'] = strategy_type
            clusters[f'cluster_{i}'] = cluster_analysis
        
        return {
            'n_clusters': n_clusters,
            'clusters': clusters,
            'cluster_assignments': dict(zip(competitor_names, cluster_labels))
        }
    
    return {'error': 'Too few competitors for meaningful clustering'}


def analyze_submission_patterns(leaderboard_df: pd.DataFrame) -> Dict:
    """
    Analyze submission patterns to infer competitor strategies.
    
    Args:
        leaderboard_df: Leaderboard data
    
    Returns:
        Dict with submission pattern analysis
    """
    patterns = {}
    
    for name in leaderboard_df['name'].unique():
        competitor_data = leaderboard_df[leaderboard_df['name'] == name]
        
        # Submission frequency analysis
        total_applies = competitor_data['apply_count'].sum()
        total_entries = competitor_data['entries'].sum()
        weeks_participated = len(competitor_data)
        
        # Calculate submission intensity
        applies_per_week = total_applies / weeks_participated
        entries_per_week = total_entries / weeks_participated
        
        # Classify submission behavior
        if applies_per_week > 2:
            submission_type = 'Heavy Optimizer'  # Multiple submissions per week
        elif applies_per_week > 1:
            submission_type = 'Active Optimizer'  # Some resubmissions
        else:
            submission_type = 'Single Shot'  # One submission per week
        
        patterns[name] = {
            'total_applies': total_applies,
            'total_entries': total_entries,
            'weeks_participated': weeks_participated,
            'applies_per_week': applies_per_week,
            'entries_per_week': entries_per_week,
            'submission_type': submission_type,
            'final_performance': competitor_data['cumulative_cost'].iloc[-1] if not competitor_data.empty else None
        }
    
    return patterns


def identify_winning_patterns(
    leaderboard_df: pd.DataFrame, 
    strategy_clusters: Dict,
    submission_patterns: Dict
) -> Dict:
    """
    Identify patterns among winning competitors.
    
    Args:
        leaderboard_df: Leaderboard data
        strategy_clusters: Strategy cluster analysis
        submission_patterns: Submission pattern analysis
    
    Returns:
        Dict with winning patterns analysis
    """
    # Get top performers (top 25%)
    week4_data = leaderboard_df[leaderboard_df['week'] == 4]
    total_competitors = len(week4_data)
    top_25pct_cutoff = int(total_competitors * 0.25)
    
    top_performers = week4_data[week4_data['rank'] <= top_25pct_cutoff]['name'].tolist()
    
    winning_patterns = {
        'top_performers': top_performers,
        'n_top_performers': len(top_performers)
    }
    
    # Analyze submission patterns of winners
    winner_submission_types = []
    winner_applies_per_week = []
    
    for name in top_performers:
        if name in submission_patterns:
            pattern = submission_patterns[name]
            winner_submission_types.append(pattern['submission_type'])
            winner_applies_per_week.append(pattern['applies_per_week'])
    
    if winner_submission_types:
        from collections import Counter
        submission_type_counts = Counter(winner_submission_types)
        winning_patterns['submission_patterns'] = {
            'types': dict(submission_type_counts),
            'avg_applies_per_week': np.mean(winner_applies_per_week),
            'dominant_type': submission_type_counts.most_common(1)[0][0]
        }
    
    # Analyze strategy clusters of winners
    if 'error' not in strategy_clusters and 'cluster_assignments' in strategy_clusters:
        winner_clusters = []
        for name in top_performers:
            if name in strategy_clusters['cluster_assignments']:
                cluster_id = strategy_clusters['cluster_assignments'][name]
                winner_clusters.append(cluster_id)
        
        if winner_clusters:
            from collections import Counter
            cluster_counts = Counter(winner_clusters)
            winning_patterns['strategy_clusters'] = {
                'cluster_distribution': dict(cluster_counts),
                'dominant_cluster': cluster_counts.most_common(1)[0][0]
            }
    
    return winning_patterns


def generate_competitive_intelligence_report(
    strategy_clusters: Dict,
    submission_patterns: Dict,
    winning_patterns: Dict,
    leaderboard_df: pd.DataFrame
) -> str:
    """
    Generate competitive intelligence report.
    """
    report = f"""# Competitive Intelligence Report

## Executive Summary

Based on analysis of {leaderboard_df['name'].nunique()} competitors across {len(leaderboard_df['week'].unique())} weeks:

"""
    
    # Strategy clusters
    if 'error' not in strategy_clusters:
        report += f"""
## Strategy Archetypes

Identified {strategy_clusters['n_clusters']} distinct competitor archetypes:

"""
        for cluster_id, cluster_data in strategy_clusters['clusters'].items():
            cluster_num = cluster_id.split('_')[1]
            report += f"""
### {cluster_data['strategy_type']} (Cluster {cluster_num})
- **Competitors**: {cluster_data['n_competitors']}
- **Average rank**: {cluster_data['avg_final_rank']:.1f}
- **Average cost**: {cluster_data['avg_final_cost']:,.0f}
- **Risk profile**: Volatility = {cluster_data['avg_cost_volatility']:.3f}
- **Members**: {', '.join(cluster_data['competitors'][:5])}{'...' if len(cluster_data['competitors']) > 5 else ''}

"""
    
    # Winning patterns
    report += f"""
## Winning Patterns

Top {winning_patterns['n_top_performers']} performers show these characteristics:

"""
    
    if 'submission_patterns' in winning_patterns:
        sub_patterns = winning_patterns['submission_patterns']
        report += f"""
### Submission Behavior
- **Dominant approach**: {sub_patterns['dominant_type']}
- **Average submissions per week**: {sub_patterns['avg_applies_per_week']:.1f}
- **Distribution**: {sub_patterns['types']}

"""
    
    if 'strategy_clusters' in winning_patterns:
        strat_clusters = winning_patterns['strategy_clusters']
        report += f"""
### Strategy Types
- **Dominant cluster**: {strat_clusters['dominant_cluster']}
- **Cluster distribution**: {strat_clusters['cluster_distribution']}

"""
    
    # Competition dynamics
    week4_data = leaderboard_df[leaderboard_df['week'] == 4]
    costs = week4_data['cumulative_cost']
    
    report += f"""
## Competition Dynamics

### Performance Distribution
- **Range**: {costs.min():,.0f} - {costs.max():,.0f} ({(costs.max()-costs.min())/costs.median()*100:.0f}% of median)
- **Winner advantage**: {(costs.median() - costs.min())/costs.min()*100:.1f}% better than median
- **Tail risk**: Bottom 10% average {costs.quantile(0.9):,.0f} vs top 10% average {costs.quantile(0.1):,.0f}

### Competitive Insights
"""
    
    cost_cv = costs.std() / costs.mean()
    if cost_cv > 0.5:
        report += "- **High variance competition**: Large skill differences, technique matters significantly\n"
    elif cost_cv > 0.2:
        report += "- **Moderate variance competition**: Both skill and luck matter\n"
    else:
        report += "- **Low variance competition**: Performance tightly clustered, luck dominates\n"
    
    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze competitor strategies and patterns')
    parser.add_argument('--data-dir', type=Path,
                       default=Path('models/results/competition_analysis'),
                       help='Directory with competition analysis data')
    
    args = parser.parse_args()
    
    print('='*80)
    print('COMPETITOR INTELLIGENCE ANALYSIS')
    print('='*80)
    print()
    
    # Load data
    try:
        leaderboard_df = pd.read_csv(args.data_dir / 'leaderboard_data.csv')
        print(f"✅ Loaded leaderboard data: {len(leaderboard_df)} records")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Classify competitor strategies
    print("Classifying competitor strategies...")
    strategy_clusters = classify_competitor_strategies(leaderboard_df)
    
    # Analyze submission patterns
    print("Analyzing submission patterns...")
    submission_patterns = analyze_submission_patterns(leaderboard_df)
    
    # Identify winning patterns
    print("Identifying winning patterns...")
    winning_patterns = identify_winning_patterns(leaderboard_df, strategy_clusters, submission_patterns)
    
    # Print results
    print("\n" + "="*80)
    print("COMPETITOR INTELLIGENCE RESULTS")
    print("="*80)
    
    # Strategy clusters
    if 'error' not in strategy_clusters:
        print("\nSTRATEGY ARCHETYPES:")
        print("-"*60)
        for cluster_id, cluster_data in strategy_clusters['clusters'].items():
            print(f"{cluster_data['strategy_type']} ({cluster_data['n_competitors']} competitors):")
            print(f"  Average rank: {cluster_data['avg_final_rank']:.1f}")
            print(f"  Average cost: {cluster_data['avg_final_cost']:,.0f}")
            print(f"  Risk level: {cluster_data['avg_cost_volatility']:.3f}")
            print(f"  Members: {', '.join(cluster_data['competitors'][:3])}...")
            print()
    
    # Submission patterns
    print("SUBMISSION BEHAVIOR ANALYSIS:")
    print("-"*60)
    
    submission_type_counts = {}
    for pattern in submission_patterns.values():
        sub_type = pattern['submission_type']
        submission_type_counts[sub_type] = submission_type_counts.get(sub_type, 0) + 1
    
    for sub_type, count in submission_type_counts.items():
        print(f"{sub_type}: {count} competitors")
    
    # Winning patterns
    print("\nWINNING PATTERNS:")
    print("-"*60)
    print(f"Top performers: {winning_patterns['n_top_performers']}")
    
    if 'submission_patterns' in winning_patterns:
        sub_patterns = winning_patterns['submission_patterns']
        print(f"Dominant submission type: {sub_patterns['dominant_type']}")
        print(f"Average submissions per week: {sub_patterns['avg_applies_per_week']:.1f}")
    
    if 'strategy_clusters' in winning_patterns:
        strat_clusters = winning_patterns['strategy_clusters']
        print(f"Dominant strategy cluster: {strat_clusters['dominant_cluster']}")
    
    # Generate competitive intelligence report
    print("\nGenerating competitive intelligence report...")
    intel_report = generate_competitive_intelligence_report(
        strategy_clusters, submission_patterns, winning_patterns, leaderboard_df
    )
    
    # Save results
    results = {
        'strategy_clusters': strategy_clusters,
        'submission_patterns': submission_patterns,
        'winning_patterns': winning_patterns
    }
    
    import json
    with open(args.data_dir / 'competitor_intelligence.json', 'w') as f:
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif pd.isna(obj):
                return None
            return obj
        
        # Convert the entire results structure
        serializable_results = convert_types(results)
        
        json.dump(serializable_results, f, indent=2)
    
    # Save intelligence report
    with open(args.data_dir / 'competitive_intelligence_report.md', 'w') as f:
        f.write(intel_report)
    
    print(f"\n✅ Competitor intelligence saved to:")
    print(f"   {args.data_dir / 'competitor_intelligence.json'}")
    print(f"   {args.data_dir / 'competitive_intelligence_report.md'}")
    
    # Strategic recommendations
    print(f"\n🎯 STRATEGIC RECOMMENDATIONS:")
    print("="*80)
    
    # Based on winning patterns
    if 'submission_patterns' in winning_patterns:
        dominant_type = winning_patterns['submission_patterns']['dominant_type']
        avg_applies = winning_patterns['submission_patterns']['avg_applies_per_week']
        
        if dominant_type == 'Heavy Optimizer' and avg_applies > 2:
            print("✅ INCREASE SUBMISSION FREQUENCY: Winners use multiple submissions per week")
        elif dominant_type == 'Single Shot':
            print("✅ QUALITY OVER QUANTITY: Winners use single, well-planned submissions")
        else:
            print("⚖️  BALANCED APPROACH: Winners use moderate submission strategies")
    
    # Based on strategy clusters
    if 'error' not in strategy_clusters:
        # Find most successful cluster
        best_cluster = None
        best_avg_rank = float('inf')
        
        for cluster_id, cluster_data in strategy_clusters['clusters'].items():
            if cluster_data['avg_final_rank'] < best_avg_rank:
                best_avg_rank = cluster_data['avg_final_rank']
                best_cluster = cluster_data
        
        if best_cluster:
            print(f"✅ EMULATE {best_cluster['strategy_type']} STRATEGY:")
            print(f"   Target volatility: {best_cluster['avg_cost_volatility']:.3f}")
            print(f"   Performance trend: {best_cluster['avg_cost_trend']:+.1f}")
    
    return results


if __name__ == '__main__':
    main()
