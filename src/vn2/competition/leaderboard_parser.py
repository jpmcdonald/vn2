"""
Leaderboard parser for competition analysis.

Parses competition leaderboard text files and extracts structured performance data.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_leaderboard_file(file_path: Path) -> pd.DataFrame:
    """
    Parse a leaderboard text file into structured DataFrame.
    
    Args:
        file_path: Path to leaderboard text file
    
    Returns:
        DataFrame with competitor performance data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into lines and clean
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    print(f"Debug: Parsing {file_path.name}, found {len(lines)} non-empty lines")
    if lines:
        print(f"Debug: First few lines: {lines[:10]}")
    
    competitors = []
    current_competitor = {}
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Skip header and comment lines
        if 'Data Scientist' in line or line.startswith('#'):
            i += 1
            continue
        
        # Check if line is a rank number
        if line.isdigit():
            rank = int(line)
            i += 1
            
            # Skip empty line after rank
            while i < len(lines) and not lines[i].strip():
                i += 1
            
            if i >= len(lines):
                break
            
            # Next line should be competitor name
            name_line = lines[i]
            i += 1
            
            # Next line should be the data
            if i < len(lines):
                data_line = lines[i]
                
                # Parse data line format: "apply_count	order_cost	cumulative_cost	entries	last_seen	team"
                parts = data_line.split('\t')
                
                if len(parts) >= 3:  # Minimum: apply_count, order_cost, cumulative_cost
                    try:
                        apply_count = int(parts[0]) if parts[0].isdigit() else 1
                        order_cost = float(parts[1])
                        cumulative_cost = float(parts[2])
                        entries = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 1
                        last_seen = parts[4] if len(parts) > 4 else ''
                        
                        # Handle country in name (some names have country codes)
                        country = ''
                        clean_name = name_line
                        
                        # Extract country if it's at the end of the name
                        country_codes = ['Uganda', 'Albania']  # Add more as needed
                        for code in country_codes:
                            if code in name_line:
                                country = code
                                clean_name = name_line.replace(code, '').strip()
                                break
                        
                        competitor = {
                            'rank': rank,
                            'name': clean_name,
                            'country': country,
                            'apply_count': apply_count,
                            'order_cost': order_cost,
                            'cumulative_cost': cumulative_cost,
                            'entries': entries,
                            'last_seen': last_seen
                        }
                        
                        competitors.append(competitor)
                        
                    except (ValueError, IndexError) as e:
                        # Skip malformed entries
                        print(f"Warning: Skipped malformed entry at rank {rank}: {e}")
                        pass
            
            i += 1
        else:
            i += 1
    
    if not competitors:
        return pd.DataFrame()
    
    df = pd.DataFrame(competitors)
    
    # Clean up data types
    df['rank'] = df['rank'].astype(int)
    df['order_cost'] = pd.to_numeric(df['order_cost'], errors='coerce')
    df['cumulative_cost'] = pd.to_numeric(df['cumulative_cost'], errors='coerce')
    df['apply_count'] = pd.to_numeric(df['apply_count'], errors='coerce').fillna(1).astype(int)
    df['entries'] = pd.to_numeric(df['entries'], errors='coerce').fillna(1).astype(int)
    
    # Extract week from filename
    week_match = re.search(r'Week(\d+)', file_path.name)
    if week_match:
        df['week'] = int(week_match.group(1))
    else:
        df['week'] = 0
    
    return df


def parse_all_leaderboards(leaderboards_dir: Path) -> pd.DataFrame:
    """
    Parse all leaderboard files in directory.
    
    Args:
        leaderboards_dir: Directory containing leaderboard files
    
    Returns:
        Combined DataFrame with all leaderboard data
    """
    all_leaderboards = []
    
    for file_path in leaderboards_dir.glob('Week*.txt'):
        try:
            df = parse_leaderboard_file(file_path)
            if not df.empty:
                all_leaderboards.append(df)
                print(f"✅ Parsed {file_path.name}: {len(df)} competitors")
            else:
                print(f"⚠️  Empty results from {file_path.name}")
        except Exception as e:
            print(f"❌ Failed to parse {file_path.name}: {e}")
    
    if not all_leaderboards:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_leaderboards, ignore_index=True)
    
    # Sort by week and rank
    combined_df = combined_df.sort_values(['week', 'rank']).reset_index(drop=True)
    
    return combined_df


def calculate_performance_metrics(leaderboard_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate performance metrics for competition analysis.
    
    Args:
        leaderboard_df: Combined leaderboard DataFrame
    
    Returns:
        DataFrame with performance metrics per competitor
    """
    if leaderboard_df.empty:
        return pd.DataFrame()
    
    metrics = []
    
    for name in leaderboard_df['name'].unique():
        competitor_data = leaderboard_df[leaderboard_df['name'] == name].sort_values('week')
        
        if len(competitor_data) == 0:
            continue
        
        # Basic metrics
        weeks_participated = competitor_data['week'].tolist()
        ranks = competitor_data['rank'].tolist()
        costs = competitor_data['order_cost'].tolist()
        cumulative_costs = competitor_data['cumulative_cost'].tolist()
        
        # Performance metrics
        final_rank = ranks[-1] if ranks else None
        final_cumulative_cost = cumulative_costs[-1] if cumulative_costs else None
        
        # Rank stability (if multiple weeks)
        rank_volatility = np.std(ranks) if len(ranks) > 1 else 0
        cost_volatility = np.std(costs) if len(costs) > 1 else 0
        
        # Performance trend (improving/declining)
        if len(ranks) > 1:
            rank_trend = ranks[-1] - ranks[0]  # Positive = getting worse
            cost_trend = np.polyfit(range(len(costs)), costs, 1)[0] if len(costs) > 1 else 0
        else:
            rank_trend = 0
            cost_trend = 0
        
        # Submission behavior
        total_entries = competitor_data['entries'].sum()
        avg_applies_per_week = competitor_data['apply_count'].mean()
        
        # Risk profile (based on cost patterns)
        if len(costs) > 1:
            cost_cv = np.std(costs) / max(np.mean(costs), 0.01)  # Coefficient of variation
        else:
            cost_cv = 0
        
        metrics.append({
            'name': name,
            'weeks_participated': len(weeks_participated),
            'weeks_list': weeks_participated,
            'final_rank': final_rank,
            'final_cumulative_cost': final_cumulative_cost,
            'rank_volatility': rank_volatility,
            'cost_volatility': cost_volatility,
            'rank_trend': rank_trend,  # Positive = declining performance
            'cost_trend': cost_trend,  # Positive = increasing costs
            'total_entries': total_entries,
            'avg_applies_per_week': avg_applies_per_week,
            'cost_cv': cost_cv,
            'mean_weekly_cost': np.mean(costs),
            'min_weekly_cost': np.min(costs),
            'max_weekly_cost': np.max(costs),
            'country': competitor_data['country'].iloc[0] if 'country' in competitor_data.columns else ''
        })
    
    return pd.DataFrame(metrics)


def identify_our_performance(leaderboard_df: pd.DataFrame, our_name: str = "Patrick McDonald") -> Dict:
    """
    Identify our performance in the leaderboards.
    
    Args:
        leaderboard_df: Combined leaderboard DataFrame
        our_name: Our competitor name
    
    Returns:
        Dict with our performance metrics
    """
    our_data = leaderboard_df[leaderboard_df['name'] == our_name]
    
    if our_data.empty:
        return {'found': False, 'message': f"Competitor '{our_name}' not found in leaderboards"}
    
    our_performance = {}
    
    for _, row in our_data.iterrows():
        week = row['week']
        our_performance[f'week_{week}'] = {
            'rank': row['rank'],
            'order_cost': row['order_cost'],
            'cumulative_cost': row['cumulative_cost'],
            'total_competitors': leaderboard_df[leaderboard_df['week'] == week]['rank'].max()
        }
    
    our_performance['found'] = True
    return our_performance


def analyze_rank_changes(leaderboard_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze rank changes between weeks to measure stability.
    
    Args:
        leaderboard_df: Combined leaderboard DataFrame
    
    Returns:
        DataFrame with rank change analysis
    """
    if leaderboard_df.empty:
        return pd.DataFrame()
    
    # Get competitors who participated in multiple weeks
    week_counts = leaderboard_df['name'].value_counts()
    multi_week_competitors = week_counts[week_counts > 1].index
    
    rank_changes = []
    
    for name in multi_week_competitors:
        competitor_data = leaderboard_df[leaderboard_df['name'] == name].sort_values('week')
        
        weeks = competitor_data['week'].tolist()
        ranks = competitor_data['rank'].tolist()
        
        # Calculate rank changes between consecutive weeks
        for i in range(len(weeks) - 1):
            week_from = weeks[i]
            week_to = weeks[i + 1]
            rank_from = ranks[i]
            rank_to = ranks[i + 1]
            
            rank_change = rank_to - rank_from  # Positive = worse rank
            
            rank_changes.append({
                'name': name,
                'week_from': week_from,
                'week_to': week_to,
                'rank_from': rank_from,
                'rank_to': rank_to,
                'rank_change': rank_change,
                'rank_change_abs': abs(rank_change),
                'improved': rank_change < 0,  # Lower rank number = better
                'declined': rank_change > 0
            })
    
    return pd.DataFrame(rank_changes)


def main():
    """Test the leaderboard parser."""
    leaderboards_dir = Path('data/raw/leaderboards')
    
    if not leaderboards_dir.exists():
        print(f"Error: {leaderboards_dir} not found")
        return
    
    print('='*80)
    print('LEADERBOARD PARSING TEST')
    print('='*80)
    print()
    
    # Parse all leaderboards
    combined_df = parse_all_leaderboards(leaderboards_dir)
    
    if combined_df.empty:
        print("No leaderboard data parsed")
        return
    
    print(f"Total records: {len(combined_df)}")
    print(f"Unique competitors: {combined_df['name'].nunique()}")
    print(f"Weeks covered: {sorted(combined_df['week'].unique())}")
    print()
    
    # Calculate performance metrics
    metrics_df = calculate_performance_metrics(combined_df)
    
    print("TOP 10 PERFORMERS (by final cumulative cost):")
    print('-'*80)
    top_10 = metrics_df.nsmallest(10, 'final_cumulative_cost')
    print(top_10[['name', 'final_rank', 'final_cumulative_cost', 'rank_volatility', 'cost_cv']].to_string(index=False))
    print()
    
    # Find our performance
    our_performance = identify_our_performance(combined_df)
    print("OUR PERFORMANCE:")
    print('-'*80)
    if our_performance['found']:
        for week_key, week_data in our_performance.items():
            if week_key.startswith('week_'):
                week = week_key.split('_')[1]
                print(f"Week {week}: Rank {week_data['rank']}/{week_data['total_competitors']} "
                      f"(Cost: {week_data['order_cost']:.1f}, Cumulative: {week_data['cumulative_cost']:.1f})")
    else:
        print(our_performance['message'])
    print()
    
    # Analyze rank changes
    rank_changes_df = analyze_rank_changes(combined_df)
    
    if not rank_changes_df.empty:
        print("RANK STABILITY ANALYSIS:")
        print('-'*80)
        print(f"Competitors with multiple weeks: {rank_changes_df['name'].nunique()}")
        print(f"Average rank change (abs): {rank_changes_df['rank_change_abs'].mean():.1f}")
        print(f"% who improved rank: {rank_changes_df['improved'].mean()*100:.1f}%")
        print(f"% who declined rank: {rank_changes_df['declined'].mean()*100:.1f}%")
        print()
        
        # Most stable performers
        stability = rank_changes_df.groupby('name')['rank_change_abs'].mean().sort_values()
        print("MOST STABLE PERFORMERS (smallest rank changes):")
        print(stability.head(10).to_string())
    
    # Save parsed data
    output_path = Path('models/results/competition_analysis/leaderboard_data.csv')
    combined_df.to_csv(output_path, index=False)
    
    metrics_path = Path('models/results/competition_analysis/competitor_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    
    if not rank_changes_df.empty:
        changes_path = Path('models/results/competition_analysis/rank_changes.csv')
        rank_changes_df.to_csv(changes_path, index=False)
    
    print(f"\n✅ Saved parsed data to:")
    print(f"   {output_path}")
    print(f"   {metrics_path}")
    if not rank_changes_df.empty:
        print(f"   {changes_path}")


if __name__ == '__main__':
    main()
