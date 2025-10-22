#!/usr/bin/env python3
"""
Calculate realized/updated expected cost after observing Week 1 actuals.

After Week 1, we know:
- Actual Week 1 demand (sales)
- Actual Week 1 ending inventory (deterministic, not probabilistic)

This reduces uncertainty for Week 3 cost calculation:
- Week 2 starting inventory is now KNOWN (not probabilistic)
- Only Week 2 and Week 3 demand remain uncertain

Usage:
    python scripts/calculate_week1_realized_cost.py [--submission PATH]
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vn2.analyze.sequential_planner import (
    Costs, _safe_pmf, _shift_right, _conv_fft,
    leftover_from_stock_and_demand, expected_pos_neg_from_Z, leftover_from_Z
)
from vn2.analyze.sequential_backtest import quantiles_to_pmf


def calculate_week1_realized_cost(
    submission_path: Path,
    checkpoints_dir: Path,
    demand_path: Path,
    selector_map_path: Path,
    costs: Costs,
    quantile_levels: np.ndarray,
    week1_date: str = '2024-04-15',
    pmf_grain: int = 500
) -> pd.DataFrame:
    """
    Calculate updated expected cost after observing Week 1 actuals.
    
    With Week 1 actuals known:
    - Week 2 starting inventory is deterministic
    - Only Week 2 and Week 3 demand are uncertain
    - Confidence intervals should be narrower
    
    Args:
        submission_path: Path to submission CSV with orders
        checkpoints_dir: Path to forecast checkpoints
        demand_path: Path to demand data with actuals
        selector_map_path: Path to selector map
        costs: Cost parameters
        quantile_levels: Quantile levels for PMF conversion
        week1_date: Date string for Week 1
        pmf_grain: PMF support size
    
    Returns:
        DataFrame with updated expected costs and confidence intervals
    """
    # Load submission
    submission = pd.read_csv(submission_path)
    
    # Load demand data to get Week 1 actuals
    demand_df = pd.read_parquet(demand_path)
    
    # Find Week 1 actuals - need to map date to demand data
    week1_ts = pd.Timestamp(week1_date)
    week1_actuals = {}
    
    # Check if we have actual data for this week
    week1_data = demand_df[demand_df['week'] == week1_ts]
    
    if len(week1_data) == 0:
        print(f"WARNING: No actual demand data for {week1_date}")
        print(f"Available data range: {demand_df['week'].min()} to {demand_df['week'].max()}")
        print()
        print("Using SIMULATED actuals from h1 forecast median for demonstration...")
        print()
        use_simulated = True
    else:
        use_simulated = False
        for _, row in week1_data.iterrows():
            key = (int(row['Store']), int(row['Product']))
            week1_actuals[key] = int(row['sales'])
    
    # Load selector map
    selector_map = pd.read_parquet(selector_map_path)
    selector_dict = {}
    for _, row in selector_map.iterrows():
        selector_dict[(int(row['store']), int(row['product']))] = row['model_name']
    
    # Load state to get initial inventory
    state_df = pd.read_parquet('data/interim/state.parquet')
    if not isinstance(state_df.index, pd.MultiIndex):
        state_df = state_df.set_index(['Store', 'Product'])
    
    results = []
    
    for _, row in submission.iterrows():
        store = int(row['Store'])
        product = int(row['Product'])
        q_week1 = int(row[week1_date])
        
        # Get Week 1 actual demand (or simulate if not available)
        if use_simulated:
            # Load h1 forecast and use median as simulated actual
            model_used = selector_dict.get((store, product))
            if model_used is None:
                continue
            
            sku_dir = checkpoints_dir / model_used / f'{store}_{product}'
            fold_0_path = sku_dir / 'fold_0.pkl'
            
            if fold_0_path.exists():
                try:
                    with open(fold_0_path, 'rb') as f:
                        checkpoint = pickle.load(f)
                    quantiles_df = checkpoint.get('quantiles')
                    if quantiles_df is not None and len(quantiles_df) >= 1:
                        q1 = quantiles_df.loc[1].values if 1 in quantiles_df.index else quantiles_df.iloc[0].values
                        # Use median (50th percentile) as simulated actual
                        median_idx = np.where(quantile_levels == 0.5)[0]
                        if len(median_idx) > 0:
                            actual_d1 = int(q1[median_idx[0]])
                        else:
                            actual_d1 = int(np.median(q1))
                    else:
                        continue
                except:
                    continue
            else:
                continue
        else:
            actual_d1 = week1_actuals.get((store, product))
            if actual_d1 is None:
                print(f"Warning: No Week 1 actual for ({store}, {product}), skipping")
                continue
        
        # Get initial state
        try:
            state_row = state_df.loc[(store, product)]
            I0 = int(state_row['on_hand'])
            Q1 = int(state_row['intransit_1'])
            Q2 = int(state_row['intransit_2'])
        except KeyError:
            print(f"Warning: No state for ({store}, {product}), skipping")
            continue
        
        # Calculate ACTUAL Week 2 starting inventory (deterministic!)
        S0 = I0 + Q1  # Week 1 starting stock
        actual_L1 = max(0, S0 - actual_d1)  # Actual leftover after Week 1
        week2_start = actual_L1 + Q2  # Week 2 starting inventory (KNOWN)
        
        # Get model and load forecasts
        model_used = selector_dict.get((store, product))
        if model_used is None:
            print(f"Warning: No model selected for ({store}, {product}), skipping")
            continue
        
        sku_dir = checkpoints_dir / model_used / f'{store}_{product}'
        fold_0_path = sku_dir / 'fold_0.pkl'
        
        if not fold_0_path.exists():
            print(f"Warning: No checkpoint for ({store}, {product}), skipping")
            continue
        
        try:
            with open(fold_0_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            quantiles_df = checkpoint.get('quantiles')
            if quantiles_df is None or len(quantiles_df) < 3:
                print(f"Warning: Insufficient quantiles for ({store}, {product}), skipping")
                continue
            
            # Load h2 (Week 2 demand) and h3 (Week 3 demand)
            q2 = quantiles_df.loc[2].values if 2 in quantiles_df.index else quantiles_df.iloc[1].values
            q3 = quantiles_df.loc[3].values if 3 in quantiles_df.index else quantiles_df.iloc[2].values
            
            h2_pmf = quantiles_to_pmf(q2, quantile_levels, pmf_grain)
            h3_pmf = quantiles_to_pmf(q3, quantile_levels, pmf_grain)
            
        except Exception as e:
            print(f"Warning: Failed to load forecast for ({store}, {product}): {e}")
            continue
        
        # NOW: Calculate expected cost with DETERMINISTIC Week 2 starting inventory
        # Week 2 starting inventory is week2_start (known, not probabilistic)
        
        # Step 1: Week 2 ending inventory = max(week2_start - D2, 0)
        # Since week2_start is deterministic, we can use leftover_from_stock_and_demand
        L2_pmf = leftover_from_stock_and_demand(week2_start, h2_pmf)
        
        # Step 2: Week 3 starting inventory = L2 + q_week1
        inv_week3_pmf = _shift_right(L2_pmf, q_week1)
        
        # Step 3: Cost at Week 3 = cost(inv_week3 - D3)
        h3_rev = _safe_pmf(h3_pmf)[::-1]
        final_Z_pmf = _conv_fft(inv_week3_pmf, h3_rev)
        final_z_min = -(len(h3_rev) - 1)
        
        E_over, E_under = expected_pos_neg_from_Z(final_Z_pmf, final_z_min)
        expected_cost_week3 = costs.holding * E_over + costs.shortage * E_under
        
        # Calculate cost distribution for confidence intervals
        idx = np.arange(len(final_Z_pmf))
        z_vals = final_z_min + idx
        
        cost_values = np.where(z_vals > 0, 
                               costs.holding * z_vals,
                               costs.shortage * (-z_vals))
        
        # Calculate percentiles
        sorted_indices = np.argsort(cost_values)
        sorted_costs = cost_values[sorted_indices]
        sorted_pmf = final_Z_pmf[sorted_indices]
        cost_cdf = np.cumsum(sorted_pmf)
        
        cost_5th = sorted_costs[np.searchsorted(cost_cdf, 0.05)]
        cost_95th = sorted_costs[np.searchsorted(cost_cdf, 0.95)]
        cost_std = np.sqrt(np.sum(final_Z_pmf * (cost_values - expected_cost_week3)**2))
        
        results.append({
            'store': store,
            'product': product,
            'model': model_used,
            'order': q_week1,
            'actual_week1_demand': actual_d1,
            'week2_start_inventory': week2_start,
            'expected_cost': expected_cost_week3,
            'cost_5th': cost_5th,
            'cost_95th': cost_95th,
            'cost_std': cost_std,
            'initial_inventory': I0,
            'intransit_1': Q1,
            'intransit_2': Q2
        })
    
    return pd.DataFrame(results)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Calculate updated expected cost after Week 1 actuals'
    )
    parser.add_argument(
        '--submission',
        type=Path,
        default=Path('data/submissions/jpatrickmcdonald_actual.csv'),
        help='Path to submission CSV'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('models/results/week1_realized_costs.csv'),
        help='Path to output CSV'
    )
    
    args = parser.parse_args()
    
    # Parameters
    checkpoints_dir = Path('models/checkpoints')
    demand_path = Path('data/processed/demand_long.parquet')
    selector_map_path = Path('models/results/selector_map_seq12_v1.parquet')
    costs = Costs(holding=0.2, shortage=1.0)
    quantile_levels = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 
                                0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    
    print('='*80)
    print('WEEK 1 REALIZED COST CALCULATION')
    print('(After observing Week 1 actuals)')
    print('='*80)
    print()
    print(f'Submission: {args.submission}')
    print(f'Demand Data: {demand_path}')
    print(f'Checkpoints: {checkpoints_dir}')
    print(f'Selector Map: {selector_map_path}')
    print(f'Costs: holding={costs.holding}, shortage={costs.shortage}')
    print()
    print('NOTE: Week 2 starting inventory is now DETERMINISTIC')
    print('      Only Week 2 and Week 3 demand remain uncertain')
    print()
    
    # Calculate
    print('Calculating updated expected costs...')
    results_df = calculate_week1_realized_cost(
        args.submission,
        checkpoints_dir,
        demand_path,
        selector_map_path,
        costs,
        quantile_levels
    )
    
    print(f'Processed {len(results_df)} SKUs')
    print()
    
    # Load original (ex-ante) results for comparison
    original_path = Path('models/results/week1_expected_costs.csv')
    if original_path.exists():
        original_df = pd.read_csv(original_path)
        
        # Merge to compare
        comparison = results_df[['store', 'product', 'expected_cost', 'cost_5th', 'cost_95th', 'cost_std']].merge(
            original_df[['store', 'product', 'expected_cost', 'cost_5th', 'cost_95th', 'cost_std']],
            on=['store', 'product'],
            suffixes=('_updated', '_original')
        )
    else:
        comparison = None
    
    # Summary
    print('='*80)
    print('UPDATED RESULTS (After Week 1 Actuals)')
    print('='*80)
    print()
    
    # Portfolio-level statistics
    portfolio_expected = results_df['expected_cost'].sum()
    portfolio_5th = results_df['cost_5th'].sum()
    portfolio_95th = results_df['cost_95th'].sum()
    portfolio_std = np.sqrt((results_df['cost_std']**2).sum())
    
    print('PORTFOLIO EXPECTED COST (UPDATED):')
    print(f'  Expected: {portfolio_expected:.2f}')
    print(f'  5th percentile: {portfolio_5th:.2f}')
    print(f'  95th percentile: {portfolio_95th:.2f}')
    print(f'  90% Confidence Interval: [{portfolio_5th:.2f}, {portfolio_95th:.2f}]')
    print(f'  Standard Deviation: {portfolio_std:.2f}')
    print(f'  Coefficient of Variation: {portfolio_std/portfolio_expected*100:.1f}%')
    print()
    
    # Compare with original if available
    if comparison is not None:
        orig_expected = comparison['expected_cost_original'].sum()
        orig_5th = comparison['cost_5th_original'].sum()
        orig_95th = comparison['cost_95th_original'].sum()
        orig_std = np.sqrt((comparison['cost_std_original']**2).sum())
        
        print('='*80)
        print('COMPARISON: Before vs After Week 1')
        print('='*80)
        print()
        print(f'{"Metric":<30} {"Original":<15} {"Updated":<15} {"Change":<15}')
        print('-'*80)
        print(f'{"Expected Cost":<30} {orig_expected:<15.2f} {portfolio_expected:<15.2f} {portfolio_expected-orig_expected:<+15.2f}')
        print(f'{"5th Percentile":<30} {orig_5th:<15.2f} {portfolio_5th:<15.2f} {portfolio_5th-orig_5th:<+15.2f}')
        print(f'{"95th Percentile":<30} {orig_95th:<15.2f} {portfolio_95th:<15.2f} {portfolio_95th-orig_95th:<+15.2f}')
        print(f'{"Std Deviation":<30} {orig_std:<15.2f} {portfolio_std:<15.2f} {portfolio_std-orig_std:<+15.2f}')
        print(f'{"CI Width":<30} {orig_95th-orig_5th:<15.2f} {portfolio_95th-portfolio_5th:<15.2f} {(portfolio_95th-portfolio_5th)-(orig_95th-orig_5th):<+15.2f}')
        print()
        
        # Percentage changes
        print('Percentage Changes:')
        print(f'  Expected Cost: {(portfolio_expected/orig_expected-1)*100:+.1f}%')
        print(f'  Std Deviation: {(portfolio_std/orig_std-1)*100:+.1f}%')
        print(f'  CI Width: {((portfolio_95th-portfolio_5th)/(orig_95th-orig_5th)-1)*100:+.1f}%')
        print()
        
        print('Uncertainty Reduction:')
        print(f'  Original CV: {orig_std/orig_expected*100:.1f}%')
        print(f'  Updated CV: {portfolio_std/portfolio_expected*100:.1f}%')
        print(f'  Reduction: {(1-portfolio_std/portfolio_expected*orig_expected/orig_std)*100:.1f} percentage points')
        print()
    
    # Per-SKU statistics
    print('PER-SKU STATISTICS:')
    print(f'  Mean Expected Cost: {results_df["expected_cost"].mean():.2f}')
    print(f'  Median Expected Cost: {results_df["expected_cost"].median():.2f}')
    print(f'  Min Expected Cost: {results_df["expected_cost"].min():.2f}')
    print(f'  Max Expected Cost: {results_df["expected_cost"].max():.2f}')
    print()
    
    # Save
    results_df.to_csv(args.output, index=False)
    print(f'Saved detailed results to: {args.output}')
    print()
    
    # Top 10 highest cost SKUs
    print('Top 10 Highest Expected Cost SKUs:')
    print('-'*80)
    top10 = results_df.nlargest(10, 'expected_cost')[
        ['store', 'product', 'model', 'order', 'actual_week1_demand', 
         'week2_start_inventory', 'expected_cost']
    ]
    print(top10.to_string(index=False))


if __name__ == '__main__':
    main()

