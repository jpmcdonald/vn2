#!/usr/bin/env python3
"""
Calculate Week 4 expected vs realized costs.

Week 4 costs come from orders placed in Week 2 (L=2 lead time).
We need to:
1. Load Week 2 orders (from order2_jpatrickmcdonald.csv)
2. Load Week 2 state (state2.csv)
3. Load Week 2 forecasts (fold_1.pkl for Week 2 decision epoch)
4. Calculate expected cost for Week 4
5. Compare with realized cost from state4.csv

Usage:
    python scripts/calculate_week4_expected_cost.py
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


def calculate_week4_expected_cost(
    order2_path: Path,
    state2_path: Path,
    state4_path: Path,
    checkpoints_dir: Path,
    selector_map_path: Path,
    costs: Costs,
    quantile_levels: np.ndarray,
    pmf_grain: int = 500
) -> pd.DataFrame:
    """
    Calculate expected cost for Week 4 using Week 2 forecasts.
    
    Week 2 orders arrive at Week 4 (L=2).
    We need to calculate what we expected Week 4 cost to be when
    we placed orders in Week 2.
    
    Args:
        order2_path: Path to Week 2 order submission
        state2_path: Path to Week 2 state (for initial inventory)
        state4_path: Path to Week 4 state (for realized costs)
        checkpoints_dir: Path to forecast checkpoints
        selector_map_path: Path to selector map
        costs: Cost parameters
        quantile_levels: Quantile levels for PMF conversion
        pmf_grain: PMF support size
    
    Returns:
        DataFrame with expected vs realized costs
    """
    # Load Week 2 orders
    order2_df = pd.read_csv(order2_path)
    # Week 2 submission has column '0' for the order quantity
    week2_col = '0' if '0' in order2_df.columns else order2_df.columns[2]
    
    # Load Week 2 state
    state2_df = pd.read_csv(state2_path)
    state2_dict = {}
    for _, row in state2_df.iterrows():
        key = (int(row['Store']), int(row['Product']))
        # Week 2 ending state:
        # - End Inventory = on_hand at end of Week 2 (start of Week 3)
        # - In Transit W+1 = order arriving at Week 3
        # - In Transit W+2 = order arriving at Week 4 (Week 2 order)
        state2_dict[key] = {
            'on_hand': int(row['End Inventory']),
            'intransit_1': int(row['In Transit W+1']),
            'intransit_2': int(row['In Transit W+2'])
        }
    
    # Load Week 4 state for realized costs
    state4_df = pd.read_csv(state4_path)
    state4_dict = {}
    for _, row in state4_df.iterrows():
        key = (int(row['Store']), int(row['Product']))
        state4_dict[key] = {
            'holding_cost': float(row['Holding Cost']),
            'shortage_cost': float(row['Shortage Cost']),
            'realized_cost': float(row['Holding Cost']) + float(row['Shortage Cost']),
            'actual_demand': int(row['Sales']),
            'start_inventory': int(row['Start Inventory'])
        }
    
    # Load selector map
    selector_map = pd.read_parquet(selector_map_path)
    selector_dict = {}
    for _, row in selector_map.iterrows():
        selector_dict[(int(row['store']), int(row['product']))] = row['model_name']
    
    results = []
    
    for _, row in order2_df.iterrows():
        store = int(row['Store'])
        product = int(row['Product'])
        q_week2 = int(row[week2_col])  # Order placed in Week 2
        
        # Get Week 2 ending state (start of Week 3)
        state2 = state2_dict.get((store, product))
        if state2 is None:
            continue
        
        I0 = state2['on_hand']  # Week 3 starting inventory
        Q1 = state2['intransit_1']  # Arrives at Week 3
        Q2 = state2['intransit_2']  # Arrives at Week 4 (this should match q_week2)
        
        # Get Week 4 realized cost
        state4 = state4_dict.get((store, product))
        if state4 is None:
            continue
        
        # Get model used for this SKU
        model_used = selector_dict.get((store, product))
        if model_used is None:
            continue
        
        # Load Week 2 forecasts (fold_1.pkl for Week 2 decision epoch)
        # fold_0 = Week 1 forecast, fold_1 = Week 2 forecast, etc.
        sku_dir = checkpoints_dir / model_used / f'{store}_{product}'
        fold_1_path = sku_dir / 'fold_1.pkl'
        
        if not fold_1_path.exists():
            continue
        
        try:
            with open(fold_1_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            quantiles_df = checkpoint.get('quantiles')
            if quantiles_df is None or len(quantiles_df) < 2:
                continue
            
            # Week 2 decision epoch:
            # - h1 = Week 3 demand (when Q1 arrives)
            # - h2 = Week 4 demand (when Q2/q_week2 arrives)
            q1 = quantiles_df.loc[1].values if 1 in quantiles_df.index else quantiles_df.iloc[0].values
            q2 = quantiles_df.loc[2].values if 2 in quantiles_df.index else quantiles_df.iloc[1].values
            
            h1_pmf = quantiles_to_pmf(q1, quantile_levels, pmf_grain)  # Week 3 demand
            h2_pmf = quantiles_to_pmf(q2, quantile_levels, pmf_grain)  # Week 4 demand
            
        except Exception as e:
            continue
        
        # Calculate expected cost for Week 4
        # Starting from Week 3 state:
        # Week 3 starting inventory: I0
        # Week 3: Q1 arrives, then demand h1 (Week 3)
        S3 = I0 + Q1  # Week 3 starting stock
        L3_pmf = leftover_from_stock_and_demand(S3, h1_pmf)  # Leftover after Week 3
        
        # Week 4 starting inventory: L3 + Q2 (or q_week2)
        # Use Q2 from state (should be same as q_week2)
        inv_week4_pmf = _shift_right(L3_pmf, Q2)
        
        # Week 4 cost: cost(inv_week4 - D4)
        # where D4 = h2 (Week 4 demand)
        h4_rev = _safe_pmf(h2_pmf)[::-1]
        final_Z_pmf = _conv_fft(inv_week4_pmf, h4_rev)
        final_z_min = -(len(h4_rev) - 1)
        
        E_over, E_under = expected_pos_neg_from_Z(final_Z_pmf, final_z_min)
        expected_cost_week4 = costs.holding * E_over + costs.shortage * E_under
        
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
        cost_std = np.sqrt(np.sum(final_Z_pmf * (cost_values - expected_cost_week4)**2))
        
        results.append({
            'store': store,
            'product': product,
            'model': model_used,
            'order_week2': q_week2,
            'expected_cost': expected_cost_week4,
            'cost_5th': cost_5th,
            'cost_95th': cost_95th,
            'cost_std': cost_std,
            'realized_cost': state4['realized_cost'],
            'holding_cost': state4['holding_cost'],
            'shortage_cost': state4['shortage_cost'],
            'actual_demand': state4['actual_demand'],
            'week4_start_inventory': state4['start_inventory'],
            'cost_difference': state4['realized_cost'] - expected_cost_week4,
            'within_ci': (cost_5th <= state4['realized_cost'] <= cost_95th)
        })
    
    return pd.DataFrame(results)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Calculate Week 4 expected vs realized costs'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('models/results/week4_expected_vs_realized.csv'),
        help='Path to output CSV'
    )
    
    args = parser.parse_args()
    
    # Parameters
    order2_path = Path('data/submissions/order2_jpatrickmcdonald.csv')
    state2_path = Path('data/states/state2.csv')
    state4_path = Path('data/states/state4.csv')
    checkpoints_dir = Path('models/checkpoints')
    selector_map_path = Path('models/results/selector_map_seq12_v1.parquet')
    costs = Costs(holding=0.2, shortage=1.0)
    quantile_levels = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 
                                0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    
    print('='*80)
    print('WEEK 4 EXPECTED VS REALIZED COST ANALYSIS')
    print('='*80)
    print()
    print(f'Week 2 Orders: {order2_path}')
    print(f'Week 2 State: {state2_path}')
    print(f'Week 4 State: {state4_path}')
    print(f'Checkpoints: {checkpoints_dir}')
    print(f'Selector Map: {selector_map_path}')
    print(f'Costs: holding={costs.holding}, shortage={costs.shortage}')
    print()
    
    # Calculate
    print('Calculating expected costs...')
    results_df = calculate_week4_expected_cost(
        order2_path,
        state2_path,
        state4_path,
        checkpoints_dir,
        selector_map_path,
        costs,
        quantile_levels
    )
    
    print(f'Processed {len(results_df)} SKUs')
    print()
    
    # Summary
    print('='*80)
    print('RESULTS')
    print('='*80)
    print()
    
    # Portfolio-level statistics
    portfolio_expected = results_df['expected_cost'].sum()
    portfolio_realized = results_df['realized_cost'].sum()
    portfolio_5th = results_df['cost_5th'].sum()
    portfolio_95th = results_df['cost_95th'].sum()
    portfolio_std = np.sqrt((results_df['cost_std']**2).sum())
    
    print('PORTFOLIO COSTS (Week 4):')
    print(f'  Expected Cost: {portfolio_expected:.2f}')
    print(f'  Realized Cost: {portfolio_realized:.2f}')
    print(f'  Difference: {portfolio_realized - portfolio_expected:+.2f}')
    print(f'  Percentage: {(portfolio_realized/portfolio_expected - 1)*100:+.1f}%')
    print()
    print(f'  90% Confidence Interval: [{portfolio_5th:.2f}, {portfolio_95th:.2f}]')
    print(f'  Standard Deviation: {portfolio_std:.2f}')
    print()
    
    # Check if realized is within CI
    in_ci = (portfolio_5th <= portfolio_realized <= portfolio_95th)
    print(f'  Realized within 90% CI: {"YES" if in_ci else "NO"}')
    if not in_ci:
        if portfolio_realized < portfolio_5th:
            print(f'  Realized is BELOW 5th percentile (better than expected)')
        else:
            print(f'  Realized is ABOVE 95th percentile (worse than expected)')
    print()
    
    # Coverage statistics
    coverage = results_df['within_ci'].mean() * 100
    print(f'Coverage: {coverage:.1f}% of SKUs have realized cost within 90% CI')
    print()
    
    # Cost breakdown
    print('COST BREAKDOWN:')
    print(f'  Total Holding Cost: {results_df["holding_cost"].sum():.2f}')
    print(f'  Total Shortage Cost: {results_df["shortage_cost"].sum():.2f}')
    print()
    
    # Top contributors
    print('Top 10 SKUs by Cost Difference (Realized - Expected):')
    print('-'*80)
    top_diff = results_df.nlargest(10, 'cost_difference')[
        ['store', 'product', 'model', 'expected_cost', 'realized_cost', 
         'cost_difference', 'actual_demand', 'week4_start_inventory']
    ]
    print(top_diff.to_string(index=False))
    print()
    
    # Save
    results_df.to_csv(args.output, index=False)
    print(f'Saved detailed results to: {args.output}')


if __name__ == '__main__':
    main()

