#!/usr/bin/env python3
"""
Calculate expected cost for Week 1 orders using h1/h2 forecasts.

This uses the SAME cost calculation that was used to select the orders,
ensuring consistency with the optimization process.

Usage:
    python scripts/calculate_week1_expected_cost.py [--submission PATH]

Arguments:
    --submission: Path to submission CSV (default: data/submissions/jpatrickmcdonald_actual.csv)
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vn2.analyze.sequential_planner import Costs, choose_order_L2
from vn2.analyze.sequential_backtest import quantiles_to_pmf


def calculate_week1_expected_cost(
    submission_path: Path,
    checkpoints_dir: Path,
    state_path: Path,
    selector_map_path: Path,
    costs: Costs,
    quantile_levels: np.ndarray,
    pmf_grain: int = 500
) -> pd.DataFrame:
    """
    Calculate expected cost for Week 1 orders using h1/h2 forecasts.
    
    This replicates the exact cost calculation used during order selection.
    
    Args:
        submission_path: Path to submission CSV with orders
        checkpoints_dir: Path to forecast checkpoints
        state_path: Path to state parquet with initial inventory
        costs: Cost parameters (holding, shortage)
        quantile_levels: Quantile levels for PMF conversion
        pmf_grain: PMF support size
    
    Returns:
        DataFrame with per-SKU expected costs
    """
    # Load submission
    submission = pd.read_csv(submission_path)
    week1_col = '2024-04-15'
    
    # Load state
    state_df = pd.read_parquet(state_path)
    if not isinstance(state_df.index, pd.MultiIndex):
        state_df = state_df.set_index(['Store', 'Product'])
    
    # Load selector map to get the correct model for each SKU
    selector_map = pd.read_parquet(selector_map_path)
    selector_dict = {}
    for _, row in selector_map.iterrows():
        selector_dict[(int(row['store']), int(row['product']))] = row['model_name']
    
    results = []
    
    for _, row in submission.iterrows():
        store = int(row['Store'])
        product = int(row['Product'])
        q_week1 = int(row[week1_col])
        
        # Get initial state
        try:
            state_row = state_df.loc[(store, product)]
            I0 = int(state_row['on_hand'])
            Q1 = int(state_row['intransit_1'])
            Q2 = int(state_row['intransit_2'])
        except KeyError:
            print(f"Warning: No state for ({store}, {product}), skipping")
            continue
        
        # Get the model used for this SKU from selector map
        model_used = selector_dict.get((store, product))
        if model_used is None:
            print(f"Warning: No model selected for ({store}, {product}), skipping")
            continue
        
        # Load forecasts from the selected model
        sku_dir = checkpoints_dir / model_used / f'{store}_{product}'
        if not sku_dir.exists():
            print(f"Warning: No checkpoint for ({store}, {product}) in {model_used}, skipping")
            continue
        
        fold_0_path = sku_dir / 'fold_0.pkl'
        if not fold_0_path.exists():
            print(f"Warning: No fold_0 for ({store}, {product}) in {model_used}, skipping")
            continue
        
        try:
            with open(fold_0_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            quantiles_df = checkpoint.get('quantiles')
            if quantiles_df is None or len(quantiles_df) < 2:
                print(f"Warning: Insufficient quantiles for ({store}, {product}), skipping")
                continue
            
            # Get h1 (horizon 1) and h2 (horizon 2) forecasts
            q1 = quantiles_df.loc[1].values if 1 in quantiles_df.index else quantiles_df.iloc[0].values
            q2 = quantiles_df.loc[2].values if 2 in quantiles_df.index else quantiles_df.iloc[1].values
            
            h1_pmf = quantiles_to_pmf(q1, quantile_levels, pmf_grain)
            h2_pmf = quantiles_to_pmf(q2, quantile_levels, pmf_grain)
            
        except Exception as e:
            print(f"Warning: Failed to load forecast for ({store}, {product}): {e}")
            continue
        
        # Load h3 (week 3 demand) - when the order arrives
        try:
            q3 = quantiles_df.loc[3].values if 3 in quantiles_df.index else quantiles_df.iloc[2].values
            h3_pmf = quantiles_to_pmf(q3, quantile_levels, pmf_grain)
        except:
            # Fallback to h2 if h3 not available
            h3_pmf = h2_pmf
        
        # Let me calculate it directly using the cost logic from choose_order_L2
        from vn2.analyze.sequential_planner import (
            _safe_pmf, _shift_right, _conv_fft, 
            leftover_from_stock_and_demand, expected_pos_neg_from_Z,
            leftover_from_Z
        )
        
        # Step 1: Leftover after week 1
        S0 = I0 + Q1
        L1_pmf = leftover_from_stock_and_demand(S0, h1_pmf)
        
        # Step 2: Inventory at start of week 2
        # Lpre = L1 + Q2 (deterministic Q2)
        Lpre_pmf = _shift_right(L1_pmf, Q2)
        
        # Step 3: Leftover after week 2
        # We need to compute leftover: L2 = max(Lpre - D2, 0)
        # This is tricky with PMFs - need to convolve
        h2_rev = _safe_pmf(h2_pmf)[::-1]
        Z_pmf = _conv_fft(Lpre_pmf, h2_rev)
        z_min = -(len(h2_rev) - 1)
        L2_pmf = leftover_from_Z(Z_pmf, z_min)
        
        # Step 4: Inventory when q_week1 arrives at start of week 3
        inv_week3_pmf = _shift_right(L2_pmf, q_week1)
        
        # Step 5: Cost at week 3 against week 3 demand (h3)
        h3_rev = _safe_pmf(h3_pmf)[::-1]
        final_Z_pmf = _conv_fft(inv_week3_pmf, h3_rev)
        final_z_min = -(len(h3_rev) - 1)
        
        E_over, E_under = expected_pos_neg_from_Z(final_Z_pmf, final_z_min)
        expected_cost_week3 = costs.holding * E_over + costs.shortage * E_under
        
        # Calculate cost distribution for confidence intervals
        # For each possible final inventory level, calculate the cost
        idx = np.arange(len(final_Z_pmf))
        z_vals = final_z_min + idx
        
        # Cost for each outcome: holding * max(z, 0) + shortage * max(-z, 0)
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
        description='Calculate expected cost for Week 1 orders'
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
        default=Path('models/results/week1_expected_costs.csv'),
        help='Path to output CSV'
    )
    
    args = parser.parse_args()
    
    # Parameters
    checkpoints_dir = Path('models/checkpoints')
    state_path = Path('data/interim/state.parquet')
    selector_map_path = Path('models/results/selector_map_seq12_v1.parquet')
    costs = Costs(holding=0.2, shortage=1.0)
    quantile_levels = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 
                                0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    
    print('='*80)
    print('WEEK 1 EXPECTED COST CALCULATION')
    print('='*80)
    print()
    print(f'Submission: {args.submission}')
    print(f'Checkpoints: {checkpoints_dir}')
    print(f'State: {state_path}')
    print(f'Selector Map: {selector_map_path}')
    print(f'Costs: holding={costs.holding}, shortage={costs.shortage}')
    print()
    
    # Calculate
    print('Calculating expected costs...')
    results_df = calculate_week1_expected_cost(
        args.submission,
        checkpoints_dir,
        state_path,
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
    portfolio_5th = results_df['cost_5th'].sum()
    portfolio_95th = results_df['cost_95th'].sum()
    portfolio_std = np.sqrt((results_df['cost_std']**2).sum())  # Sum of variances
    
    print('PORTFOLIO EXPECTED COST:')
    print(f'  Expected: {portfolio_expected:.2f}')
    print(f'  5th percentile: {portfolio_5th:.2f}')
    print(f'  95th percentile: {portfolio_95th:.2f}')
    print(f'  90% Confidence Interval: [{portfolio_5th:.2f}, {portfolio_95th:.2f}]')
    print(f'  Standard Deviation: {portfolio_std:.2f}')
    print(f'  Coefficient of Variation: {portfolio_std/portfolio_expected*100:.1f}%')
    print()
    
    # Per-SKU statistics
    print('PER-SKU STATISTICS:')
    print(f'  Mean Expected Cost: {results_df["expected_cost"].mean():.2f}')
    print(f'  Median Expected Cost: {results_df["expected_cost"].median():.2f}')
    print(f'  Min Expected Cost: {results_df["expected_cost"].min():.2f}')
    print(f'  Max Expected Cost: {results_df["expected_cost"].max():.2f}')
    print()
    
    # Percentiles
    print('Expected Cost Distribution (Per-SKU):')
    for p in [5, 25, 50, 75, 95]:
        val = results_df['expected_cost'].quantile(p/100)
        print(f'  {p}th percentile: {val:.2f}')
    print()
    
    # Save
    results_df.to_csv(args.output, index=False)
    print(f'Saved detailed results to: {args.output}')
    print()
    
    # Top 10 highest cost SKUs
    print('Top 10 Highest Expected Cost SKUs:')
    print('-'*80)
    top10 = results_df.nlargest(10, 'expected_cost')[
        ['store', 'product', 'model', 'order', 'expected_cost']
    ]
    print(top10.to_string(index=False))


if __name__ == '__main__':
    main()

