#!/usr/bin/env python3
"""
Calculate and apply systematic bias corrections based on Week 4 results.

This script analyzes the Week 4 expected vs realized cost results to identify
systematic biases per model and applies corrections to improve cost estimates.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vn2.analyze.sequential_planner import Costs


def calculate_model_biases(results_path: Path) -> pd.DataFrame:
    """
    Calculate systematic biases per model from Week 4 results.
    
    Args:
        results_path: Path to Week 4 expected vs realized results CSV
    
    Returns:
        DataFrame with bias corrections per model
    """
    df = pd.read_csv(results_path)
    
    # Calculate bias metrics per model
    model_biases = []
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        # Calculate various bias metrics
        mean_cost_ratio = (model_df['realized_cost'] / np.maximum(model_df['expected_cost'], 0.01)).mean()
        median_cost_ratio = (model_df['realized_cost'] / np.maximum(model_df['expected_cost'], 0.01)).median()
        
        # Separate shortage and holding biases
        shortage_skus = model_df[model_df['shortage_cost'] > 0]
        holding_skus = model_df[model_df['holding_cost'] > 0]
        
        shortage_bias = shortage_skus['shortage_cost'].mean() / np.maximum(shortage_skus['expected_cost'].mean(), 0.01) if len(shortage_skus) > 0 else 1.0
        holding_bias = holding_skus['holding_cost'].mean() / np.maximum(holding_skus['expected_cost'].mean(), 0.01) if len(holding_skus) > 0 else 1.0
        
        # Coverage and calibration metrics
        coverage = model_df['within_ci'].mean()
        target_coverage = 0.9
        coverage_bias = coverage / target_coverage
        
        # Zero prediction rate (major issue)
        zero_pred_rate = (model_df['expected_cost'] < 0.01).mean()
        
        bias_info = {
            'model': model,
            'n_skus': len(model_df),
            'mean_cost_ratio': mean_cost_ratio,
            'median_cost_ratio': median_cost_ratio,
            'shortage_bias': shortage_bias,
            'holding_bias': holding_bias,
            'coverage': coverage,
            'coverage_bias': coverage_bias,
            'zero_pred_rate': zero_pred_rate,
            'total_cost_error': model_df['cost_difference'].sum(),
            # Recommended corrections
            'cost_multiplier': min(max(median_cost_ratio, 0.5), 3.0),  # Bound corrections
            'variance_multiplier': max(1.0 / coverage_bias, 1.0) if coverage_bias > 0 else 2.0,
            'min_cost_floor': 0.1 if zero_pred_rate > 0.1 else 0.01
        }
        
        model_biases.append(bias_info)
    
    return pd.DataFrame(model_biases).sort_values('total_cost_error', ascending=False)


def apply_bias_correction_to_forecasts(
    checkpoints_dir: Path,
    bias_corrections: pd.DataFrame,
    target_skus: pd.DataFrame = None
) -> None:
    """
    Apply bias corrections to forecast checkpoints.
    
    Args:
        checkpoints_dir: Path to forecast checkpoints
        bias_corrections: DataFrame with bias corrections per model
        target_skus: Optional DataFrame with specific SKUs to target
    """
    corrections_applied = 0
    
    for _, bias_row in bias_corrections.iterrows():
        model_name = bias_row['model']
        cost_mult = bias_row['cost_multiplier']
        var_mult = bias_row['variance_multiplier']
        min_floor = bias_row['min_cost_floor']
        
        # Skip models with small biases
        if abs(cost_mult - 1.0) < 0.1 and abs(var_mult - 1.0) < 0.1:
            continue
        
        model_dir = checkpoints_dir / model_name
        if not model_dir.exists():
            continue
        
        print(f"Applying corrections to {model_name}: cost_mult={cost_mult:.2f}, var_mult={var_mult:.2f}")
        
        # Apply corrections to all SKUs for this model
        for sku_dir in model_dir.iterdir():
            if not sku_dir.is_dir():
                continue
            
            # Extract store, product from directory name
            try:
                store, product = map(int, sku_dir.name.split('_'))
            except:
                continue
            
            # Skip if targeting specific SKUs and this isn't one
            if target_skus is not None:
                if not ((target_skus['store'] == store) & (target_skus['product'] == product)).any():
                    continue
            
            # Process all folds for this SKU
            for fold_file in sku_dir.glob('fold_*.pkl'):
                try:
                    with open(fold_file, 'rb') as f:
                        checkpoint = pickle.load(f)
                    
                    if 'quantiles' not in checkpoint:
                        continue
                    
                    quantiles_df = checkpoint['quantiles']
                    
                    # Apply bias corrections
                    corrected_quantiles = quantiles_df.copy()
                    
                    # Apply cost multiplier (shift distribution)
                    corrected_quantiles = corrected_quantiles * cost_mult
                    
                    # Apply variance multiplier (spread distribution)
                    if var_mult != 1.0:
                        medians = corrected_quantiles.median(axis=1)
                        for idx in corrected_quantiles.index:
                            median_val = medians[idx]
                            deviations = corrected_quantiles.loc[idx] - median_val
                            corrected_quantiles.loc[idx] = median_val + deviations * var_mult
                    
                    # Apply minimum floor
                    corrected_quantiles = np.maximum(corrected_quantiles, min_floor)
                    
                    # Ensure monotonicity within each row (quantiles must be non-decreasing)
                    for idx in corrected_quantiles.index:
                        values = corrected_quantiles.loc[idx].values
                        for i in range(1, len(values)):
                            if values[i] < values[i-1]:
                                values[i] = values[i-1]
                        corrected_quantiles.loc[idx] = values
                    
                    # Update checkpoint
                    checkpoint['quantiles'] = corrected_quantiles
                    checkpoint['bias_corrected'] = True
                    checkpoint['corrections_applied'] = {
                        'cost_multiplier': cost_mult,
                        'variance_multiplier': var_mult,
                        'min_floor': min_floor
                    }
                    
                    # Save corrected checkpoint
                    with open(fold_file, 'wb') as f:
                        pickle.dump(checkpoint, f)
                    
                    corrections_applied += 1
                    
                except Exception as e:
                    print(f"Warning: Failed to correct {fold_file}: {e}")
                    continue
    
    print(f"Applied bias corrections to {corrections_applied} checkpoint files")


def create_model_selector_adjustments(
    bias_corrections: pd.DataFrame,
    selector_map_path: Path,
    output_path: Path
) -> None:
    """
    Create adjusted model selector based on bias corrections.
    
    Args:
        bias_corrections: DataFrame with bias corrections per model
        selector_map_path: Path to current selector map
        output_path: Path to save adjusted selector map
    """
    # Load current selector
    selector_df = pd.read_parquet(selector_map_path)
    
    # Create bias-adjusted scores
    bias_dict = bias_corrections.set_index('model')['cost_multiplier'].to_dict()
    
    # Penalize models with high bias
    selector_df['bias_penalty'] = selector_df['model_name'].map(bias_dict).fillna(1.0)
    selector_df['adjusted_score'] = selector_df.get('total_cost_w8', 0) * selector_df['bias_penalty']
    
    # For severely biased models, consider switching to better alternatives
    severely_biased = bias_corrections[bias_corrections['cost_multiplier'] > 2.0]['model'].tolist()
    
    print(f"Severely biased models to consider replacing: {severely_biased}")
    
    # Save adjusted selector (for now, keep original but add bias info)
    selector_df.to_parquet(output_path)
    print(f"Saved bias-adjusted selector to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Apply bias corrections based on Week 4 results')
    parser.add_argument('--apply-corrections', action='store_true', 
                       help='Apply corrections to checkpoint files (destructive!)')
    parser.add_argument('--target-problem-skus', action='store_true',
                       help='Only apply corrections to identified problem SKUs')
    
    args = parser.parse_args()
    
    # Paths
    results_path = Path('models/results/week4_expected_vs_realized.csv')
    problem_skus_path = Path('models/results/problem_skus_analysis.csv')
    checkpoints_dir = Path('models/checkpoints')
    selector_map_path = Path('models/results/selector_map_seq12_v1.parquet')
    
    if not results_path.exists():
        print(f"Error: {results_path} not found. Run Week 4 analysis first.")
        return
    
    print('='*80)
    print('SYSTEMATIC BIAS CORRECTION ANALYSIS')
    print('='*80)
    print()
    
    # Calculate bias corrections
    bias_corrections = calculate_model_biases(results_path)
    
    print('BIAS CORRECTIONS PER MODEL:')
    print('-'*80)
    print(bias_corrections[['model', 'n_skus', 'mean_cost_ratio', 'coverage', 
                           'zero_pred_rate', 'cost_multiplier', 'variance_multiplier', 
                           'min_cost_floor']].to_string(index=False))
    print()
    
    # Save bias analysis
    bias_output_path = Path('models/results/bias_corrections.csv')
    bias_corrections.to_csv(bias_output_path, index=False)
    print(f'Saved bias corrections to: {bias_output_path}')
    print()
    
    # Create adjusted selector
    adjusted_selector_path = Path('models/results/selector_map_bias_adjusted.parquet')
    if selector_map_path.exists():
        create_model_selector_adjustments(bias_corrections, selector_map_path, adjusted_selector_path)
        print()
    
    # Apply corrections to checkpoints if requested
    if args.apply_corrections:
        print('APPLYING BIAS CORRECTIONS TO CHECKPOINTS...')
        print('WARNING: This will modify checkpoint files!')
        print()
        
        target_skus = None
        if args.target_problem_skus and problem_skus_path.exists():
            target_skus = pd.read_csv(problem_skus_path)
            print(f'Targeting {len(target_skus)} problem SKUs for corrections')
        
        apply_bias_correction_to_forecasts(checkpoints_dir, bias_corrections, target_skus)
        print()
        print('✅ Bias corrections applied to checkpoints')
    else:
        print('DRY RUN: Use --apply-corrections to modify checkpoint files')
        print()
        
        # Show what would be corrected
        severe_biases = bias_corrections[bias_corrections['cost_multiplier'] > 1.5]
        if len(severe_biases) > 0:
            print('MODELS THAT WOULD BE CORRECTED:')
            print(severe_biases[['model', 'cost_multiplier', 'zero_pred_rate']].to_string(index=False))


if __name__ == '__main__':
    main()
