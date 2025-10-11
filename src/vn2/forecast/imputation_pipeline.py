"""
Pipeline to integrate stockout imputation into forecasting workflow.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from vn2.uncertainty.stockout_imputation import (
    impute_all_stockouts,
    impute_stockout_sip
)


def create_imputed_training_data(
    df: pd.DataFrame,
    surd_transforms: pd.DataFrame,
    q_levels: np.ndarray,
    n_neighbors: int = 20,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Replace stockout observations with imputed demand for model training.
    
    Args:
        df: Original demand data with stockouts
        surd_transforms: Per-SKU transforms from SURD analysis
        q_levels: Quantile levels for SIP reconstruction
        n_neighbors: Number of neighbor profiles for matching
        verbose: Print progress
        
    Returns:
        DataFrame with 'sales' column adjusted for stockouts
    """
    df_imputed = df.copy()
    
    # Add imputed flag
    df_imputed['imputed'] = False
    
    # Impute all stockouts
    imputed_sips = impute_all_stockouts(
        df, surd_transforms, q_levels, 
        n_neighbors=n_neighbors, verbose=verbose
    )
    
    # Replace point estimates with median of imputed SIP
    for (store, product, week), sip in imputed_sips.items():
        mask = (df_imputed['Store'] == store) & \
               (df_imputed['Product'] == product) & \
               (df_imputed['week'] == week)
        
        # Use median of imputed SIP as point estimate
        imputed_demand = sip.loc[0.5]  # median
        df_imputed.loc[mask, 'sales'] = imputed_demand
        df_imputed.loc[mask, 'imputed'] = True
    
    return df_imputed


def create_imputed_sip_library(
    df: pd.DataFrame,
    surd_transforms: pd.DataFrame,
    q_levels: np.ndarray,
    horizon: int = 8,
    n_neighbors: int = 20
) -> Dict[int, Dict[Tuple[int, int], pd.Series]]:
    """
    Create forward-looking SIP library with stockout imputation.
    
    For each SKU at each forecast origin, generate SIPs for h=1..horizon
    weeks ahead, with stockout weeks replaced by imputed SIPs.
    
    Args:
        df: Historical demand
        surd_transforms: Transform assignments
        q_levels: Quantile levels
        horizon: Forecast horizon
        n_neighbors: Number of neighbor profiles
        
    Returns:
        Nested dict: {horizon_step: {(store, product): SIP}}
    """
    # Create imputed training data
    df_imputed = create_imputed_training_data(
        df, surd_transforms, q_levels, n_neighbors, verbose=False
    )
    
    sip_library = {h: {} for h in range(1, horizon + 1)}
    
    # TODO: Integrate with your forecasting models
    # This is a placeholder showing the structure
    # When generating prediction intervals, if a training week was imputed,
    # the model has seen "true demand" estimates rather than censored sales
    
    print(f"📦 SIP library structure ready for {horizon} horizon steps")
    print(f"   Use df_imputed for training models - {df_imputed['imputed'].sum()} weeks imputed")
    
    return sip_library


def compute_imputation_summary(
    df_original: pd.DataFrame,
    df_imputed: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute summary statistics comparing original vs imputed demand.
    
    Args:
        df_original: Original data with stockouts
        df_imputed: Data with imputed values
        
    Returns:
        Summary DataFrame with comparison metrics
    """
    imputed_mask = df_imputed['imputed'] == True
    
    if imputed_mask.sum() == 0:
        print("No imputations found")
        return pd.DataFrame()
    
    summary = pd.DataFrame({
        'metric': [
            'n_imputed',
            'pct_imputed',
            'mean_original_sales',
            'mean_imputed_sales',
            'mean_lift',
            'median_original_sales',
            'median_imputed_sales',
            'median_lift',
            'total_original_volume',
            'total_imputed_volume',
            'volume_lift'
        ],
        'value': [
            imputed_mask.sum(),
            imputed_mask.mean() * 100,
            df_original.loc[imputed_mask, 'sales'].mean(),
            df_imputed.loc[imputed_mask, 'sales'].mean(),
            df_imputed.loc[imputed_mask, 'sales'].mean() - df_original.loc[imputed_mask, 'sales'].mean(),
            df_original.loc[imputed_mask, 'sales'].median(),
            df_imputed.loc[imputed_mask, 'sales'].median(),
            df_imputed.loc[imputed_mask, 'sales'].median() - df_original.loc[imputed_mask, 'sales'].median(),
            df_original.loc[imputed_mask, 'sales'].sum(),
            df_imputed.loc[imputed_mask, 'sales'].sum(),
            df_imputed.loc[imputed_mask, 'sales'].sum() - df_original.loc[imputed_mask, 'sales'].sum()
        ]
    })
    
    return summary


def save_imputation_artifacts(
    df_imputed: pd.DataFrame,
    imputed_sips: Dict[Tuple[int, int, int], pd.Series],
    summary: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Save all imputation artifacts for downstream use.
    
    Args:
        df_imputed: Imputed demand data
        imputed_sips: Full SIP library for stockout weeks
        summary: Summary statistics
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save imputed data
    df_imputed.to_parquet(output_path / 'demand_imputed.parquet')
    
    # Save SIPs as separate file
    sip_df_list = []
    for (store, product, week), sip in imputed_sips.items():
        for q_level, q_value in sip.items():
            sip_df_list.append({
                'Store': store,
                'Product': product,
                'week': week,
                'quantile': q_level,
                'value': q_value
            })
    
    if len(sip_df_list) > 0:
        sip_df = pd.DataFrame(sip_df_list)
        sip_df.to_parquet(output_path / 'imputed_sips.parquet')
    
    # Save summary
    summary.to_csv(output_path / 'imputation_summary.csv', index=False)
    
    print(f"💾 Saved imputation artifacts to {output_dir}")
    print(f"   - demand_imputed.parquet ({len(df_imputed)} rows)")
    print(f"   - imputed_sips.parquet ({len(imputed_sips)} SIPs)")
    print(f"   - imputation_summary.csv")

