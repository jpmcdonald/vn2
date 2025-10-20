"""
Create transform-space winsorized imputed data for non-SLURP models.

Uses SURD transforms per SKU to winsorize extreme imputed values at μ + 3σ
in transform space, then inverts and clips to [50, 500].
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vn2.forecast.transforms import apply_transform, inverse_transform


def winsorize_imputed_data(
    demand_df: pd.DataFrame,
    surd_df: pd.DataFrame,
    z_threshold: float = 3.0,
    min_cap: float = 50.0,
    max_cap: float = 500.0
) -> pd.DataFrame:
    """
    Winsorize imputed demand values in transform space.
    
    Args:
        demand_df: Demand data with 'sales', 'imputed', 'in_stock' columns
        surd_df: SURD transforms with 'Store', 'Product', 'best_transform'
        z_threshold: Number of standard deviations for winsorization
        min_cap: Minimum cap value
        max_cap: Maximum cap value
    
    Returns:
        DataFrame with winsorized sales
    """
    df = demand_df.copy()
    df['sales_winsor'] = df['sales'].copy()
    
    # Process each SKU
    for _, surd_row in surd_df.iterrows():
        store = surd_row['Store']
        product = surd_row['Product']
        transform_name = surd_row['best_transform']
        
        # Get SKU data
        mask = (df['Store'] == store) & (df['Product'] == product)
        sku_data = df.loc[mask].copy()
        
        if len(sku_data) == 0:
            continue
        
        # Get non-stockout observations for statistics
        non_stockout_mask = sku_data['in_stock'] == 1
        non_stockout_sales = sku_data.loc[non_stockout_mask, 'sales'].values
        
        if len(non_stockout_sales) < 3:
            # Too few observations, just clip to [min_cap, max_cap]
            df.loc[mask, 'sales_winsor'] = np.clip(
                sku_data['sales'].values,
                min_cap,
                max_cap
            )
            continue
        
        # Transform non-stockout sales
        sales_trans = apply_transform(non_stockout_sales, transform_name)
        
        # Compute statistics in transform space
        mu_trans = np.mean(sales_trans)
        sigma_trans = np.std(sales_trans)
        
        # Winsorization threshold in transform space
        upper_trans = mu_trans + z_threshold * sigma_trans
        
        # Transform all sales for this SKU
        all_sales_trans = apply_transform(sku_data['sales'].values, transform_name)
        
        # Winsorize in transform space
        all_sales_trans_winsor = np.minimum(all_sales_trans, upper_trans)
        
        # Back-transform
        all_sales_winsor = inverse_transform(all_sales_trans_winsor, transform_name, variance_trans=None)
        
        # Clip to [min_cap, max_cap]
        all_sales_winsor = np.clip(all_sales_winsor, min_cap, max_cap)
        
        # Update only imputed values (preserve non-imputed observations)
        imputed_mask = sku_data['imputed'] == True
        sku_data.loc[imputed_mask, 'sales_winsor'] = all_sales_winsor[imputed_mask]
        
        # Update main dataframe
        df.loc[mask, 'sales_winsor'] = sku_data['sales_winsor'].values
    
    return df


def main():
    print("="*60)
    print("Creating Transform-Space Winsorized Data")
    print("="*60)
    
    # Paths
    demand_path = Path('data/processed/demand_imputed.parquet')
    surd_path = Path('data/processed/surd_transforms.parquet')
    output_path = Path('data/processed/demand_imputed_winsor.parquet')
    
    # Load data
    print("\n📊 Loading data...")
    df = pd.read_parquet(demand_path)
    surd_df = pd.read_parquet(surd_path)
    
    print(f"   Demand: {len(df):,} rows, {len(df[['Store', 'Product']].drop_duplicates())} SKUs")
    print(f"   SURD: {len(surd_df)} SKU transforms")
    print(f"   Imputed: {df['imputed'].sum():,} ({df['imputed'].mean()*100:.1f}%)")
    
    # Check for extreme values
    extreme_mask = df['sales'] > 1000
    if extreme_mask.any():
        print(f"   ⚠️  Extreme values (>1000): {extreme_mask.sum():,}")
        print(f"      Max: {df.loc[extreme_mask, 'sales'].max():,.0f}")
    
    # Winsorize
    print("\n🔧 Winsorizing in transform space (μ + 3σ)...")
    df_winsor = winsorize_imputed_data(df, surd_df, z_threshold=3.0, min_cap=50, max_cap=500)
    
    # Replace 'sales' with 'sales_winsor' and drop the temp column
    df_winsor['sales'] = df_winsor['sales_winsor']
    df_winsor = df_winsor.drop(columns=['sales_winsor'])
    
    # Statistics
    print("\n📈 Winsorization results:")
    changed_mask = df['sales'] != df_winsor['sales']
    print(f"   Values changed: {changed_mask.sum():,} ({changed_mask.mean()*100:.1f}%)")
    print(f"   Max before: {df['sales'].max():,.0f}")
    print(f"   Max after: {df_winsor['sales'].max():,.0f}")
    
    # Check imputed values specifically
    imputed_mask = df['imputed'] == True
    imputed_changed = (df.loc[imputed_mask, 'sales'] != df_winsor.loc[imputed_mask, 'sales']).sum()
    print(f"   Imputed values changed: {imputed_changed:,} ({imputed_changed/imputed_mask.sum()*100:.1f}%)")
    
    # Save
    print(f"\n💾 Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_winsor.to_parquet(output_path)
    
    file_size = output_path.stat().st_size / 1024  # KB
    print(f"   ✅ Saved ({file_size:.0f} KB)")
    
    print("\n" + "="*60)
    print("✅ Winsorized data created successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

