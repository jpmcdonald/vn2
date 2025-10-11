#!/usr/bin/env python3
"""
Quick example of stockout imputation usage.

Run after completing EDA notebook (02_comprehensive_time_series_eda.ipynb).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from rich import print as rprint

from vn2.uncertainty.stockout_imputation import (
    impute_stockout_sip,
    find_neighbor_profiles
)
from vn2.forecast.imputation_pipeline import (
    create_imputed_training_data,
    compute_imputation_summary
)


def main():
    rprint("[bold blue]Stockout Imputation Example[/bold blue]\n")
    
    # Load data
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    df = pd.read_parquet(data_dir / 'demand_long.parquet')
    surd = pd.read_parquet(data_dir / 'surd_transforms.parquet')
    surd = surd.set_index(['Store', 'Product'])
    
    rprint(f"📊 Loaded {len(df):,} observations")
    rprint(f"   Stockouts: {(~df['in_stock']).sum():,} ({(~df['in_stock']).mean()*100:.1f}%)\n")
    
    # Find an example stockout
    stockout_example = df[df['in_stock'] == False].iloc[0]
    sku_id = (stockout_example['Store'], stockout_example['Product'])
    week = stockout_example['week']
    stock = stockout_example['sales']
    
    rprint(f"[cyan]Example stockout:[/cyan]")
    rprint(f"  SKU: Store {sku_id[0]}, Product {sku_id[1]}")
    rprint(f"  Week: {week}")
    rprint(f"  Censored sales: {stock:.1f}\n")
    
    # Find neighbors
    neighbors = find_neighbor_profiles(sku_id, week, df, n_neighbors=20)
    rprint(f"[cyan]Found {len(neighbors)} neighbor profiles[/cyan]")
    rprint(f"  Distance range: [{neighbors['distance'].min():.3f}, {neighbors['distance'].max():.3f}]")
    rprint(f"  Sales range: [{neighbors['sales'].min():.1f}, {neighbors['sales'].max():.1f}]\n")
    
    # Impute single SKU
    q_levels = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    transform = surd.loc[sku_id, 'best_transform'] if sku_id in surd.index else 'log'
    
    imputed_sip = impute_stockout_sip(
        sku_id=sku_id,
        week=week,
        stock_level=stock,
        q_levels=q_levels,
        df=df,
        transform_name=transform,
        n_neighbors=20
    )
    
    rprint(f"[green]Imputed SIP:[/green]")
    rprint(f"  Transform: {transform}")
    rprint(f"  Median: {imputed_sip.loc[0.5]:.2f} (vs censored {stock:.1f})")
    rprint(f"  Mean: {imputed_sip.mean():.2f}")
    rprint(f"  80% PI: [{imputed_sip.loc[0.1]:.2f}, {imputed_sip.loc[0.9]:.2f}]")
    rprint(f"  90% PI: [{imputed_sip.loc[0.05]:.2f}, {imputed_sip.loc[0.95]:.2f}]\n")
    
    # Bulk imputation
    rprint("[cyan]Imputing all stockouts...[/cyan]")
    df_imputed = create_imputed_training_data(
        df, surd, q_levels, n_neighbors=20, verbose=False
    )
    
    summary = compute_imputation_summary(df, df_imputed)
    
    rprint("\n[bold green]Summary:[/bold green]")
    for _, row in summary.iterrows():
        metric = row['metric']
        value = row['value']
        if 'pct' in metric:
            rprint(f"  {metric}: [yellow]{value:.1f}%[/yellow]")
        elif 'lift' in metric:
            rprint(f"  {metric}: [green]+{value:.2f}[/green]")
        elif isinstance(value, (int, float)) and value > 100:
            rprint(f"  {metric}: {value:,.0f}")
        else:
            rprint(f"  {metric}: {value:.2f}")
    
    rprint(f"\n[bold green]✅ Complete![/bold green]")
    rprint(f"   Use df_imputed for model training")
    rprint(f"   Imputed weeks have 'imputed' flag = True")


if __name__ == "__main__":
    main()

