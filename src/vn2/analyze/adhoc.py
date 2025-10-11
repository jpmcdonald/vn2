"""Ad-hoc analysis tools"""

import pandas as pd
import numpy as np
from rich import print as rprint
from rich.table import Table


def describe_sku(
    sales: pd.DataFrame, 
    store: int, 
    product: int
) -> pd.DataFrame:
    """
    Describe sales for a specific SKU.
    
    Args:
        sales: Sales DataFrame
        store: Store ID
        product: Product ID
        
    Returns:
        Summary statistics
    """
    mask = (sales["Store"] == store) & (sales["Product"] == product)
    df = sales[mask]
    
    if len(df) == 0:
        rprint(f"[yellow]No data found for Store {store}, Product {product}[/yellow]")
        return pd.DataFrame()
    
    # Extract numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numeric_cols].describe()
    
    rprint(f"[bold]Store {store}, Product {product}[/bold]")
    rprint(stats)
    
    return stats


def segment_abc(
    df: pd.DataFrame, 
    value_col: str = "revenue",
    a_pct: float = 0.8,
    b_pct: float = 0.15
) -> pd.Series:
    """
    ABC segmentation based on value.
    
    Args:
        df: DataFrame indexed by SKU with value column
        value_col: Column to use for segmentation
        a_pct: Cumulative percentage for A items
        b_pct: Cumulative percentage for B items
        
    Returns:
        Series with ABC labels per SKU
    """
    df = df.sort_values(value_col, ascending=False)
    df["cumsum"] = df[value_col].cumsum()
    df["cum_pct"] = df["cumsum"] / df[value_col].sum()
    
    labels = pd.Series("C", index=df.index)
    labels[df["cum_pct"] <= a_pct] = "A"
    labels[(df["cum_pct"] > a_pct) & (df["cum_pct"] <= a_pct + b_pct)] = "B"
    
    return labels


def segment_xyz(
    df: pd.DataFrame,
    demand_col: str = "demand",
    x_cv: float = 0.5,
    z_cv: float = 1.0
) -> pd.Series:
    """
    XYZ segmentation based on coefficient of variation.
    
    Args:
        df: DataFrame with demand time series
        demand_col: Column with demand values
        x_cv: CV threshold for X items (low variability)
        z_cv: CV threshold for Z items (high variability)
        
    Returns:
        Series with XYZ labels per SKU
    """
    grouped = df.groupby(level=[0, 1])[demand_col]
    
    cv = grouped.std() / (grouped.mean() + 1e-12)
    
    labels = pd.Series("Z", index=cv.index)
    labels[cv <= x_cv] = "X"
    labels[(cv > x_cv) & (cv <= z_cv)] = "Y"
    
    return labels


def print_segments(abc: pd.Series, xyz: pd.Series) -> None:
    """Pretty print segment summary"""
    combined = pd.DataFrame({"ABC": abc, "XYZ": xyz})
    summary = combined.groupby(["ABC", "XYZ"]).size().unstack(fill_value=0)
    
    table = Table(title="Segmentation Summary")
    table.add_column("ABC", style="cyan")
    
    for col in summary.columns:
        table.add_column(str(col), justify="right")
    
    for idx in summary.index:
        row = [str(idx)] + [str(summary.loc[idx, col]) for col in summary.columns]
        table.add_row(*row)
    
    rprint(table)

