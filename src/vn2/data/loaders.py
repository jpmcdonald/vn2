"""Data loading and validation"""

from pathlib import Path
from typing import Optional
import pandas as pd


def submission_index(raw_dir: str) -> pd.MultiIndex:
    """
    Load canonical index from submission template.
    This defines the required ordering for all submissions.
    """
    tpl = pd.read_csv(Path(raw_dir) / "Week 0 - Submission Template.csv")
    return pd.MultiIndex.from_frame(
        tpl[["Store", "Product"]], 
        names=["Store", "Product"]
    )


def load_initial_state(raw_dir: str, index: pd.MultiIndex) -> pd.DataFrame:
    """
    Load initial inventory state.
    
    Returns DataFrame with columns: [on_hand, intransit_1, intransit_2]
    """
    instock = pd.read_csv(Path(raw_dir) / "Week 0 - In Stock.csv")
    df = instock.set_index(["Store", "Product"]).reindex(index).fillna(0)
    
    # Robust to different column naming
    qty_col = [c for c in df.columns if c not in ["Store", "Product"]][-1]
    out = pd.DataFrame(index=index)
    out["on_hand"] = df[qty_col].astype(float)
    out["intransit_1"] = 0.0
    out["intransit_2"] = 0.0
    
    return out


def load_sales(raw_dir: str, week: Optional[int] = None) -> pd.DataFrame:
    """
    Load historical sales data.
    
    Args:
        raw_dir: Path to raw data directory
        week: If specified, load specific week file; else load full history
        
    Returns:
        DataFrame with sales data indexed by (Store, Product) and date columns
    """
    if week is None:
        # Load full historical sales
        sales = pd.read_csv(Path(raw_dir) / "Week 0 - 2024-04-08 - Sales.csv")
    else:
        # Load specific week
        sales = pd.read_csv(Path(raw_dir) / f"Week {week} - 2024-04-08 - Sales.csv")
    
    return sales


def load_master(raw_dir: str) -> pd.DataFrame:
    """
    Load master product/store hierarchy.
    
    Contains: ProductGroup, Division, Department, DepartmentGroup, StoreFormat, Format
    """
    master = pd.read_csv(Path(raw_dir) / "Week 0 - Master.csv")
    return master.set_index(["Store", "Product"])

