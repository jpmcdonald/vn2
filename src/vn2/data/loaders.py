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
    Load initial inventory state (competition week 0).

    Prefers ``Week 0 - 2024-04-08 - Initial State.csv`` (End Inventory, in-transit),
    matching ``szablowski.harness.load_initial_states``. Falls back to In Stock.csv
    for legacy layouts.

    Returns DataFrame with columns: [on_hand, intransit_1, intransit_2]
    """
    raw = Path(raw_dir)
    init_path = raw / "Week 0 - 2024-04-08 - Initial State.csv"
    out = pd.DataFrame(index=index)

    if init_path.exists():
        inv = pd.read_csv(init_path)
        inv = inv.set_index(["Store", "Product"]).reindex(index)
        end_col = "End Inventory" if "End Inventory" in inv.columns else None
        start_col = "Start Inventory" if "Start Inventory" in inv.columns else None
        if end_col:
            oh = inv[end_col].fillna(0).astype(float)
        elif start_col:
            oh = inv[start_col].fillna(0).astype(float)
        else:
            oh = pd.Series(0.0, index=index)
        out["on_hand"] = oh
        if "In Transit W+1" in inv.columns:
            out["intransit_1"] = inv["In Transit W+1"].fillna(0).astype(float)
        else:
            out["intransit_1"] = pd.Series(0.0, index=index)
        if "In Transit W+2" in inv.columns:
            out["intransit_2"] = inv["In Transit W+2"].fillna(0).astype(float)
        else:
            out["intransit_2"] = pd.Series(0.0, index=index)
        return out

    instock = pd.read_csv(raw / "Week 0 - In Stock.csv")
    df = instock.set_index(["Store", "Product"]).reindex(index).fillna(0)
    qty_col = [c for c in df.columns if c not in ["Store", "Product"]][-1]
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

    Columns include Store, Product plus ProductGroup, Division, Department,
    DepartmentGroup, StoreFormat, Format (as in raw Week 0 - Master.csv).
    """
    return pd.read_csv(Path(raw_dir) / "Week 0 - Master.csv")

