"""Submission builder and validator"""

from pathlib import Path
from typing import Optional
import pandas as pd


def build_submission(
    index: pd.MultiIndex, 
    orders: pd.Series, 
    out_path: str,
    validate: bool = True
) -> str:
    """
    Build and validate submission CSV.
    
    Args:
        index: Canonical MultiIndex from submission template
        orders: Order quantities indexed by (Store, Product)
        out_path: Output file path
        validate: Whether to validate before writing
        
    Returns:
        Path to written file
    """
    # Reindex to canonical ordering
    orders_aligned = orders.reindex(index).fillna(0).astype(int)
    
    # Build submission DataFrame
    out = pd.DataFrame(index=index)
    out = out.reset_index()
    out["0"] = orders_aligned.values
    
    if validate:
        validate_submission(out, index)
    
    # Write to file
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    
    return out_path


def validate_submission(df: pd.DataFrame, expected_index: pd.MultiIndex) -> None:
    """
    Validate submission format.
    
    Raises ValueError if invalid.
    """
    # Check shape
    if len(df) != len(expected_index):
        raise ValueError(
            f"Submission has {len(df)} rows, expected {len(expected_index)}"
        )
    
    # Check columns
    expected_cols = ["Store", "Product", "0"]
    if list(df.columns) != expected_cols:
        raise ValueError(
            f"Submission columns {list(df.columns)} != expected {expected_cols}"
        )
    
    # Check Store, Product match expected index
    submission_idx = pd.MultiIndex.from_frame(
        df[["Store", "Product"]], 
        names=["Store", "Product"]
    )
    if not submission_idx.equals(expected_index):
        raise ValueError("Submission index does not match template ordering")
    
    # Check order quantities
    orders = df["0"]
    if orders.isna().any():
        raise ValueError("Submission contains NaN values")
    
    if (orders < 0).any():
        raise ValueError("Submission contains negative orders")
    
    # Success
    print(f"✓ Submission validated: {len(df)} rows, all constraints satisfied")

