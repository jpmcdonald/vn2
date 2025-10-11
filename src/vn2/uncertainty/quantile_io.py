"""Quantile function I/O for SIP construction"""

from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np


def load_quantiles(
    dir_path: str, 
    horizon: int, 
    q_levels: List[float]
) -> Dict[int, pd.DataFrame]:
    """
    Load quantile tables from disk.
    
    Expected format: week+{t}.parquet with columns for quantile levels
    
    Args:
        dir_path: Directory containing quantile files
        horizon: Number of weeks ahead to load
        q_levels: Expected quantile levels (e.g., [0.01, 0.05, ..., 0.99])
        
    Returns:
        Dict mapping week offset (1-based) to quantile DataFrames
    """
    out = {}
    
    for t in range(1, horizon + 1):
        fp = Path(dir_path) / f"week+{t}.parquet"
        
        if not fp.exists():
            raise FileNotFoundError(f"Missing quantile file: {fp}")
        
        Q = pd.read_parquet(fp)
        
        # Normalize column names to floats
        try:
            Q.columns = [float(str(c).replace("q_", "").replace("_", ".")) for c in Q.columns]
        except Exception as e:
            raise ValueError(f"Could not parse quantile column names in {fp}: {e}")
        
        # Sort columns
        Q = Q[sorted(Q.columns)]
        
        out[t] = Q
    
    return out


def save_quantiles(
    quantiles: Dict[int, pd.DataFrame],
    dir_path: str
) -> None:
    """
    Save quantile tables to disk.
    
    Args:
        quantiles: Dict mapping week offset to quantile DataFrames
        dir_path: Output directory
    """
    out_dir = Path(dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for t, Q in quantiles.items():
        fp = out_dir / f"week+{t}.parquet"
        Q.to_parquet(fp)


def quantiles_to_sip_samples(
    q_tables: Dict[int, pd.DataFrame],
    idx: pd.MultiIndex,
    n_sims: int,
    seed: int = 42
) -> np.ndarray:
    """
    Sample from quantile functions via inverse transform.
    
    Args:
        q_tables: Dict[week_offset -> DataFrame[index=SKUs, cols=quantiles]]
        idx: Target MultiIndex for alignment
        n_sims: Number of scenarios to sample
        seed: Random seed
        
    Returns:
        Array of shape [n_sims, horizon, n_items]
    """
    rng = np.random.default_rng(seed)
    horizon = len(q_tables)
    n_items = len(idx)
    
    # Generate uniform draws
    U = rng.random((n_sims, horizon, n_items))
    out = np.zeros_like(U)
    
    for t in range(1, horizon + 1):
        Q = q_tables[t].reindex(idx)
        
        if Q.isna().any().any():
            raise ValueError(f"Quantile table for week+{t} has NaNs after reindex")
        
        q_levels = np.array(sorted(Q.columns), dtype=float)
        Q_values = Q.values  # shape [n_items, n_quantiles]
        
        u = U[:, t - 1, :]  # shape [n_sims, n_items]
        
        # Vectorized inverse quantile via linear interpolation
        for i in range(n_items):
            out[:, t - 1, i] = np.interp(u[:, i], q_levels, Q_values[i, :])
    
    return out

