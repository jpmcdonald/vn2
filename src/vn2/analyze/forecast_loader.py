"""
Forecast loader for sequential backtest.

Loads forecast checkpoints and converts quantiles to PMFs.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Optional, Tuple
from .sequential_backtest import quantiles_to_pmf


def load_forecasts_for_sku(
    store: int,
    product: int,
    model_name: str,
    checkpoints_dir: Path,
    n_folds: int = 12,
    quantile_levels: Optional[np.ndarray] = None,
    pmf_grain: int = 500
) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
    """
    Load h1 and h2 PMFs for a SKU from forecast checkpoints.
    
    Args:
        store: Store ID
        product: Product ID
        model_name: Model name (e.g., 'zinb', 'qrf', 'naive4')
        checkpoints_dir: Path to checkpoints directory
        n_folds: Number of folds (weeks) to load
        quantile_levels: Quantile levels (default: [0.01, 0.05, ..., 0.99])
        pmf_grain: PMF support size
    
    Returns:
        (h1_pmfs, h2_pmfs): Lists of PMFs (or None if missing)
    """
    if quantile_levels is None:
        quantile_levels = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    
    model_dir = checkpoints_dir / model_name
    sku_dir = model_dir / f"{store}_{product}"
    
    if not sku_dir.exists():
        # SKU not found: return all None
        return [None] * n_folds, [None] * n_folds
    
    h1_pmfs = []
    h2_pmfs = []
    
    for fold_idx in range(n_folds):
        fold_file = sku_dir / f"fold_{fold_idx}.pkl"
        
        if not fold_file.exists():
            h1_pmfs.append(None)
            h2_pmfs.append(None)
            continue
        
        try:
            with open(fold_file, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # Extract quantiles DataFrame
            quantiles_df = checkpoint.get('quantiles')
            
            if quantiles_df is None or len(quantiles_df) < 2:
                h1_pmfs.append(None)
                h2_pmfs.append(None)
                continue
            
            # Convert step 1 (h1) and step 2 (h2) to PMFs
            q1 = quantiles_df.loc[1].values if 1 in quantiles_df.index else quantiles_df.iloc[0].values
            q2 = quantiles_df.loc[2].values if 2 in quantiles_df.index else quantiles_df.iloc[1].values
            
            pmf1 = quantiles_to_pmf(q1, quantile_levels, pmf_grain)
            pmf2 = quantiles_to_pmf(q2, quantile_levels, pmf_grain)
            
            h1_pmfs.append(pmf1)
            h2_pmfs.append(pmf2)
            
        except Exception as e:
            print(f"Warning: Failed to load {fold_file}: {e}")
            h1_pmfs.append(None)
            h2_pmfs.append(None)
    
    return h1_pmfs, h2_pmfs


def get_available_models(checkpoints_dir: Path) -> List[str]:
    """
    Get list of available model names in checkpoints directory.
    
    Args:
        checkpoints_dir: Path to checkpoints directory
    
    Returns:
        List of model names
    """
    if not checkpoints_dir.exists():
        return []
    
    models = []
    for item in checkpoints_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.') and item.name != '__pycache__':
            # Check if it has any SKU subdirectories
            sku_dirs = [d for d in item.iterdir() if d.is_dir() and '_' in d.name]
            if sku_dirs:
                models.append(item.name)
    
    return sorted(models)


def get_available_skus(checkpoints_dir: Path, model_name: str) -> List[Tuple[int, int]]:
    """
    Get list of available SKUs for a model.
    
    Args:
        checkpoints_dir: Path to checkpoints directory
        model_name: Model name
    
    Returns:
        List of (store, product) tuples
    """
    model_dir = checkpoints_dir / model_name
    if not model_dir.exists():
        return []
    
    skus = []
    for sku_dir in model_dir.iterdir():
        if sku_dir.is_dir() and '_' in sku_dir.name:
            try:
                store, product = sku_dir.name.split('_')
                skus.append((int(store), int(product)))
            except ValueError:
                continue
    
    return sorted(skus)

