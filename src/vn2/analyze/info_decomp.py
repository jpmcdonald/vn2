"""
Partial information decomposition (PID) utilities for exploratory analysis.

We implement a minimal Imin-based PID for two sources X1, X2 and target Y:
  - Unique(X1;Y) = I(X1;Y) - Redundancy
  - Unique(X2;Y) = I(X2;Y) - Redundancy
  - Redundancy = min{ I(X1;Y), I(X2;Y) }
  - Synergy = I(X1,X2;Y) - Unique(X1;Y) - Unique(X2;Y) - Redundancy

This is not BROJA/partial information decomposition, but a practical
lower-bound heuristic suitable for quick EDA without heavy dependencies.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from pyitlib import discrete_random_variable as drv


def _discretize(series: pd.Series, num_bins: int = 8) -> np.ndarray:
    values = series.values
    if np.issubdtype(values.dtype, np.number):
        bins = np.nanpercentile(values[~np.isnan(values)], np.linspace(0, 100, num_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 3:
            # Fallback: rank transform
            ranks = pd.Series(values).rank(method="average").values
            return np.nan_to_num(np.digitize(ranks, np.linspace(0, ranks.max() + 1, num_bins + 1)) - 1, nan=0).astype(int)
        return np.nan_to_num(np.digitize(values, bins) - 1, nan=0).astype(int)
    return pd.Series(values).ffill().bfill().astype(int).values


def pid_imin(x1: pd.Series, x2: pd.Series, y: pd.Series, num_bins: int = 8) -> Dict[str, float]:
    X1 = _discretize(x1, num_bins)
    X2 = _discretize(x2, num_bins)
    Y = _discretize(y, num_bins)

    # Mutual informations in bits
    i_x1_y = float(drv.information_mutual(X1, Y, estimator="ML"))
    i_x2_y = float(drv.information_mutual(X2, Y, estimator="ML"))
    # Compute MI for joint (X1,X2) with Y by hashing pairs to a single discrete variable
    base = int(max(X1.max(), X2.max()) + 1)
    X12_joint = (X1.astype(int) * base + X2.astype(int)).astype(int)
    i_x1x2_y = float(drv.information_mutual(X12_joint, Y, estimator="ML"))

    redundancy = min(i_x1_y, i_x2_y)
    unique1 = max(0.0, i_x1_y - redundancy)
    unique2 = max(0.0, i_x2_y - redundancy)
    synergy = max(0.0, i_x1x2_y - unique1 - unique2 - redundancy)

    return {
        "mi_x1_y": i_x1_y,
        "mi_x2_y": i_x2_y,
        "mi_x1x2_y": i_x1x2_y,
        "redundancy": redundancy,
        "unique_x1": unique1,
        "unique_x2": unique2,
        "synergy": synergy,
    }


def run_pid_for_sku(df: pd.DataFrame, store: int, product: int, target_col: str = "sales") -> Dict[str, float]:
    sku = df[(df["Store"] == store) & (df["Product"] == product)].sort_values("week")
    if len(sku) < 16:
        return {}
    # Choose two sources: lagged sales and in_stock
    y = sku[target_col]
    x1 = y.shift(1).fillna(0)
    x2 = sku.get("in_stock", pd.Series(np.ones(len(sku)), index=sku.index))
    return pid_imin(x1, x2, y)


def sample_pid(df: pd.DataFrame, n: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    skus = df[["Store", "Product"]].drop_duplicates().values
    if len(skus) == 0:
        return pd.DataFrame()
    idx = rng.choice(len(skus), size=min(n, len(skus)), replace=False)
    rows = []
    for i in idx:
        s, p = skus[i]
        metrics = run_pid_for_sku(df, int(s), int(p))
        if metrics:
            metrics.update({"Store": int(s), "Product": int(p)})
            rows.append(metrics)
    return pd.DataFrame(rows)


def compute_pid_feature_weights(df: pd.DataFrame, n: int = None, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produce per-SKU feature weights from Imin PID for a fixed feature set.
    Returns (weights_df, k_df) where:
      - weights_df: columns [Store, Product, feature, weight]
      - k_df: columns [Store, Product, k] recommended neighbor count
    """
    rng = np.random.default_rng(seed)
    skus = df[["Store", "Product"]].drop_duplicates().values
    if n is not None and n < len(skus):
        idx = rng.choice(len(skus), size=n, replace=False)
        skus = skus[idx]
    weight_rows = []
    k_rows = []
    for s, p in skus:
        sku = df[(df['Store']==s) & (df['Product']==p)].sort_values('week')
        if len(sku) < 16:
            continue
        y = sku['sales']
        # Feature set
        feats = {
            'lag_1': y.shift(1).fillna(0),
            'lag_2': y.shift(2).fillna(0),
            'month': pd.to_datetime(sku['week']).dt.month,
            'in_stock': sku.get('in_stock', pd.Series(np.ones(len(sku)), index=sku.index))
        }
        # Pairwise MI to score features (simple proxy: unique ≈ MI - min(MI, other MI))
        mi = {f: float(drv.information_mutual(_discretize(v), _discretize(y))) for f, v in feats.items()}
        # Normalize to [0.5, 2.0]
        if len(mi) == 0:
            continue
        mi_vals = np.array(list(mi.values()))
        if np.all(mi_vals == 0):
            scale = {k: 1.0 for k in mi}
        else:
            mi_norm = (mi_vals - mi_vals.min()) / (np.ptp(mi_vals) + 1e-9)
            scale = {k: float(0.5 + 1.5 * mi_norm[i]) for i, k in enumerate(mi.keys())}
        for f, w in scale.items():
            weight_rows.append({'Store': int(s), 'Product': int(p), 'feature': f, 'weight': w})
        # k recommendation: higher total MI -> smaller k
        total_mi = mi_vals.sum()
        k = int(np.clip(100 - 60 * total_mi, 10, 120))
        k_rows.append({'Store': int(s), 'Product': int(p), 'k': k})
    weights_df = pd.DataFrame(weight_rows)
    k_df = pd.DataFrame(k_rows)
    return weights_df, k_df


