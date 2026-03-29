"""
Rule-based regime labels from entropy trajectories (Task 4 prototype).

Groups rows by (store, product, model_name), ordered by fold_idx.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def classify_series(h_outcome: np.ndarray, cv_thresh: float = 0.2, slope_thresh: float = 0.03) -> str:
    """
    Classify a short series of H_outcome values.

    - volatile: high coefficient of variation across folds
    - trending: |slope| over fold index large
    - stable: otherwise
    - unknown: fewer than 3 points
    """
    x = np.asarray(h_outcome, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return "unknown"
    m = np.mean(np.abs(x)) + 1e-8
    cv = float(np.std(x) / m)
    if cv >= cv_thresh:
        return "volatile"
    t = np.arange(len(x), dtype=float)
    slope = float(np.polyfit(t, x, 1)[0])
    if abs(slope) >= slope_thresh:
        return "trending"
    return "stable"


def add_entropy_regime_labels(
    df: pd.DataFrame,
    cv_thresh: float = 0.2,
    slope_thresh: float = 0.03,
) -> pd.DataFrame:
    """Add per-row `entropy_regime` from grouped H_outcome_w2 series."""
    need = {"store", "product", "model_name", "fold_idx", "H_outcome_w2"}
    if not need.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns {need}")

    regime = pd.Series(index=df.index, dtype=object)
    for _, g in df.groupby(["store", "product", "model_name"], sort=False):
        g = g.sort_values("fold_idx")
        lab = classify_series(g["H_outcome_w2"].values, cv_thresh, slope_thresh)
        regime.loc[g.index] = lab
    out = df.copy()
    out["entropy_regime"] = regime.astype(str)
    return out


def add_sensitivity_ratio_column(df: pd.DataFrame) -> pd.DataFrame:
    """Within each (store, product, model_name), compute ΔH_out / ΔH_dem across consecutive folds."""
    from vn2.analyze.entropy_metrics import sensitivity_ratio

    need = {"store", "product", "model_name", "fold_idx", "H_demand_h2", "H_outcome_w2"}
    if not need.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns {need}")

    sen = pd.Series(np.nan, index=df.index, dtype=float)
    for _, g in df.groupby(["store", "product", "model_name"], sort=False):
        g = g.sort_values("fold_idx")
        hd = g["H_demand_h2"].values
        ho = g["H_outcome_w2"].values
        r = np.full(len(g), np.nan)
        for i in range(1, len(g)):
            r[i] = sensitivity_ratio(hd[i - 1], hd[i], ho[i - 1], ho[i])
        sen.loc[g.index] = r
    out = df.copy()
    out["sensitivity_ratio"] = sen
    return out
