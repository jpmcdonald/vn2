"""
Feature engineering revamp: integrate Szabłowski features into the existing
VN2 forecast pipeline.

Provides drop-in feature functions compatible with src/vn2/forecast/features.py
and a monkey-patch function to augment create_features with the new features.

New features from Szabłowski (arXiv:2601.18919v1):
  - Stockout-aware effective sales masking
  - Seasonality-strength proxy (correlation with seasonal lags)
  - Robust z-score spike detection + time_since_spike
  - Rolling non-zero rate (12-week)
  - 4-week rolling slope proxy
  - Dynamic per-series annualized scaling (optional preprocessing)
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Stockout-aware masking
# ---------------------------------------------------------------------------

def mask_stockout_weeks(y: pd.Series, in_stock: pd.Series) -> pd.Series:
    """Replace sales during stockout weeks with NaN.

    This is applied BEFORE computing any features so that out-of-stock
    periods don't bias lag/rolling computations downward.
    """
    return y.where(in_stock.astype(bool))


# ---------------------------------------------------------------------------
# Seasonality-strength proxy
# ---------------------------------------------------------------------------

def create_seasonality_strength(
    y: pd.Series,
    window: int = 13,
    seasonal_lag: int = 52,
) -> pd.DataFrame:
    """Correlation between recent y and its seasonal lag.

    Lets the model learn to weight seasonal features differently for series
    with strong vs weak seasonality.
    """
    df = pd.DataFrame(index=y.index)
    n = len(y)
    vals = y.values
    ss = np.full(n, np.nan)

    for t in range(seasonal_lag, n):
        current = vals[max(0, t - window):t]
        lagged = vals[max(0, t - window - seasonal_lag):t - seasonal_lag]
        min_len = min(len(current), len(lagged))
        if min_len < 3:
            continue
        c = current[:min_len]
        l = lagged[:min_len]
        mask = ~(np.isnan(c) | np.isnan(l))
        if mask.sum() >= 3:
            cc = np.corrcoef(c[mask], l[mask])
            if cc.shape == (2, 2) and not np.isnan(cc[0, 1]):
                ss[t] = cc[0, 1]

    df["seasonality_strength"] = ss
    return df


# ---------------------------------------------------------------------------
# Spike detection (robust z-score + time_since_spike)
# ---------------------------------------------------------------------------

def create_spike_features(
    y: pd.Series,
    rolling_window: int = 13,
    spike_threshold: float = 3.0,
) -> pd.DataFrame:
    """Robust z-score and time since last spike.

    Uses rolling median and MAD for outlier-resistant spike detection.
    """
    df = pd.DataFrame(index=y.index)
    n = len(y)

    rmedian = y.rolling(rolling_window, min_periods=2).median()
    rmad = y.rolling(rolling_window, min_periods=2).apply(
        lambda w: np.median(np.abs(w - np.median(w))), raw=True
    )
    safe_mad = rmad.replace(0, np.nan)
    robust_z = ((y - rmedian) / (1.4826 * safe_mad)).values
    df["robust_zscore"] = robust_z

    tss = np.full(n, np.nan)
    last_spike = -999
    for t in range(n):
        if not np.isnan(robust_z[t]) and abs(robust_z[t]) > spike_threshold:
            last_spike = t
        if last_spike >= 0:
            tss[t] = t - last_spike
    df["time_since_spike"] = tss

    return df


# ---------------------------------------------------------------------------
# Rolling non-zero rate
# ---------------------------------------------------------------------------

def create_rolling_nonzero_rate(
    y: pd.Series,
    window: int = 12,
) -> pd.DataFrame:
    """Proportion of non-zero and non-NaN values in rolling window."""
    df = pd.DataFrame(index=y.index)
    nonzero = (~y.isna() & (y > 0)).astype(float)
    df["rolling_nonzero_rate_12"] = nonzero.rolling(window, min_periods=1).mean().values
    return df


# ---------------------------------------------------------------------------
# 4-week rolling slope proxy
# ---------------------------------------------------------------------------

def create_slope_proxy(y: pd.Series, window: int = 4) -> pd.DataFrame:
    """Linear regression slope over a short rolling window."""
    df = pd.DataFrame(index=y.index)
    slope = y.rolling(window, min_periods=2).apply(
        lambda w: np.polyfit(np.arange(len(w)), w, 1)[0] if len(w) >= 2 else np.nan,
        raw=True,
    )
    df["slope_4w"] = slope.values
    return df


# ---------------------------------------------------------------------------
# Annualized scaling (standalone)
# ---------------------------------------------------------------------------

def compute_annualized_scale(
    y: pd.Series,
    in_stock: Optional[pd.Series] = None,
    lookback: int = 52,
) -> float:
    """Compute Szabłowski's annualized scale factor for a single SKU.

    scale = max(53 * mean_nonmissing(y_eff[t-52:t]), 1)
    Returns a single scalar (computed at the last available time step).
    """
    if in_stock is not None:
        y_eff = y.where(in_stock.astype(bool))
    else:
        y_eff = y

    valid = y_eff.dropna().values[-lookback:]
    if len(valid) == 0:
        return 1.0
    return max(53.0 * np.mean(valid), 1.0)


# ---------------------------------------------------------------------------
# Combined feature augmentation
# ---------------------------------------------------------------------------

def create_szablowski_features(
    y: pd.Series,
    in_stock: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Create all Szabłowski-derived features for a single SKU.

    Designed to be called alongside existing create_features() and
    concatenated into the feature matrix.

    Parameters
    ----------
    y : demand series
    in_stock : optional in-stock flag series (same index as y)

    Returns
    -------
    DataFrame with columns: seasonality_strength, robust_zscore,
    time_since_spike, rolling_nonzero_rate_12, slope_4w
    """
    if in_stock is not None:
        y_eff = mask_stockout_weeks(y, in_stock)
    else:
        y_eff = y

    parts = [
        create_seasonality_strength(y_eff),
        create_spike_features(y_eff),
        create_rolling_nonzero_rate(y_eff),
        create_slope_proxy(y_eff),
    ]
    return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
# Integration helper for the existing pipeline
# ---------------------------------------------------------------------------

def augment_create_features(original_create_features):
    """Decorator/wrapper that adds Szabłowski features to the existing pipeline.

    Usage::

        from vn2.forecast import features
        from szablowski.feature_revamp import augment_create_features
        features.create_features = augment_create_features(features.create_features)

    After patching, all downstream code that calls create_features() will
    automatically get the Szabłowski features appended.
    """
    def augmented(df, sku_id, master_df=None, lookback=52, for_prediction=False):
        X = original_create_features(df, sku_id, master_df, lookback, for_prediction)
        if X.empty:
            return X

        _dc = "week_date" if "week_date" in df.columns else "week"
        _sc = "sales" if "sales" in df.columns else "demand"
        sku_df = df[
            (df["Store"] == sku_id[0]) & (df["Product"] == sku_id[1])
        ].sort_values(_dc)

        y = sku_df[_sc]
        in_stock = sku_df["in_stock"] if "in_stock" in sku_df.columns else None
        szab_feats = create_szablowski_features(y, in_stock)
        szab_feats.index = sku_df.index

        # Align to X's index
        common = X.index.intersection(szab_feats.index)
        for col in szab_feats.columns:
            X.loc[common, col] = szab_feats.loc[common, col].values

        return X

    return augmented
