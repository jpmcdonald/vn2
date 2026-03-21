"""
Feature engineering reproducing Szabłowski (arXiv:2601.18919v1) Section 6.2.

Stockout-aware effective sales, dynamic per-series scaling, and the full
feature set: demand level, trend/momentum, seasonality, intermittency/spikes.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Stockout-aware effective sales
# ---------------------------------------------------------------------------

def compute_effective_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Mask out-of-stock weeks to NaN, creating the effective-sales series y_eff.

    Expects columns: Store, Product, week_date, sales, in_stock.
    Adds column 'y_eff' (sales where in_stock else NaN).
    """
    out = df.copy()
    sales_col = "sales" if "sales" in out.columns else "demand"
    out["y_eff"] = out[sales_col].where(out["in_stock"].astype(bool))
    return out


# ---------------------------------------------------------------------------
# Dynamic per-series scaling
# ---------------------------------------------------------------------------

def compute_scale_factors(
    df: pd.DataFrame,
    min_instock_obs: int = 45,
) -> pd.DataFrame:
    """Compute annualised scale factor per (Store, Product, week_date).

    scale(i,t) = max(53 * mean_nonmissing(y_eff[t-52:t]), 1)
    For series with fewer than `min_instock_obs` in-stock observations in the
    trailing 52-week window, fall back to an expanding mean (annualised).

    Adds column 'scale_factor'.
    """
    wc = _week_col(df)
    df = df.sort_values(["Store", "Product", wc])
    scales = []

    for (store, product), grp in df.groupby(["Store", "Product"]):
        y_eff = grp["y_eff"].values
        n = len(y_eff)
        sf = np.ones(n, dtype=np.float64)

        for t in range(n):
            lo = max(0, t - 52)
            window = y_eff[lo:t]
            valid = window[~np.isnan(window)]

            if len(valid) >= min_instock_obs:
                sf[t] = max(53.0 * np.mean(valid), 1.0)
            else:
                # Warm-start: expanding mean, annualised
                all_valid = y_eff[:t][~np.isnan(y_eff[:t])]
                if len(all_valid) > 0:
                    sf[t] = max(53.0 * np.mean(all_valid), 1.0)
                else:
                    sf[t] = 1.0

        scales.append(
            pd.DataFrame(
                {"Store": store, "Product": product, wc: grp[wc].values, "scale_factor": sf}
            )
        )

    return pd.concat(scales, ignore_index=True)


# ---------------------------------------------------------------------------
# Lag features  (on scaled y_eff)
# ---------------------------------------------------------------------------

def _lag_features(y: np.ndarray) -> dict:
    """Short-term lags, seasonal lags."""
    n = len(y)
    feats: dict = {}
    for lag in [0, 1, 2, 3]:
        col = f"lag_{lag}"
        feats[col] = np.full(n, np.nan)
        if lag < n:
            feats[col][lag:] = y[: n - lag]
    for lag in [51, 52, 53]:
        col = f"lag_{lag}"
        feats[col] = np.full(n, np.nan)
        if lag < n:
            feats[col][lag:] = y[: n - lag]
    return feats


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def _rolling_features(y: pd.Series) -> dict:
    """Rolling means, medians, EWM, std, IQR on y_eff (scaled)."""
    feats: dict = {}
    for w in [3, 5, 13]:
        r = y.rolling(w, min_periods=1)
        feats[f"rmean_{w}"] = r.mean().values
        feats[f"rmedian_{w}"] = r.median().values

    for span in [5, 10]:
        feats[f"ewm_{span}"] = y.ewm(span=span, min_periods=1).mean().values

    r13 = y.rolling(13, min_periods=2)
    feats["rstd_13"] = r13.std().values
    feats["riqr_13"] = (
        y.rolling(13, min_periods=2).quantile(0.75).values
        - y.rolling(13, min_periods=2).quantile(0.25).values
    )
    return feats


# ---------------------------------------------------------------------------
# Trend / momentum
# ---------------------------------------------------------------------------

def _trend_features(y: pd.Series) -> dict:
    feats: dict = {}
    feats["delta_1"] = y.diff(1).values
    feats["delta_5"] = y.diff(5).values
    slope = y.rolling(4, min_periods=2).apply(
        lambda w: np.polyfit(np.arange(len(w)), w, 1)[0] if len(w) >= 2 else np.nan,
        raw=True,
    )
    feats["slope_4w"] = slope.values
    return feats


# ---------------------------------------------------------------------------
# Seasonality features
# ---------------------------------------------------------------------------

def _seasonality_features(
    week_of_year: np.ndarray,
    y_eff: np.ndarray,
    n_harmonics: int = 3,
) -> dict:
    """Fourier terms, last-year-window average, seasonality-strength proxy."""
    n = len(week_of_year)
    feats: dict = {}

    feats["week_of_year"] = week_of_year.astype(np.int32)

    for k in range(1, n_harmonics + 1):
        feats[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * week_of_year / 52.0)
        feats[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * week_of_year / 52.0)

    # Last-year-window average (mean of lags 49–55 where available)
    ly_avg = np.full(n, np.nan)
    for t in range(55, n):
        window = y_eff[t - 55 : t - 48]  # lags 49-55
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            ly_avg[t] = np.mean(valid)
    feats["last_year_window_avg"] = ly_avg

    # Seasonality-strength proxy: correlation between current y_eff and seasonal lag
    ss = np.full(n, np.nan)
    for t in range(52, n):
        current = y_eff[max(0, t - 13) : t]
        lagged = y_eff[max(0, t - 13 - 52) : t - 52]
        min_len = min(len(current), len(lagged))
        if min_len < 3:
            continue
        c = current[:min_len]
        l = lagged[:min_len]
        mask = ~(np.isnan(c) | np.isnan(l))
        if mask.sum() >= 3:
            cc = np.corrcoef(c[mask], l[mask])
            if cc.shape == (2, 2):
                ss[t] = cc[0, 1]
    feats["seasonality_strength"] = ss

    return feats


# ---------------------------------------------------------------------------
# Intermittency / spike features
# ---------------------------------------------------------------------------

def _spike_features(y: pd.Series) -> dict:
    """Robust z-score, time_since_spike, rolling non-zero rate."""
    n = len(y)
    feats: dict = {}

    rmedian = y.rolling(13, min_periods=2).median()
    rmad = y.rolling(13, min_periods=2).apply(
        lambda w: np.median(np.abs(w - np.median(w))), raw=True
    )
    safe_mad = rmad.replace(0, np.nan)
    feats["robust_zscore"] = ((y - rmedian) / (1.4826 * safe_mad)).values

    # time_since_spike: weeks since last |robust_z| > 3
    rz = feats["robust_zscore"]
    tss = np.full(n, np.nan)
    last_spike = -999
    for t in range(n):
        if not np.isnan(rz[t]) and abs(rz[t]) > 3.0:
            last_spike = t
        if last_spike >= 0:
            tss[t] = t - last_spike
    feats["time_since_spike"] = tss

    # Rolling non-zero rate (12-week window)
    nonzero = (~y.isna() & (y > 0)).astype(float)
    feats["rolling_nonzero_rate_12"] = nonzero.rolling(12, min_periods=1).mean().values

    return feats


# ---------------------------------------------------------------------------
# Two-level median imputation
# ---------------------------------------------------------------------------

def two_level_median_imputation(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Per-series median fill, then global median fill for remaining NaNs."""
    out = df.copy()
    for col in feature_cols:
        if col not in out.columns:
            continue
        series_medians = out.groupby(["Store", "Product"])[col].transform("median")
        out[col] = out[col].fillna(series_medians)
        global_median = out[col].median()
        out[col] = out[col].fillna(global_median if not np.isnan(global_median) else 0.0)
    return out


# ---------------------------------------------------------------------------
# Time-decayed observation weights
# ---------------------------------------------------------------------------

def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Stepwise yearly decay: 1.0 for most recent ~53 weeks, 0.5 prior, 0.25 before that.

    Expects df already sorted by time within each SKU. Returns array aligned to df index.
    """
    wc = _week_col(df)
    weights = np.ones(len(df), dtype=np.float64)

    for _, grp in df.groupby(["Store", "Product"]):
        idx = grp.index
        n = len(idx)
        if n <= 53:
            weights[idx] = 1.0
        elif n <= 106:
            weights[idx[:n - 53]] = 0.5
            weights[idx[n - 53:]] = 1.0
        else:
            weights[idx[:n - 106]] = 0.25
            weights[idx[n - 106:n - 53]] = 0.5
            weights[idx[n - 53:]] = 1.0

    return weights


# ---------------------------------------------------------------------------
# Master feature builder
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    n_harmonics: int = 3,
) -> Tuple[pd.DataFrame, list]:
    """End-to-end feature pipeline for the Szabłowski CatBoost reproduction.

    Parameters
    ----------
    df : DataFrame
        Must contain Store, Product, week_date (or week), sales, in_stock.
    n_harmonics : int
        Number of Fourier harmonics (default 3).

    Returns
    -------
    featured_df : DataFrame
        Original columns plus all engineered features, scale_factor, y_scaled,
        sample_weight.  NaNs filled via two-level median imputation.
    feature_cols : list[str]
        Names of the feature columns (input to CatBoost).
    """
    wc = _week_col(df)
    df = df.sort_values(["Store", "Product", wc]).reset_index(drop=True)

    # Step 1: effective sales
    df = compute_effective_sales(df)

    # Step 2: scale factors
    sf = compute_scale_factors(df)
    df = df.merge(sf, on=["Store", "Product", wc], how="left")
    df["y_scaled"] = df["y_eff"] / df["scale_factor"]

    # Step 3: build features per SKU on scaled y_eff
    all_feat_dfs = []
    feature_names: set = set()

    for (store, product), grp in df.groupby(["Store", "Product"]):
        y_scaled = grp["y_scaled"].values
        y_series = pd.Series(y_scaled, index=grp.index)
        woy = grp[wc].dt.isocalendar().week.values.astype(float) if hasattr(grp[wc].dt, "isocalendar") else grp["week_of_year"].values if "week_of_year" in grp.columns else np.zeros(len(grp))

        # Try to get week_of_year from the date column
        if hasattr(grp[wc], "dt"):
            try:
                woy = grp[wc].dt.isocalendar().week.values.astype(float)
            except Exception:
                woy = (grp[wc].dt.day_of_year // 7 + 1).values.astype(float)

        feats: dict = {}
        feats.update(_lag_features(y_scaled))
        feats.update(_rolling_features(y_series))
        feats.update(_trend_features(y_series))
        feats.update(_seasonality_features(woy, grp["y_eff"].values / grp["scale_factor"].values, n_harmonics))
        feats.update(_spike_features(y_series))

        feat_df = pd.DataFrame(feats, index=grp.index)
        feature_names.update(feat_df.columns)
        all_feat_dfs.append(feat_df)

    features = pd.concat(all_feat_dfs)
    df = pd.concat([df, features], axis=1)

    # Categorical features
    df["store_id"] = df["Store"].astype("category")
    df["product_id"] = df["Product"].astype("category")

    feature_cols = sorted(feature_names) + ["store_id", "product_id"]

    # Step 4: two-level median imputation (numeric features only)
    numeric_feats = [c for c in feature_cols if c not in ("store_id", "product_id")]
    df = two_level_median_imputation(df, numeric_feats)

    # Step 5: sample weights
    df["sample_weight"] = compute_sample_weights(df)

    return df, feature_cols


# ---------------------------------------------------------------------------
# Train/val/test split helpers
# ---------------------------------------------------------------------------

def train_val_test_split(
    df: pd.DataFrame,
    test_weeks: int = 18,
    val_frac: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split matching the paper.

    test  = last `test_weeks` weeks of data
    val   = most recent `val_frac` of remaining weeks
    train = everything else
    """
    wc = _week_col(df)
    weeks_sorted = np.sort(df[wc].unique())
    n_weeks = len(weeks_sorted)

    test_cutoff = weeks_sorted[max(0, n_weeks - test_weeks)]
    remaining_weeks = weeks_sorted[weeks_sorted < test_cutoff]
    n_remain = len(remaining_weeks)
    val_start_idx = max(0, int(n_remain * (1 - val_frac)))
    val_cutoff = remaining_weeks[val_start_idx]

    train = df[df[wc] < val_cutoff]
    val = df[(df[wc] >= val_cutoff) & (df[wc] < test_cutoff)]
    test = df[df[wc] >= test_cutoff]
    return train, val, test


def fold_split(
    df: pd.DataFrame,
    fold_idx: int = 0,
    holdout_weeks: int = 18,
    n_test_weeks: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling-origin fold split for backtest evaluation.

    fold_idx=0 is the most recent fold.  Returns (train, test) where test
    contains n_test_weeks of data (h=1, h=2, h=3).
    """
    wc = _week_col(df)
    weeks_sorted = np.sort(df[wc].unique())
    n = len(weeks_sorted)

    test_end_idx = n - fold_idx
    test_start_idx = max(0, test_end_idx - n_test_weeks)
    train_end_idx = test_start_idx

    if train_end_idx < 52:
        return pd.DataFrame(), pd.DataFrame()

    test_weeks = weeks_sorted[test_start_idx:test_end_idx]
    train = df[df[wc] < weeks_sorted[train_end_idx]]
    test = df[df[wc].isin(test_weeks)]
    return train, test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _week_col(df: pd.DataFrame) -> str:
    if "week_date" in df.columns:
        return "week_date"
    if "week" in df.columns:
        return "week"
    raise KeyError("DataFrame must contain 'week_date' or 'week' column")
