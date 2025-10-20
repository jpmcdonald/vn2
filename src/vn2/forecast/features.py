"""
Feature engineering for density forecasting.

Creates calendar, lag, rolling, trend, and hierarchy features for ML models.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from scipy import stats


def create_calendar_features(dates: pd.Series) -> pd.DataFrame:
    """
    Create calendar-based features.
    
    Args:
        dates: Series or DatetimeIndex of dates
        
    Returns:
        DataFrame with calendar features
    """
    # Convert to DatetimeIndex if needed
    if isinstance(dates, pd.Series):
        dt = pd.to_datetime(dates).dt
        idx = dates.index
    else:
        dt = dates
        idx = dates
    
    df = pd.DataFrame(index=idx)
    df['week_of_year'] = dt.isocalendar().week.values if hasattr(dt, 'isocalendar') else dt.week
    df['month'] = dt.month
    df['quarter'] = dt.quarter
    df['day_of_week'] = dt.dayofweek if hasattr(dt, 'dayofweek') else dt.day_of_week
    df['is_month_start'] = dt.is_month_start.astype(int)
    df['is_month_end'] = dt.is_month_end.astype(int)
    df['is_quarter_start'] = dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = dt.is_quarter_end.astype(int)
    
    # Cyclical encoding for seasonality
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df


def create_lag_features(
    y: pd.Series, 
    lags: List[int] = [1, 2, 4, 8, 13, 52]
) -> pd.DataFrame:
    """
    Create lag features.
    
    Args:
        y: Time series
        lags: List of lag periods
        
    Returns:
        DataFrame with lag features
    """
    df = pd.DataFrame(index=y.index)
    for lag in lags:
        df[f'lag_{lag}'] = y.shift(lag)
    return df


def create_rolling_features(
    y: pd.Series,
    windows: List[int] = [4, 8, 13, 26]
) -> pd.DataFrame:
    """
    Create rolling window statistics.
    
    Args:
        y: Time series
        windows: List of window sizes
        
    Returns:
        DataFrame with rolling features
    """
    df = pd.DataFrame(index=y.index)
    
    for window in windows:
        rolling = y.rolling(window, min_periods=1)
        df[f'rolling_mean_{window}'] = rolling.mean()
        df[f'rolling_std_{window}'] = rolling.std()
        df[f'rolling_cv_{window}'] = rolling.std() / (rolling.mean() + 1e-9)
        df[f'rolling_min_{window}'] = rolling.min()
        df[f'rolling_max_{window}'] = rolling.max()
        df[f'rolling_median_{window}'] = rolling.median()
    
    return df


def create_trend_features(
    y: pd.Series,
    window: int = 13
) -> pd.DataFrame:
    """
    Create trend features via linear regression slope.
    
    Args:
        y: Time series
        window: Window for trend estimation
        
    Returns:
        DataFrame with trend features
    """
    df = pd.DataFrame(index=y.index)
    
    def compute_slope(window_data):
        if len(window_data) < 2 or window_data.isna().all():
            return np.nan
        x = np.arange(len(window_data))
        y_vals = window_data.values
        # Remove nans
        mask = ~np.isnan(y_vals)
        if mask.sum() < 2:
            return np.nan
        try:
            slope, _ = np.polyfit(x[mask], y_vals[mask], 1)
            return slope
        except:
            return np.nan
    
    df[f'trend_slope_{window}'] = y.rolling(window, min_periods=2).apply(
        compute_slope, raw=False
    )
    
    return df


def create_intermittency_features(
    y: pd.Series,
    windows: List[int] = [4, 8, 13]
) -> pd.DataFrame:
    """
    Create intermittency-specific features.
    
    Args:
        y: Time series
        windows: Windows for zero rate calculation
        
    Returns:
        DataFrame with intermittency features
    """
    df = pd.DataFrame(index=y.index)
    
    # Zero indicator
    is_zero = (y == 0).astype(int)
    
    for window in windows:
        # Zero rate (proportion of zeros in window)
        df[f'zero_rate_{window}'] = is_zero.rolling(
            window, min_periods=1
        ).mean()
    
    # Weeks since last nonzero
    nonzero_indices = np.where(y.values > 0)[0]
    weeks_since = np.zeros(len(y))
    for i in range(len(y)):
        prior_nonzeros = nonzero_indices[nonzero_indices < i]
        if len(prior_nonzeros) > 0:
            weeks_since[i] = i - prior_nonzeros[-1]
        else:
            weeks_since[i] = i  # No prior nonzero
    df['weeks_since_nonzero'] = weeks_since
    
    # Average demand interval (ADI)
    if len(nonzero_indices) > 1:
        intervals = np.diff(nonzero_indices)
        df['adi'] = intervals.mean() if len(intervals) > 0 else len(y)
    else:
        df['adi'] = len(y)
    
    return df


def create_seasonality_features(
    dates: pd.Series,
    n_fourier: int = 3
) -> pd.DataFrame:
    """
    Create Fourier seasonal features for annual cycle.
    
    Args:
        dates: Series or DatetimeIndex of dates
        n_fourier: Number of Fourier pairs (sin/cos)
        
    Returns:
        DataFrame with Fourier features
    """
    # Convert to datetime accessor if needed
    if isinstance(dates, pd.Series):
        dt = pd.to_datetime(dates).dt
        idx = dates.index
    else:
        dt = dates
        idx = dates
    
    df = pd.DataFrame(index=idx)
    
    # Day of year for annual cycle
    day_of_year = dt.dayofyear if hasattr(dt, 'dayofyear') else dt.day_of_year
    
    for k in range(1, n_fourier + 1):
        df[f'fourier_sin_{k}'] = np.sin(2 * np.pi * k * day_of_year / 365.25)
        df[f'fourier_cos_{k}'] = np.cos(2 * np.pi * k * day_of_year / 365.25)
    
    return df


def create_features(
    df: pd.DataFrame,
    sku_id: Tuple[int, int],
    master_df: Optional[pd.DataFrame] = None,
    lookback: int = 52,
    for_prediction: bool = False
) -> pd.DataFrame:
    """
    Create full feature set for a single SKU.
    
    Args:
        df: Full dataframe with demand history
        sku_id: (Store, Product) tuple
        master_df: Master data for hierarchy features
        lookback: Minimum history required
        for_prediction: If True, generate features for future prediction
        
    Returns:
        DataFrame with all features
    """
    # Filter to SKU
    sku_df = df[
        (df['Store'] == sku_id[0]) & (df['Product'] == sku_id[1])
    ].sort_values('week_date').copy()
    
    if len(sku_df) < lookback:
        # Insufficient history
        return pd.DataFrame()
    
    y = sku_df['sales']
    dates = pd.to_datetime(sku_df['week_date'])
    
    # Create feature dataframes
    features = []
    
    # Calendar
    features.append(create_calendar_features(dates))
    
    # Lags
    features.append(create_lag_features(y))
    
    # Rolling stats
    features.append(create_rolling_features(y))
    
    # Trend
    features.append(create_trend_features(y))
    
    # Intermittency
    features.append(create_intermittency_features(y))
    
    # Seasonality (Fourier)
    features.append(create_seasonality_features(dates))
    
    # Combine all
    X = pd.concat(features, axis=1)
    X.index = sku_df.index
    
    # Add in_stock indicator if available (for stockout-aware models)
    if 'in_stock' in sku_df.columns:
        X['in_stock'] = sku_df['in_stock'].values
    
    # Add hierarchy features if available
    if master_df is not None:
        hierarchy = master_df[
            (master_df['Store'] == sku_id[0]) & 
            (master_df['Product'] == sku_id[1])
        ]
        if len(hierarchy) > 0:
            for col in ['ProductGroup', 'Department', 'StoreFormat']:
                if col in hierarchy.columns:
                    X[col] = hierarchy[col].iloc[0]
    
    return X


def prepare_train_test_split(
    df: pd.DataFrame,
    sku_id: Tuple[int, int],
    holdout_weeks: int = 12,
    fold_idx: int = 0,
    master_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Prepare rolling-origin train/test split for a single SKU.
    
    Args:
        df: Full dataframe
        sku_id: (Store, Product)
        holdout_weeks: Total weeks held out
        fold_idx: Which fold (0 = most recent, 11 = earliest)
        master_df: Master data
        
    Returns:
        (y_train, X_train, y_test, X_test)
    """
    sku_df = df[
        (df['Store'] == sku_id[0]) & (df['Product'] == sku_id[1])
    ].sort_values('week_date')
    
    # Split point: holdout_weeks - fold_idx from end
    split_idx = len(sku_df) - (holdout_weeks - fold_idx)
    
    if split_idx < 52:  # Need at least 1 year history
        return None, None, None, None
    
    train_df = sku_df.iloc[:split_idx]
    test_df = sku_df.iloc[split_idx:split_idx + 2]  # h=1,2
    
    # Create features
    all_df = pd.concat([train_df, test_df])
    X_all = create_features(
        pd.concat([df, all_df]), 
        sku_id, 
        master_df
    )
    
    if X_all.empty:
        return None, None, None, None
    
    y_train = train_df['sales']
    y_test = test_df['sales']
    
    X_train = X_all.loc[train_df.index]
    X_test = X_all.loc[test_df.index]
    
    return y_train, X_train, y_test, X_test

