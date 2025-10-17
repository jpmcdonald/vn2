"""
Full SIP replacement for stockout-censored demand using profile matching.

This module implements density function reconstruction by comparing demand
prediction intervals of similar non-stockout profiles with stockout conditions,
modeling constrained sales as Min(Demand, Stock).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Callable, Optional
from dataclasses import dataclass
from scipy.spatial.distance import mahalanobis
from scipy.stats import pearsonr
import warnings
from joblib import Parallel, delayed
import multiprocessing


@dataclass
class TransformPair:
    """Forward and inverse transform functions"""
    forward: Callable[[np.ndarray], np.ndarray]
    inverse: Callable[[np.ndarray], np.ndarray]
    name: str


# Transform library
TRANSFORMS: Dict[str, TransformPair] = {
    'identity': TransformPair(
        forward=lambda x: x,
        inverse=lambda x: x,
        name='identity'
    ),
    'sqrt': TransformPair(
        forward=lambda x: np.sqrt(np.maximum(x, 0)),
        inverse=lambda x: np.square(x),
        name='sqrt'
    ),
    'cbrt': TransformPair(
        forward=lambda x: np.cbrt(x),
        inverse=lambda x: np.power(x, 3),
        name='cbrt'
    ),
    'log1p': TransformPair(
        forward=lambda x: np.log1p(np.maximum(x, 0)),
        inverse=lambda x: np.expm1(x),
        name='log1p'
    ),
    'log': TransformPair(
        forward=lambda x: np.log(np.maximum(x, 1e-6)),
        inverse=lambda x: np.exp(x),
        name='log'
    ),
}


@dataclass
class ProfileFeatures:
    """Features for profile matching"""
    rolling_mean_4: float
    rolling_std_4: float
    rolling_cv_4: float
    seasonal_mean: float  # mean for this retail week historically
    trend_slope: float    # recent trend
    zeros_pct: float
    product_group: int
    department: int
    store_format: int


def extract_profile_features(
    sku_id: Tuple[int, int],
    week: int,
    df: pd.DataFrame,
    lookback: int = 4
) -> ProfileFeatures:
    """
    Extract features for profile matching from recent history.
    
    Args:
        sku_id: (Store, Product) tuple
        week: Target week number
        df: Full dataframe with history
        lookback: Number of weeks to look back
        
    Returns:
        ProfileFeatures object
    """
    store, product = sku_id
    
    # Get SKU history
    sku_df = df[(df['Store'] == store) & (df['Product'] == product)].sort_values('week')
    
    # Recent history (last 'lookback' weeks before target)
    recent = sku_df[sku_df['week'] < week].tail(lookback)
    
    if len(recent) == 0:
        # Cold start: use overall SKU statistics
        recent = sku_df[sku_df['week'] < week]
    
    # Rolling statistics
    sales = recent['sales'].values
    rolling_mean = sales.mean() if len(sales) > 0 else 0
    rolling_std = sales.std() if len(sales) > 1 else 0
    rolling_cv = rolling_std / (rolling_mean + 1e-9)
    
    # Seasonal context (same retail week in past)
    target_retail_week = df[(df['Store'] == store) & 
                            (df['Product'] == product) & 
                            (df['week'] == week)]['retail_week'].values
    if len(target_retail_week) > 0:
        target_retail_week = target_retail_week[0]
        seasonal = sku_df[sku_df['retail_week'] == target_retail_week]
        seasonal_mean = seasonal['sales'].mean() if len(seasonal) > 0 else rolling_mean
    else:
        seasonal_mean = rolling_mean
    
    # Trend
    if len(sales) >= 2:
        x = np.arange(len(sales))
        trend_slope = np.polyfit(x, sales, 1)[0]
    else:
        trend_slope = 0.0
    
    # Intermittency
    zeros_pct = (sales == 0).mean() if len(sales) > 0 else 0
    
    # Hierarchy
    metadata = df[(df['Store'] == store) & (df['Product'] == product)].iloc[0]
    
    # Handle categorical codes
    def get_cat_code(val):
        if hasattr(val, 'codes'):
            return int(val.codes[0]) if len(val.codes) > 0 else 0
        return 0
    
    return ProfileFeatures(
        rolling_mean_4=float(rolling_mean),
        rolling_std_4=float(rolling_std),
        rolling_cv_4=float(rolling_cv),
        seasonal_mean=float(seasonal_mean),
        trend_slope=float(trend_slope),
        zeros_pct=float(zeros_pct),
        product_group=get_cat_code(metadata['ProductGroup']),
        department=get_cat_code(metadata['Department']),
        store_format=get_cat_code(metadata['StoreFormat']),
    )


def compute_profile_distance(
    target: ProfileFeatures,
    candidate: ProfileFeatures,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute weighted distance between profiles.
    
    Prioritizes:
    - Recent volatility (CV)
    - Scale (rolling mean)
    - Product hierarchy
    """
    if weights is None:
        weights = {
            'rolling_mean_4': 0.3,
            'rolling_cv_4': 0.3,
            'seasonal_mean': 0.2,
            'trend_slope': 0.1,
            'product_group_match': 0.1,
        }
    
    # Continuous features: normalized absolute difference
    mean_diff = abs(target.rolling_mean_4 - candidate.rolling_mean_4) / (target.rolling_mean_4 + 1e-6)
    cv_diff = abs(target.rolling_cv_4 - candidate.rolling_cv_4)
    seasonal_diff = abs(target.seasonal_mean - candidate.seasonal_mean) / (target.seasonal_mean + 1e-6)
    trend_diff = abs(target.trend_slope - candidate.trend_slope) / (abs(target.trend_slope) + 1e-6)
    
    # Categorical: binary match
    pg_match = 0 if target.product_group == candidate.product_group else 1
    
    distance = (
        weights['rolling_mean_4'] * mean_diff +
        weights['rolling_cv_4'] * cv_diff +
        weights['seasonal_mean'] * seasonal_diff +
        weights['trend_slope'] * trend_diff +
        weights['product_group_match'] * pg_match
    )
    
    return float(distance)


def find_neighbor_profiles(
    target_sku: Tuple[int, int],
    target_week: int,
    df: pd.DataFrame,
    n_neighbors: int = 20,
    retail_week_window: int = 2
) -> pd.DataFrame:
    """
    Find similar non-stockout profiles for tail reconstruction.
    
    Args:
        target_sku: (Store, Product)
        target_week: Week with stockout
        df: Full demand history
        n_neighbors: Number of neighbors to return
        retail_week_window: Seasonal window (±weeks around target retail_week)
        
    Returns:
        DataFrame of neighbor observations (Store, Product, week, sales, ...)
    """
    # Extract target features
    target_features = extract_profile_features(target_sku, target_week, df)
    
    # Get target retail week for seasonality matching
    target_retail_week_vals = df[(df['Store'] == target_sku[0]) & 
                                  (df['Product'] == target_sku[1]) & 
                                  (df['week'] == target_week)]['retail_week'].values
    
    if len(target_retail_week_vals) == 0:
        warnings.warn(f"Target week {target_week} not found for SKU {target_sku}")
        return pd.DataFrame()
    
    target_retail_week = target_retail_week_vals[0]
    
    # Candidate pool: non-stockout weeks in similar season
    candidates = df[
        (df['in_stock'] == True) &  # Not stocked out
        (df['week'] < target_week) &  # Historical data only
        (df['retail_week'].between(
            target_retail_week - retail_week_window, 
            target_retail_week + retail_week_window
        ))
    ].copy()
    
    if len(candidates) == 0:
        warnings.warn(f"No candidates found for SKU {target_sku}, week {target_week}")
        return pd.DataFrame()
    
    # Compute distances for each candidate
    distances = []
    for idx, row in candidates.iterrows():
        cand_sku = (row['Store'], row['Product'])
        cand_week = row['week']
        
        # Extract features at this candidate point
        try:
            cand_features = extract_profile_features(cand_sku, cand_week, df)
            distance = compute_profile_distance(target_features, cand_features)
            distances.append(distance)
        except Exception as e:
            distances.append(np.inf)
    
    candidates['distance'] = distances
    
    # Return top k neighbors
    neighbors = candidates.nsmallest(n_neighbors, 'distance')
    
    return neighbors


def fit_empirical_quantiles_below_threshold(
    sales_data: np.ndarray,
    threshold: float,
    q_levels: np.ndarray
) -> np.ndarray:
    """
    Fit empirical quantiles to data censored at threshold.
    
    Only uses observations strictly below threshold (uncensored region).
    
    Args:
        sales_data: Historical sales (may include censored observations)
        threshold: Censoring level (stock level)
        q_levels: Quantile levels to estimate
        
    Returns:
        Quantile values (NaN for quantiles above censoring)
    """
    uncensored = sales_data[sales_data < threshold]
    
    if len(uncensored) == 0:
        return np.full(len(q_levels), np.nan)
    
    # Empirical quantiles
    quantiles = np.quantile(uncensored, q_levels)
    
    # Mark quantiles above threshold as NaN (need imputation)
    quantiles[quantiles >= threshold] = np.nan
    
    return quantiles


def splice_tail_from_neighbors(
    obs_quantiles: np.ndarray,
    neighbor_quantiles: np.ndarray,  # shape: (n_neighbors, n_quantiles)
    stock_level: float,
    q_levels: np.ndarray,
    aggregation: str = 'median'
) -> np.ndarray:
    """
    Splice observed quantiles (below stock) with neighbor tails (above stock).
    
    Args:
        obs_quantiles: Observed quantiles (NaN above stock)
        neighbor_quantiles: Full quantile functions from neighbors
        stock_level: Censoring threshold
        q_levels: Quantile levels
        aggregation: How to aggregate neighbor tails ('median', 'mean', 'weighted')
        
    Returns:
        Full reconstructed quantile function
    """
    reconstructed = obs_quantiles.copy()
    
    # Find splice point
    valid_mask = ~np.isnan(obs_quantiles)
    if valid_mask.sum() == 0:
        splice_idx = 0
    else:
        splice_idx = valid_mask.sum()
    
    if splice_idx >= len(q_levels):
        return reconstructed  # No censoring
    
    # Extract neighbor tails above stock
    neighbor_tails = []
    for neighbor_q in neighbor_quantiles:
        # Find where this neighbor crosses stock level
        cross_idx = np.searchsorted(neighbor_q, stock_level)
        if cross_idx < len(neighbor_q):
            tail = neighbor_q[cross_idx:]
            neighbor_tails.append(tail)
    
    if len(neighbor_tails) == 0:
        # Fallback: exponential extrapolation
        valid_vals = obs_quantiles[valid_mask]
        if len(valid_vals) > 1:
            obs_cv = valid_vals.std() / (valid_vals.mean() + 1e-9)
            tail_length = len(q_levels) - splice_idx
            extrapolated = stock_level * (1 + obs_cv * np.arange(1, tail_length + 1))
            reconstructed[splice_idx:] = extrapolated
        else:
            reconstructed[splice_idx:] = stock_level * 1.5
        return reconstructed
    
    # Align tails (different neighbors may cross at different quantiles)
    # Use the quantile levels from splice_idx onward
    tail_q_levels = q_levels[splice_idx:]
    
    # Interpolate each neighbor tail to common grid
    aligned_tails = []
    for tail in neighbor_tails:
        # Original quantile grid for this tail
        orig_grid = np.linspace(0, 1, len(tail))
        # Interpolate to target grid
        target_grid = tail_q_levels
        aligned = np.interp(target_grid, orig_grid, tail)
        aligned_tails.append(aligned)
    
    aligned_tails = np.array(aligned_tails)
    
    # Aggregate
    if aggregation == 'median':
        aggregated_tail = np.median(aligned_tails, axis=0)
    elif aggregation == 'mean':
        aggregated_tail = np.mean(aligned_tails, axis=0)
    else:  # weighted by inverse distance (TODO: implement)
        aggregated_tail = np.median(aligned_tails, axis=0)
    
    # Ensure continuity at splice point
    if splice_idx > 0 and splice_idx < len(reconstructed):
        offset = reconstructed[splice_idx - 1] - aggregated_tail[0]
        aggregated_tail = aggregated_tail + offset
        # Ensure monotonicity
        aggregated_tail = np.maximum.accumulate(aggregated_tail)
    
    # Insert tail
    reconstructed[splice_idx:] = aggregated_tail[:len(reconstructed) - splice_idx]
    
    return reconstructed


def impute_stockout_sip(
    sku_id: Tuple[int, int],
    week: int,
    stock_level: float,
    q_levels: np.ndarray,
    df: pd.DataFrame,
    transform_name: str = 'log',
    n_neighbors: int = 20,
    aggregation: str = 'median'
) -> pd.Series:
    """
    Full SIP reconstruction for a stockout week.
    
    This is the main entry point for stockout imputation.
    
    Args:
        sku_id: (Store, Product)
        week: Week with stockout
        stock_level: Stock available (= observed sales)
        q_levels: Quantile levels (e.g., [0.01, 0.05, ..., 0.99])
        df: Full demand dataframe
        transform_name: Variance-stabilizing transform
        n_neighbors: Number of neighbor profiles
        aggregation: Tail aggregation method
        
    Returns:
        Full reconstructed SIP as pd.Series indexed by q_levels
    """
    # Get transform functions
    if transform_name not in TRANSFORMS:
        warnings.warn(f"Unknown transform '{transform_name}', using identity")
        transform_name = 'identity'
    
    tfn = TRANSFORMS[transform_name].forward
    inv_tfn = TRANSFORMS[transform_name].inverse
    
    # Find neighbor profiles
    neighbors = find_neighbor_profiles(sku_id, week, df, n_neighbors)
    
    if len(neighbors) == 0:
        # Fallback: use simple tail mean from uncensored data
        store, product = sku_id
        sku_hist = df[(df['Store'] == store) & (df['Product'] == product) & (df['week'] < week)]
        if len(sku_hist) > 0:
            fallback_q = np.quantile(sku_hist['sales'].values, q_levels)
            return pd.Series(fallback_q, index=q_levels)
        else:
            # Last resort: uniform around stock level
            return pd.Series(np.linspace(0, stock_level * 2, len(q_levels)), index=q_levels)
    
    # Get observed quantiles below stock (in transform space)
    store, product = sku_id
    sku_hist = df[(df['Store'] == store) & (df['Product'] == product) & (df['week'] < week)]
    sales_hist = sku_hist['sales'].values
    
    if len(sales_hist) == 0:
        # Cold start: use neighbor quantiles directly
        neighbor_sales = neighbors['sales'].values
        transformed_neighbors = tfn(neighbor_sales)
        obs_q_transformed = np.quantile(transformed_neighbors, q_levels)
    else:
        transformed_sales = tfn(sales_hist)
        obs_q_transformed = fit_empirical_quantiles_below_threshold(
            transformed_sales, tfn(stock_level), q_levels
        )
    
    # Extract neighbor quantiles (in transform space)
    neighbor_quantiles_transformed = []
    for _, neighbor in neighbors.iterrows():
        n_store, n_product = neighbor['Store'], neighbor['Product']
        n_week = neighbor['week']
        
        # Get this neighbor's historical data up to their week
        n_hist = df[
            (df['Store'] == n_store) & 
            (df['Product'] == n_product) & 
            (df['week'] <= n_week)
        ]['sales'].values
        
        if len(n_hist) > 0:
            n_q_transformed = np.quantile(tfn(n_hist), q_levels)
            neighbor_quantiles_transformed.append(n_q_transformed)
    
    neighbor_quantiles_transformed = np.array(neighbor_quantiles_transformed)
    
    # Splice in transform space
    reconstructed_q_transformed = splice_tail_from_neighbors(
        obs_q_transformed,
        neighbor_quantiles_transformed,
        tfn(stock_level),
        q_levels,
        aggregation
    )
    
    # Back-transform
    reconstructed_q = inv_tfn(reconstructed_q_transformed)
    
    # Bias correction for log/log1p transforms
    if transform_name in ['log', 'log1p']:
        # Estimate variance in log-space from neighbors
        neighbor_log_vars = [np.var(tfn(neighbors['sales'].values))]
        correction_factor = np.exp(np.median(neighbor_log_vars) / 2)
        reconstructed_q *= correction_factor
    
    # Ensure monotonicity and non-negativity
    reconstructed_q = np.maximum(0, reconstructed_q)
    reconstructed_q = np.maximum.accumulate(reconstructed_q)
    
    return pd.Series(reconstructed_q, index=q_levels)


def _impute_single_stockout(
    row: pd.Series,
    df: pd.DataFrame,
    surd_transforms: pd.DataFrame,
    q_levels: np.ndarray,
    n_neighbors: int
) -> Tuple[Tuple[int, int, int], Optional[pd.Series]]:
    """
    Helper function to impute a single stockout (for parallel execution).
    
    Returns:
        Tuple of ((store, product, week), imputed_sip) or ((store, product, week), None) on failure
    """
    sku_id = (row['Store'], row['Product'])
    week = row['week']
    stock_level = row['sales']
    
    # Get transform for this SKU
    try:
        if sku_id in surd_transforms.index:
            transform_name = surd_transforms.loc[sku_id, 'best_transform']
        else:
            transform_name = 'log'
    except:
        transform_name = 'log'
    
    # Impute
    try:
        sip = impute_stockout_sip(
            sku_id, week, stock_level, q_levels, df,
            transform_name, n_neighbors
        )
        return ((sku_id[0], sku_id[1], week), sip)
    except Exception as e:
        return ((sku_id[0], sku_id[1], week), None)


def impute_all_stockouts(
    df: pd.DataFrame,
    surd_transforms: pd.DataFrame,
    q_levels: np.ndarray,
    state_df: Optional[pd.DataFrame] = None,
    n_neighbors: int = 20,
    verbose: bool = True,
    n_jobs: int = -1
) -> Dict[Tuple[int, int, int], pd.Series]:
    """
    Impute SIPs for all detected stockout weeks (parallelized).
    
    Args:
        df: Demand dataframe with 'in_stock' flag
        surd_transforms: Per-SKU best transforms
        q_levels: Quantile levels
        state_df: Optional inventory state for precise stock level detection
        n_neighbors: Number of neighbors for matching
        verbose: Print progress
        n_jobs: Number of parallel jobs (-1 = all cores, 1 = sequential)
        
    Returns:
        Dictionary mapping (Store, Product, week) -> imputed SIP
    """
    stockouts = df[df['in_stock'] == False]
    
    if verbose:
        n_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        print(f"Imputing {len(stockouts)} stockout observations using {n_cores} cores...")
    
    # Parallel execution
    if n_jobs == 1:
        # Sequential fallback
        results = [
            _impute_single_stockout(row, df, surd_transforms, q_levels, n_neighbors)
            for _, row in stockouts.iterrows()
        ]
    else:
        # Parallel processing
        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
            delayed(_impute_single_stockout)(row, df, surd_transforms, q_levels, n_neighbors)
            for _, row in stockouts.iterrows()
        )
    
    # Collect results
    imputed_sips = {}
    failed_count = 0
    for key, sip in results:
        if sip is not None:
            imputed_sips[key] = sip
        else:
            failed_count += 1
    
    if verbose:
        print(f"✅ Successfully imputed {len(imputed_sips)} / {len(stockouts)} stockouts")
        if failed_count > 0:
            print(f"⚠️  {failed_count} imputations failed")
    
    return imputed_sips


def validate_imputation_quality(
    imputed_sips: Dict,
    df: pd.DataFrame,
    coverage_levels: List[float] = [0.8, 0.9, 0.95]
) -> pd.DataFrame:
    """
    Validate imputed SIPs on held-out non-stockout data.
    
    Strategy: Artificially censor high-stock weeks and check if
    imputation recovers true demand quantiles.
    """
    # TODO: Implement cross-validation
    pass

