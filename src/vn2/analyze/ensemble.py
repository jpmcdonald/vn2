"""
Ensemble methods: linear pool, cohort-gated selector, decision-level pooling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sklearn.isotonic import IsotonicRegression


def blend_quantiles(
    quantiles_dict: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    quantile_levels: np.ndarray,
    monotonic_fix: bool = True
) -> pd.DataFrame:
    """
    Blend quantiles from multiple models via weighted average.
    
    Args:
        quantiles_dict: {model_name: DataFrame(index=quantile_levels, columns=[step])}
        weights: {model_name: weight}, must sum to 1.0
        quantile_levels: array of quantile levels (e.g., [0.05, 0.10, ..., 0.95])
        monotonic_fix: apply isotonic regression to enforce monotonicity
    
    Returns:
        DataFrame(index=quantile_levels, columns=[step]) with blended quantiles
    """
    # Validate weights
    total_w = sum(weights.values())
    if not np.isclose(total_w, 1.0):
        raise ValueError(f"Weights must sum to 1.0, got {total_w}")
    
    # Check all models have same structure
    first_model = list(quantiles_dict.keys())[0]
    ref_df = quantiles_dict[first_model]
    steps = ref_df.columns.tolist()
    
    # Initialize blended result
    blended = pd.DataFrame(0.0, index=quantile_levels, columns=steps)
    
    # Weighted sum
    for model, w in weights.items():
        if model not in quantiles_dict:
            raise ValueError(f"Model {model} not in quantiles_dict")
        df = quantiles_dict[model]
        blended += w * df.loc[quantile_levels, steps]
    
    # Monotonic fix per step
    if monotonic_fix:
        for step in steps:
            y = blended[step].values
            if not np.all(np.diff(y) >= 0):
                # Apply isotonic regression
                iso = IsotonicRegression(increasing=True)
                y_fixed = iso.fit_transform(quantile_levels, y)
                blended[step] = y_fixed
    
    return blended


def cohort_selector_rules(
    cohort_features: pd.DataFrame,
    per_sku_champions: pd.DataFrame,
    features: List[str] = ['rate_bin', 'zero_bin', 'cv_bin', 'stockout_bin']
) -> Dict[Tuple, str]:
    """
    Learn simple rules: for each cohort combo, pick the most frequent champion.
    
    Args:
        cohort_features: DataFrame(store, product, rate_bin, zero_bin, cv_bin, stockout_bin, ...)
        per_sku_champions: DataFrame(store, product, model_name, win_share, ...)
        features: list of cohort feature names to use
    
    Returns:
        Dict mapping (rate_bin, zero_bin, cv_bin, stockout_bin) -> model_name
    """
    # Join
    merged = per_sku_champions.merge(cohort_features, on=['store', 'product'], how='left')
    
    # For each cohort combo, find model with highest total win_share
    rules = {}
    for combo, group in merged.groupby(features):
        win_sums = group.groupby('model_name')['win_share'].sum()
        best_model = win_sums.idxmax()
        rules[combo] = best_model
    
    return rules


def apply_cohort_selector(
    cohort_features: pd.DataFrame,
    rules: Dict[Tuple, str],
    features: List[str],
    fallback_model: str = 'qrf'
) -> pd.DataFrame:
    """
    Apply cohort rules to assign a model per SKU.
    
    Args:
        cohort_features: DataFrame(store, product, rate_bin, zero_bin, ...)
        rules: Dict from cohort_selector_rules
        features: list of feature names matching rule keys
        fallback_model: model to use if no rule matches
    
    Returns:
        DataFrame(store, product, selected_model)
    """
    def lookup(row):
        key = tuple(row[f] for f in features)
        return rules.get(key, fallback_model)
    
    result = cohort_features[['store', 'product']].copy()
    result['selected_model'] = cohort_features.apply(lookup, axis=1)
    return result


def decision_level_pool(
    cost_curves: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Pool expected cost curves across models and return pooled curve.
    
    Args:
        cost_curves: {model_name: array of expected costs over Q grid}
        weights: optional {model_name: weight}, defaults to uniform
    
    Returns:
        pooled_cost: array of pooled expected costs
    """
    if weights is None:
        weights = {m: 1.0 / len(cost_curves) for m in cost_curves}
    
    # Validate
    total_w = sum(weights.values())
    if not np.isclose(total_w, 1.0):
        raise ValueError(f"Weights must sum to 1.0, got {total_w}")
    
    # Check all curves have same length
    lengths = [len(c) for c in cost_curves.values()]
    if len(set(lengths)) > 1:
        raise ValueError(f"Cost curves have different lengths: {lengths}")
    
    pooled = np.zeros(lengths[0])
    for model, w in weights.items():
        pooled += w * cost_curves[model]
    
    return pooled


def grid_search_linear_pool_weights(
    eval_folds_path: Path,
    models: List[str],
    cost_col: str = 'sip_realized_cost_w2',
    weight_grid: Optional[np.ndarray] = None,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Grid search over linear pool weights to minimize aggregate cost.
    
    For simplicity, search over uniform grids for 2-model case (ZINB+QRF).
    
    Args:
        eval_folds_path: path to eval_folds parquet
        models: list of 2 model names to blend
        cost_col: cost column to minimize
        weight_grid: optional grid of weights for model[0], defaults to [0.0, 0.1, ..., 1.0]
        top_k: return top k weight combinations
    
    Returns:
        DataFrame(w_model0, w_model1, total_cost) sorted by cost
    """
    if len(models) != 2:
        raise NotImplementedError("Grid search currently supports 2 models only")
    
    if weight_grid is None:
        weight_grid = np.linspace(0.0, 1.0, 11)
    
    df = pd.read_parquet(eval_folds_path)
    df = df[df['model_name'].isin(models)].copy()
    
    # Pivot to get cost per (store, product, fold_idx) per model
    pivot = df.pivot_table(
        index=['store', 'product', 'fold_idx'],
        columns='model_name',
        values=cost_col,
        aggfunc='sum'
    ).reset_index()
    
    # Drop rows with missing models
    pivot = pivot.dropna(subset=models)
    
    results = []
    for w0 in weight_grid:
        w1 = 1.0 - w0
        # Blend costs
        blended_cost = w0 * pivot[models[0]] + w1 * pivot[models[1]]
        total = blended_cost.sum()
        results.append({
            f'w_{models[0]}': w0,
            f'w_{models[1]}': w1,
            'total_cost': total
        })
    
    result_df = pd.DataFrame(results).sort_values('total_cost').head(top_k)
    return result_df

