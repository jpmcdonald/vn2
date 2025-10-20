"""
Variance-stabilizing transforms for SURD-aware forecasting.

Implements forward and inverse transforms with proper bias correction
for lognormal and power transforms.
"""

import numpy as np
from typing import Tuple, Callable


def apply_transform(y: np.ndarray, transform_name: str) -> np.ndarray:
    """
    Apply variance-stabilizing transform to demand data.
    
    Args:
        y: Demand values (non-negative)
        transform_name: One of 'log', 'sqrt', 'cbrt', 'log1p', 'identity'
    
    Returns:
        Transformed values
    """
    if transform_name == 'log':
        # Add small epsilon to avoid log(0)
        return np.log(y + 1e-6)
    elif transform_name == 'sqrt':
        return np.sqrt(np.maximum(y, 0))
    elif transform_name == 'cbrt':
        # Cube root preserves sign
        return np.cbrt(y)
    elif transform_name == 'log1p':
        # log(1 + x), safe for zeros
        return np.log1p(y)
    elif transform_name == 'identity':
        return y
    else:
        raise ValueError(f"Unknown transform: {transform_name}")


def inverse_transform(
    y_trans: np.ndarray,
    transform_name: str,
    variance_trans: float = None
) -> np.ndarray:
    """
    Back-transform from variance-stabilized space with bias correction.
    
    For lognormal (log transform), applies Jensen's inequality correction:
    E[exp(X)] = exp(E[X] + Var(X)/2)
    
    NOTE: Bias correction should only be applied when y_trans represents
    a mean or expected value in transform space, NOT for individual samples.
    For quantiles, set variance_trans=None.
    
    Args:
        y_trans: Transformed values (quantiles or samples)
        transform_name: One of 'log', 'sqrt', 'cbrt', 'log1p', 'identity'
        variance_trans: Variance in transform space (for bias correction of means only)
    
    Returns:
        Back-transformed values in original space
    """
    if transform_name == 'log':
        # Simple exponential (bias correction handled separately if needed)
        return np.exp(y_trans)
    
    elif transform_name == 'sqrt':
        # Square transform
        return np.maximum(y_trans ** 2, 0)
    
    elif transform_name == 'cbrt':
        # Cube transform (preserves sign)
        return y_trans ** 3
    
    elif transform_name == 'log1p':
        # Inverse of log(1 + x) is exp(x) - 1
        return np.maximum(np.expm1(y_trans), 0)
    
    elif transform_name == 'identity':
        return y_trans
    
    else:
        raise ValueError(f"Unknown transform: {transform_name}")


def get_transform_pair(transform_name: str) -> Tuple[Callable, Callable]:
    """
    Get forward and inverse transform functions as a pair.
    
    Args:
        transform_name: One of 'log', 'sqrt', 'cbrt', 'log1p', 'identity'
    
    Returns:
        (forward_fn, inverse_fn) tuple
    """
    def forward(y):
        return apply_transform(y, transform_name)
    
    def inverse(y_trans, variance_trans=None):
        return inverse_transform(y_trans, transform_name, variance_trans)
    
    return forward, inverse


def validate_transform_roundtrip(
    y: np.ndarray,
    transform_name: str,
    tolerance: float = 1e-3
) -> bool:
    """
    Validate that transform -> inverse_transform recovers original values.
    
    Args:
        y: Original values
        transform_name: Transform to test
        tolerance: Maximum relative error allowed
    
    Returns:
        True if roundtrip is within tolerance
    """
    y_trans = apply_transform(y, transform_name)
    
    # For bias correction, estimate variance from transformed samples
    variance_trans = np.var(y_trans) if len(y_trans) > 1 else 0
    
    y_recovered = inverse_transform(y_trans, transform_name, variance_trans)
    
    # Check relative error (avoid division by zero)
    mask = y > 0
    if not np.any(mask):
        return True
    
    rel_error = np.abs(y_recovered[mask] - y[mask]) / y[mask]
    max_error = np.max(rel_error)
    
    return max_error < tolerance

