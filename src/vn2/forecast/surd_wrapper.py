"""
SURD Transform Wrapper for Forecast Models.

SURD (Systematic Unsupervised Representation Discovery) selects variance-stabilizing
transforms per time series. This wrapper applies the selected transform before
any forecast model and inverts after prediction.

Key Benefits:
- Stabilizes variance across the series
- Can improve forecast accuracy and calibration
- Works with any BaseForecaster as a wrapper

Supported Transforms:
- identity: No transformation
- log: Natural log (for positive, multiplicative series)
- log1p: log(1 + x) (for series with zeros)
- sqrt: Square root (for count data)
- cbrt: Cube root (for heavy-tailed distributions)

Usage:
    # Wrap an existing forecaster with SURD transform
    base_forecaster = ETSForecaster(config)
    surd_forecaster = SURDWrapper(base_forecaster, transform='log1p')
    surd_forecaster.fit(y, X)
    quantiles = surd_forecaster.predict_quantiles(steps=3)
"""

from __future__ import annotations

from typing import Dict, Optional, Callable
import numpy as np
import pandas as pd

from vn2.forecast.models.base import BaseForecaster, ForecastConfig


# Transform functions and their inverses
TRANSFORMS: Dict[str, Dict[str, Callable]] = {
    'identity': {
        'forward': lambda x: x,
        'inverse': lambda x: x,
    },
    'log': {
        'forward': lambda x: np.log(np.maximum(x, 1e-10)),
        'inverse': lambda x: np.exp(x),
    },
    'log1p': {
        'forward': lambda x: np.log1p(x),
        'inverse': lambda x: np.expm1(x),
    },
    'sqrt': {
        'forward': lambda x: np.sqrt(np.maximum(x, 0)),
        'inverse': lambda x: np.square(x),
    },
    'cbrt': {
        'forward': lambda x: np.cbrt(x),
        'inverse': lambda x: np.power(x, 3),
    },
}


def select_best_transform(y: np.ndarray) -> str:
    """Select the best variance-stabilizing transform for a series.
    
    Uses coefficient of variation (CV) reduction as the selection criterion.
    
    Args:
        y: Time series values
    
    Returns:
        Name of best transform
    """
    y = np.array(y, dtype=float)
    y_positive = y[y > 0] if np.any(y > 0) else y
    
    if len(y_positive) < 2:
        return 'identity'
    
    # Compute CV for each transform
    best_transform = 'identity'
    best_cv_stability = float('inf')
    
    for name, funcs in TRANSFORMS.items():
        try:
            y_transformed = funcs['forward'](y_positive)
            
            # Skip if transform produces inf/nan
            if not np.all(np.isfinite(y_transformed)):
                continue
            
            # Compute rolling CV stability
            if len(y_transformed) >= 4:
                window = min(len(y_transformed) // 2, 12)
                rolling_mean = pd.Series(y_transformed).rolling(window).mean().dropna()
                rolling_std = pd.Series(y_transformed).rolling(window).std().dropna()
                
                if len(rolling_mean) > 0 and rolling_mean.abs().min() > 1e-10:
                    rolling_cv = rolling_std / rolling_mean.abs()
                    cv_stability = rolling_cv.std()  # Lower = more stable
                    
                    if cv_stability < best_cv_stability:
                        best_cv_stability = cv_stability
                        best_transform = name
        except Exception:
            continue
    
    return best_transform


class SURDWrapper(BaseForecaster):
    """Wrapper that applies SURD transforms to any forecaster.
    
    This wrapper:
    1. Transforms the input series using the selected (or auto-selected) transform
    2. Fits the base forecaster on transformed data
    3. Inverts the predictions back to original space
    
    For quantile forecasts, the inversion preserves rank ordering but may not
    preserve exact coverage due to Jensen's inequality.
    
    Supports per-SKU transform lookup from a surd_transforms_df when sku_id
    is set by the pipeline before fit().
    """
    
    def __init__(
        self,
        base_forecaster: BaseForecaster,
        transform: str = 'auto',
        transform_config: Optional[Dict] = None,
        surd_transforms_df: Optional[pd.DataFrame] = None,
    ):
        """Initialize SURD wrapper.
        
        Args:
            base_forecaster: The forecaster to wrap
            transform: Transform name, 'auto' for automatic selection,
                       or 'lookup' to use surd_transforms_df with sku_id
            transform_config: Optional configuration for transforms
            surd_transforms_df: Pre-computed per-SKU transforms (Store, Product, best_transform)
        """
        super().__init__(base_forecaster.config, name=f"SURD_{base_forecaster.name}")
        self.base_forecaster = base_forecaster
        self.transform_name = transform
        self.transform_config = transform_config or {}
        self.surd_transforms_df = surd_transforms_df
        self.sku_id: Optional[tuple] = None
        
        self._selected_transform: str = 'identity'

    def _lookup_transform(self) -> str:
        """Look up the per-SKU transform from surd_transforms_df."""
        if self.surd_transforms_df is None or self.sku_id is None:
            return 'identity'
        df = self.surd_transforms_df
        if isinstance(df.index, pd.MultiIndex):
            try:
                return df.loc[self.sku_id, 'best_transform']
            except KeyError:
                return 'identity'
        mask = (df['Store'] == self.sku_id[0]) & (df['Product'] == self.sku_id[1])
        if mask.any():
            return df.loc[mask, 'best_transform'].iloc[0]
        return 'identity'

    def _forward(self, x: np.ndarray) -> np.ndarray:
        return TRANSFORMS[self._selected_transform]['forward'](x)

    def _inverse(self, x: np.ndarray) -> np.ndarray:
        return TRANSFORMS[self._selected_transform]['inverse'](x)
    
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> 'SURDWrapper':
        """Fit the wrapped forecaster on transformed data.
        
        Args:
            y: Time series to fit
            X: Optional exogenous features
        
        Returns:
            Self for chaining
        """
        y_values = y.values if hasattr(y, 'values') else np.array(y)
        
        # Select transform
        if self.transform_name == 'lookup':
            self._selected_transform = self._lookup_transform()
        elif self.transform_name == 'auto':
            self._selected_transform = select_best_transform(y_values)
        else:
            self._selected_transform = self.transform_name
        
        # Transform data
        y_transformed = pd.Series(self._forward(y_values), index=y.index if hasattr(y, 'index') else None)
        
        # Fit base forecaster on transformed data
        self.base_forecaster.fit(y_transformed, X)
        self.is_fitted_ = True
        
        # Store metadata
        self.set_metadata('surd_transform', self._selected_transform)
        
        return self
    
    def predict_quantiles(
        self,
        steps: int = 3,
        X_future: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Predict quantiles and invert back to original space.
        
        Args:
            steps: Number of steps ahead to forecast
            X_future: Optional future exogenous features
        
        Returns:
            DataFrame with quantile forecasts in original space
        """
        if not self.is_fitted_:
            raise ValueError("Forecaster must be fitted before prediction")
        
        # Get quantiles in transformed space
        quantiles_transformed = self.base_forecaster.predict_quantiles(steps, X_future)
        
        # Invert each quantile back to original space
        quantiles_original = quantiles_transformed.copy()
        
        for col in quantiles_original.columns:
            quantiles_original[col] = self._inverse(quantiles_transformed[col].values)
        
        # Ensure non-negative and monotonic
        quantiles_original = quantiles_original.clip(lower=0)
        
        # Ensure monotonicity across quantiles for each horizon
        for idx in quantiles_original.index:
            row = quantiles_original.loc[idx].values
            quantiles_original.loc[idx] = np.maximum.accumulate(row)
        
        return quantiles_original
    
    @property
    def selected_transform(self) -> str:
        """Get the selected transform name."""
        return self._selected_transform


def apply_surd_to_forecaster(
    forecaster_class,
    config: ForecastConfig,
    transform: str = 'auto',
    **forecaster_kwargs
) -> SURDWrapper:
    """Factory function to create a SURD-wrapped forecaster.
    
    Args:
        forecaster_class: Class of forecaster to wrap (e.g., ETSForecaster)
        config: Forecast configuration
        transform: Transform name or 'auto'
        **forecaster_kwargs: Additional kwargs for the base forecaster
    
    Returns:
        SURD-wrapped forecaster instance
    """
    base = forecaster_class(config, **forecaster_kwargs)
    return SURDWrapper(base, transform=transform)


def compute_surd_selections(
    demand_df: pd.DataFrame,
    min_observations: int = 12
) -> pd.DataFrame:
    """Compute best SURD transform for each SKU.
    
    Args:
        demand_df: DataFrame with columns [store, product, week, demand]
        min_observations: Minimum observations required for selection
    
    Returns:
        DataFrame with columns [store, product, best_transform]
    """
    results = []
    
    for (store, product), group in demand_df.groupby(['store', 'product']):
        y = group['demand'].values
        
        if len(y) < min_observations:
            transform = 'identity'
        else:
            transform = select_best_transform(y)
        
        results.append({
            'store': store,
            'product': product,
            'best_transform': transform
        })
    
    return pd.DataFrame(results)

