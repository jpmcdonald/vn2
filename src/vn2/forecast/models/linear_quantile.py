"""
Linear Quantile Regression forecaster.

Uses sklearn's QuantileRegressor to directly predict quantiles
with a linear model. Fast baseline for high-dimensional features.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from sklearn.linear_model import QuantileRegressor
from .base import BaseForecaster, ForecastConfig


class LinearQuantileForecaster(BaseForecaster):
    """
    Linear Quantile Regression forecaster.
    
    Trains separate linear models for each quantile.
    Fast and interpretable baseline.
    """
    
    def __init__(
        self, 
        config: ForecastConfig,
        alpha: float = 1.0,
        solver: str = 'highs'
    ):
        """
        Args:
            config: Forecast configuration
            alpha: L1 regularization strength
            solver: 'highs' (default), 'interior-point', or 'revised simplex'
        """
        super().__init__(config, name="LinearQuantile")
        self.alpha = alpha
        self.solver = solver
        self.models: Dict[float, QuantileRegressor] = {}
        self.feature_names = None
        
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> 'LinearQuantileForecaster':
        """
        Fit linear quantile regression models.
        
        Args:
            y: Time series of demand
            X: Feature matrix (required for this model)
        """
        if X is None or len(X) == 0:
            raise ValueError("LinearQuantileForecaster requires features (X)")
        
        self.feature_names = X.columns.tolist()
        X_vals = X.values
        y_vals = y.values
        
        # Train one model per quantile
        for q in self.config.quantiles:
            model = QuantileRegressor(
                quantile=q,
                alpha=self.alpha,
                solver=self.solver
            )
            
            try:
                model.fit(X_vals, y_vals)
                self.models[q] = model
            except Exception as e:
                # If fitting fails for a quantile, use fallback
                # (e.g., predict constant)
                self.models[q] = None
        
        self.is_fitted_ = True
        return self
    
    def predict_quantiles(
        self, 
        steps: int = 2, 
        X_future: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate quantile forecasts.
        
        Args:
            steps: Number of steps ahead (1-2)
            X_future: Future features (required)
            
        Returns:
            DataFrame with index=steps and columns=quantiles
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call .fit() first.")
        
        if X_future is None or len(X_future) < steps:
            raise ValueError("X_future required and must have at least 'steps' rows")
        
        X_future_vals = X_future.iloc[:steps].values
        
        quantiles_dict = {}
        
        for step in range(1, steps + 1):
            x = X_future_vals[step - 1:step]  # Shape (1, n_features)
            
            step_quantiles = []
            for q in self.config.quantiles:
                model = self.models.get(q)
                
                if model is not None:
                    try:
                        pred = model.predict(x)[0]
                        # Ensure non-negative
                        pred = max(pred, 0)
                    except Exception:
                        # Fallback to median
                        median_model = self.models.get(0.5)
                        if median_model is not None:
                            pred = max(median_model.predict(x)[0], 0)
                        else:
                            pred = 0.0
                else:
                    # Model failed to fit, use simple fallback
                    pred = 0.0
                
                step_quantiles.append(pred)
            
            quantiles_dict[step] = step_quantiles
        
        # Convert to DataFrame
        df = pd.DataFrame(quantiles_dict, index=self.config.quantiles).T
        df.index.name = 'step'
        
        return df

