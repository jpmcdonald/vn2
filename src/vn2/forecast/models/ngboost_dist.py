"""
NGBoost distributional regression forecaster.

Uses NGBoost to learn full predictive distributions (LogNormal or Poisson).
Natural for density forecasting with parametric distributions.
"""

import numpy as np
import pandas as pd
from typing import Optional
from .base import BaseForecaster, ForecastConfig


class NGBoostForecaster(BaseForecaster):
    """
    NGBoost distributional forecaster.
    
    Learns distribution parameters via gradient boosting.
    Supports LogNormal (for log-transformed data) or Poisson (for counts).
    """
    
    def __init__(
        self, 
        config: ForecastConfig,
        dist: str = 'LogNormal',
        n_estimators: int = 100,
        learning_rate: float = 0.01
    ):
        """
        Args:
            config: Forecast configuration
            dist: 'LogNormal' or 'Poisson'
            n_estimators: Number of boosting iterations
            learning_rate: Learning rate
        """
        super().__init__(config, name=f"NGBoost_{dist}")
        self.dist_name = dist
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.model = None
        
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> 'NGBoostForecaster':
        """
        Fit NGBoost model.
        
        Args:
            y: Time series of demand
            X: Feature matrix (required)
        """
        if X is None or len(X) == 0:
            raise ValueError("NGBoostForecaster requires features (X)")
        
        try:
            from ngboost import NGBRegressor
            from ngboost.distns import LogNormal, Poisson
        except ImportError:
            raise ImportError("NGBoost not installed. Run: pip install ngboost")
        
        # Select distribution
        if self.dist_name == 'LogNormal':
            dist = LogNormal
            # For LogNormal, ensure positive values
            y_vals = np.maximum(y.values, 0.01)
        elif self.dist_name == 'Poisson':
            dist = Poisson
            y_vals = y.values
        else:
            raise ValueError(f"Unsupported distribution: {self.dist_name}")
        
        X_vals = X.values
        
        try:
            self.model = NGBRegressor(
                Dist=dist,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                verbose=False
            )
            
            self.model.fit(X_vals, y_vals)
            self.is_fitted_ = True
            
        except Exception as e:
            self.model = None
            self.is_fitted_ = False
            raise RuntimeError(f"Failed to fit NGBoost model: {e}")
        
        return self
    
    def predict_quantiles(
        self, 
        steps: int = 2, 
        X_future: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate quantile forecasts from learned distributions.
        
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
        
        # Get predictive distributions
        forecast_dists = self.model.pred_dist(X_future_vals)
        
        quantiles_dict = {}
        
        for step in range(1, steps + 1):
            dist = forecast_dists[step - 1]
            
            # Get quantiles from the distribution
            quantile_values = []
            for q in self.config.quantiles:
                try:
                    # Use the distribution's ppf (percent point function / inverse CDF)
                    q_val = dist.ppf(q)
                    # Ensure non-negative
                    q_val = max(q_val, 0)
                except Exception:
                    # Fallback to median if quantile extraction fails
                    q_val = max(dist.ppf(0.5), 0) if q >= 0.5 else 0
                
                quantile_values.append(q_val)
            
            quantiles_dict[step] = quantile_values
        
        # Convert to DataFrame
        df = pd.DataFrame(quantiles_dict, index=self.config.quantiles).T
        df.index.name = 'step'
        
        return df

