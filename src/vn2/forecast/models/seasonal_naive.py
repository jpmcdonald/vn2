"""
Seasonal Naive forecaster with bootstrap prediction intervals.

Simple but effective baseline for seasonal data.
"""

import numpy as np
import pandas as pd
from typing import Optional
from .base import BaseForecaster, ForecastConfig


class SeasonalNaiveForecaster(BaseForecaster):
    """
    Seasonal naive with residual bootstrap for density forecasting.
    
    Forecast = last year's value + bootstrap residual
    """
    
    def __init__(self, config: ForecastConfig, season_length: int = 52):
        super().__init__(config, name=f"SeasonalNaive_S{season_length}")
        self.season_length = season_length
        self.last_values_ = None
        self.residuals_ = None
        
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> 'SeasonalNaiveForecaster':
        """Fit seasonal naive model"""
        y_vals = y.values
        
        if len(y_vals) < self.season_length:
            # Not enough history, use simple naive
            self.last_values_ = np.array([y_vals[-1], y_vals[-1]])
            self.residuals_ = y_vals - np.mean(y_vals)
        else:
            # Store last seasonal cycle
            self.last_values_ = y_vals[-self.season_length:]
            
            # Compute residuals for bootstrap
            residuals = []
            for i in range(self.season_length, len(y_vals)):
                lag_val = y_vals[i - self.season_length]
                residual = y_vals[i] - lag_val
                residuals.append(residual)
            
            self.residuals_ = np.array(residuals) if len(residuals) > 0 else np.array([0.0])
        
        self.is_fitted_ = True
        return self
    
    def predict_quantiles(
        self, 
        steps: int = 2, 
        X_future: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Generate quantile forecasts"""
        if not self.is_fitted_:
            raise ValueError("Model not fitted")
        
        # For h=1,2, use last values + residual quantiles
        quantiles_dict = {}
        
        for step in range(1, steps + 1):
            # Base forecast from seasonal lag
            if len(self.last_values_) >= step:
                base_forecast = self.last_values_[-step]
            else:
                base_forecast = self.last_values_[-1]
            
            # Add residual quantiles
            residual_quantiles = np.quantile(self.residuals_, self.quantiles)
            forecast_quantiles = base_forecast + residual_quantiles
            
            # Ensure non-negative
            forecast_quantiles = np.maximum(0, forecast_quantiles)
            
            quantiles_dict[step] = forecast_quantiles
        
        df = pd.DataFrame(quantiles_dict, index=self.quantiles).T
        df.index.name = 'step'
        return df
    
    def simulate_paths(
        self, 
        steps: int = 2, 
        n_sims: int = 1000,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Simulate paths via bootstrap residuals"""
        if not self.is_fitted_:
            raise ValueError("Model not fitted")
        
        rng = np.random.default_rng(seed)
        sims = np.zeros((n_sims, steps))
        
        for step in range(steps):
            # Base forecast
            if len(self.last_values_) > step:
                base = self.last_values_[-(step + 1)]
            else:
                base = self.last_values_[-1]
            
            # Bootstrap residuals
            residuals_sampled = rng.choice(self.residuals_, size=n_sims, replace=True)
            sims[:, step] = np.maximum(0, base + residuals_sampled)
        
        return sims

