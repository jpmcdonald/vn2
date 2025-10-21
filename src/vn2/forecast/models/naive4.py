"""
Naive 4-week rolling average forecaster with density estimation.

Simple baseline: mean of last 4 weeks as point forecast, with Poisson-based
density for probabilistic forecasts.
"""

import numpy as np
import pandas as pd
from typing import Optional
from scipy.stats import poisson

from .base import BaseForecaster, ForecastConfig


class Naive4WeekForecaster(BaseForecaster):
    """
    Naive 4-week rolling average forecaster.
    
    Point forecast: mean of last 4 weeks
    Density: Poisson distribution with lambda = point forecast (or discretized normal)
    """
    
    def __init__(
        self,
        quantiles: np.ndarray = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]),
        horizon: int = 2,
        use_normal: bool = False
    ):
        config = ForecastConfig(
            quantiles=quantiles,
            horizon=horizon,
            transform_name='none'
        )
        super().__init__(config, name="Naive4Week")
        self.use_normal = use_normal
        self.last_4_vals_ = None
        self.point_forecast_ = None
    
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> 'Naive4WeekForecaster':
        """
        Fit by storing last 4 observations.
        
        Args:
            y: Historical demand series
            X: Not used
        
        Returns:
            self
        """
        # Get last 4 values
        y_vals = y.values
        if len(y_vals) >= 4:
            self.last_4_vals_ = y_vals[-4:]
        elif len(y_vals) > 0:
            # If less than 4 weeks, use what we have and repeat
            self.last_4_vals_ = np.pad(y_vals, (4 - len(y_vals), 0), mode='edge')[-4:]
        else:
            # No data: default to zeros
            self.last_4_vals_ = np.zeros(4)
        
        # Compute point forecast
        self.point_forecast_ = np.mean(self.last_4_vals_)
        
        self.is_fitted_ = True
        return self
    
    def predict_quantiles(
        self,
        steps: int = 2,
        X_future: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate quantile forecasts using Poisson or normal distribution.
        
        Args:
            steps: Number of steps ahead
            X_future: Not used
        
        Returns:
            DataFrame with quantile forecasts for each step
        """
        if not self.is_fitted_:
            # Return zeros if not fitted
            result = pd.DataFrame(
                0.0,
                index=range(1, steps + 1),
                columns=self.config.quantiles
            )
            result.index.name = 'step'
            return result
        
        # Use point forecast for all horizons
        lambda_param = max(0.1, self.point_forecast_)  # Avoid zero
        
        forecasts = []
        for step in range(1, steps + 1):
            if self.use_normal:
                # Normal approximation with sigma = sqrt(lambda)
                from scipy.stats import norm
                sigma = max(1.0, np.sqrt(lambda_param))
                quantile_vals = norm.ppf(self.config.quantiles, loc=lambda_param, scale=sigma)
                quantile_vals = np.maximum(0, quantile_vals)  # Truncate at zero
            else:
                # Poisson distribution
                quantile_vals = poisson.ppf(self.config.quantiles, mu=lambda_param)
            
            forecasts.append(quantile_vals)
        
        result = pd.DataFrame(
            forecasts,
            index=range(1, steps + 1),
            columns=self.config.quantiles
        )
        result.index.name = 'step'
        return result
    
    def predict_point(self, steps: int = 2) -> np.ndarray:
        """Return point forecast (same for all horizons)."""
        if not self.is_fitted_:
            return np.zeros(steps)
        return np.full(steps, self.point_forecast_)

