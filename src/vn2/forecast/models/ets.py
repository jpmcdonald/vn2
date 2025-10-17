"""
ETS (Error, Trend, Seasonal) exponential smoothing forecaster.

Uses statsmodels ETSModel for state-space exponential smoothing with
automatic prediction intervals via simulation.
"""

import numpy as np
import pandas as pd
from typing import Optional
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from .base import BaseForecaster, ForecastConfig


class ETSForecaster(BaseForecaster):
    """
    Exponential Smoothing State Space Model (ETS) forecaster.
    """
    
    def __init__(
        self, 
        config: ForecastConfig,
        error: str = 'add',
        trend: Optional[str] = None,
        seasonal: Optional[str] = None,
        seasonal_periods: int = 52
    ):
        """
        Args:
            config: Forecast configuration
            error: Error type ('add' or 'mul')
            trend: Trend type (None, 'add', 'mul')
            seasonal: Seasonal type (None, 'add', 'mul')
            seasonal_periods: Number of periods in a season (52 for weekly annual)
        """
        name = f"ETS({error[0]},{trend[0] if trend else 'N'},{seasonal[0] if seasonal else 'N'})"
        super().__init__(config, name=name)
        self.error = error
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.fitted_model = None
    
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> 'ETSForecaster':
        """
        Fit ETS model to time series.
        
        Args:
            y: Time series of demand
            X: Optional exogenous features (not used by ETS)
        """
        try:
            # For intermittent series, use simple models (no seasonality)
            # and handle zeros by adding small constant
            y_vals = y.values
            
            # If too many zeros, use simpler model
            zero_rate = (y_vals == 0).mean()
            if zero_rate > 0.7:
                # Very intermittent - use simple exponential smoothing
                error = 'add'
                trend = None
                seasonal = None
            else:
                error = self.error
                trend = self.trend
                seasonal = self.seasonal
            
            # Add small constant if multiplicative components are used
            if error == 'mul' or trend == 'mul' or seasonal == 'mul':
                y_adj = y_vals + 0.1
            else:
                y_adj = y_vals
            
            # Fit model
            model = ETSModel(
                y_adj,
                error=error,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=self.seasonal_periods if seasonal else None
            )
            
            self.fitted_model = model.fit(disp=False, maxiter=100)
            self.is_fitted_ = True
            
        except Exception as e:
            # If fitting fails, mark as not fitted
            self.fitted_model = None
            self.is_fitted_ = False
            raise RuntimeError(f"Failed to fit ETS model: {e}")
        
        return self
    
    def predict_quantiles(
        self, 
        steps: int = 2, 
        X_future: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate quantile forecasts via simulation.
        
        Args:
            steps: Number of steps ahead (1-2)
            X_future: Optional future features (not used)
            
        Returns:
            DataFrame with index=steps and columns=quantiles
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call .fit() first.")
        
        # Use statsmodels simulation for prediction intervals
        # Get simulated paths
        n_sims = 1000
        simulations = self.fitted_model.simulate(
            nsimulations=steps,
            repetitions=n_sims,
            random_state=self.config.random_state
        )
        
        # simulations shape: (steps, n_sims)
        quantiles_dict = {}
        for step in range(1, steps + 1):
            step_sims = simulations[step - 1, :]
            
            # Ensure non-negative (for additive errors, can go negative)
            step_sims = np.maximum(step_sims, 0)
            
            # Compute quantiles
            quantile_values = np.quantile(step_sims, self.config.quantiles)
            quantiles_dict[step] = quantile_values
        
        # Convert to DataFrame
        df = pd.DataFrame(quantiles_dict, index=self.config.quantiles).T
        df.index.name = 'step'
        
        return df

