"""
Croston and related methods for intermittent demand forecasting.

Implements:
- Classic Croston (1972)
- SBA: Syntetos-Boylan Approximation (2001)
- TSB: Teunter-Syntetos-Babai (2011)
"""

import numpy as np
import pandas as pd
from typing import Optional
from .base import BaseForecaster, ForecastConfig


class CrostonForecaster(BaseForecaster):
    """
    Croston-family forecasters for intermittent demand.
    
    Separates demand into:
    - Interval: time between nonzero demands
    - Size: magnitude of nonzero demands
    
    Generates density forecasts via simulation.
    """
    
    def __init__(self, config: ForecastConfig, variant: str = 'classic', alpha: float = 0.1):
        """
        Args:
            config: Forecast configuration
            variant: 'classic', 'sba', or 'tsb'
            alpha: Smoothing parameter
        """
        name = f"Croston_{variant}"
        super().__init__(config, name=name)
        self.variant = variant
        self.alpha = alpha
        self.interval_forecast_ = None
        self.size_forecast_ = None
        self.probability_demand_ = None
        self.is_fitted_ = False
        
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> 'CrostonForecaster':
        """Fit Croston model to time series"""
        y_vals = y.values
        
        # Identify nonzero demands
        nonzero_indices = np.where(y_vals > 0)[0]
        
        if len(nonzero_indices) == 0:
            # No demand observed - use conservative fallback
            self.interval_forecast_ = max(len(y_vals), 4)  # At least 4 periods
            self.size_forecast_ = 1.0  # Assume at least 1 unit when demand occurs
            self.probability_demand_ = 0.1  # Small but non-zero probability
            self.is_fitted_ = True
            return self
        
        # Extract intervals and sizes
        sizes = y_vals[nonzero_indices]
        
        if len(nonzero_indices) > 1:
            intervals = np.diff(nonzero_indices)
        else:
            intervals = np.array([len(y_vals)])
        
        # Exponential smoothing for interval and size
        interval_smooth = self._smooth(intervals, self.alpha)
        size_smooth = self._smooth(sizes, self.alpha)
        
        # Forecast depends on variant
        if self.variant == 'classic':
            # Croston (1972): E[D] = size / interval
            self.interval_forecast_ = max(interval_smooth, 1.0)  # At least 1 period
            self.size_forecast_ = max(size_smooth, 0.5)  # At least 0.5 units
            
        elif self.variant == 'sba':
            # Syntetos-Boylan Approximation (2001)
            # Adjusts for bias in Croston
            self.interval_forecast_ = max(interval_smooth, 1.0)
            self.size_forecast_ = max(size_smooth * (1 - self.alpha / 2), 0.5)
            
        elif self.variant == 'tsb':
            # Teunter-Syntetos-Babai (2011)
            # Directly forecasts probability and size
            p = 1 / interval_smooth if interval_smooth > 0 else 0.1
            self.probability_demand_ = min(max(p, 0.01), 0.9)  # Bound probability
            self.size_forecast_ = max(size_smooth, 0.5)
            self.interval_forecast_ = max(interval_smooth, 1.0)
        else:
            raise ValueError(f"Unknown variant: {self.variant}")
        
        self.is_fitted_ = True
        return self
    
    def _smooth(self, values: np.ndarray, alpha: float) -> float:
        """Simple exponential smoothing"""
        if len(values) == 0:
            return 0.0
        
        smooth = values[0]
        for val in values[1:]:
            smooth = alpha * val + (1 - alpha) * smooth
        return smooth
    
    def predict_quantiles(self, steps: int = 2, X_future: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate quantile forecasts via simulation.
        
        Args:
            steps: Number of steps ahead to forecast
            X_future: Future features (not used by Croston)
        
        Returns:
            DataFrame with index=steps and columns=quantiles
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted")
        
        # Generate via simulation, then compute empirical quantiles
        n_sims = 10000
        sims = self._simulate_demand(steps, n_sims)
        
        # Compute quantiles for each step
        quantiles_dict = {}
        for step in range(1, steps + 1):
            step_sims = sims[:, step - 1]
            quantile_values = np.quantile(step_sims, self.config.quantiles)
            quantiles_dict[step] = quantile_values
        
        # Convert to DataFrame (index=steps, columns=quantiles)
        df = pd.DataFrame(quantiles_dict, index=self.config.quantiles).T
        df.index.name = 'step'
        return df
    
    def _simulate_demand(self, steps: int, n_sims: int) -> np.ndarray:
        """Simulate demand paths"""
        rng = np.random.default_rng(self.config.random_state)
        sims = np.zeros((n_sims, steps))
        
        if self.variant == 'tsb' and self.probability_demand_ is not None:
            # TSB: use probability directly
            p = self.probability_demand_
            for step in range(steps):
                # Bernoulli × size
                occurs = rng.binomial(1, p, n_sims)
                sizes = rng.exponential(self.size_forecast_, n_sims)
                sims[:, step] = occurs * sizes
        else:
            # Classic/SBA: simulate via interval
            for sim_idx in range(n_sims):
                time_since_last = rng.exponential(self.interval_forecast_)
                for step in range(steps):
                    time_since_last += 1
                    if time_since_last >= self.interval_forecast_:
                        # Demand occurs
                        sims[sim_idx, step] = rng.exponential(self.size_forecast_)
                        time_since_last = 0
        
        return sims
    
    def simulate_paths(
        self, 
        steps: int = 2, 
        n_sims: int = 1000,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Override to use native simulation"""
        if not self.is_fitted_:
            raise ValueError("Model not fitted")
        
        # Use internal simulation
        rng_backup = np.random.get_state()
        if seed is not None:
            np.random.seed(seed)
        
        sims = self._simulate_demand(steps, n_sims)
        
        np.random.set_state(rng_backup)
        return sims

