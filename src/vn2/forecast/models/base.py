"""
Base forecaster interface for density forecasting.

All forecast models must implement this interface to be compatible with
the training pipeline and optimization framework.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ForecastConfig:
    """Configuration for a forecaster"""
    quantiles: np.ndarray
    horizon: int = 2  # forecast steps
    transform_name: str = 'identity'
    random_state: int = 42


class BaseForecaster(ABC):
    """
    Abstract base class for all density forecasters.
    
    All models must generate quantile forecasts for h=1 and h=2 weeks ahead.
    """
    
    def __init__(self, config: ForecastConfig, name: str = "BaseForecaster"):
        self.config = config
        self.name = name
        self.quantiles = config.quantiles
        self.horizon = config.horizon
        self.transform_name = config.transform_name
        self.is_fitted_ = False
        self.metadata_ = {}
        
    @abstractmethod
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> 'BaseForecaster':
        """
        Fit the forecaster to training data.
        
        Args:
            y: Time series of demand (already transformed if needed)
            X: Optional exogenous features (calendar, lags, etc.)
            
        Returns:
            self (fitted)
        """
        pass
    
    @abstractmethod
    def predict_quantiles(
        self, 
        steps: int = 2, 
        X_future: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate quantile forecasts for h=1..steps ahead.
        
        Args:
            steps: Number of steps ahead to forecast
            X_future: Optional future exogenous features
            
        Returns:
            DataFrame with:
              - index: forecast steps (1, 2, ..., steps)
              - columns: quantile levels from self.quantiles
              - values: forecasted demand at each quantile
        """
        pass
    
    def simulate_paths(
        self, 
        steps: int = 2, 
        n_sims: int = 1000,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate demand paths for Monte Carlo optimization.
        
        Default implementation samples from quantile function via
        inverse transform. Models can override for native sampling.
        
        Args:
            steps: Forecast horizon
            n_sims: Number of simulation paths
            seed: Random seed
            
        Returns:
            Array of shape (n_sims, steps) with simulated demand
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before simulation")
        
        rng = np.random.default_rng(seed)
        quantiles_df = self.predict_quantiles(steps)
        
        # Sample uniformly, interpolate quantiles
        sims = np.zeros((n_sims, steps))
        for step_idx in range(steps):
            step = step_idx + 1
            q_vals = quantiles_df.loc[step].values
            # Inverse transform sampling
            u = rng.uniform(0, 1, n_sims)
            sims[:, step_idx] = np.interp(u, self.quantiles, q_vals)
        
        return sims
    
    def get_params(self) -> Dict[str, Any]:
        """Return model parameters for checkpointing"""
        return {
            'quantiles': self.quantiles,
            'horizon': self.horizon,
            'transform_name': self.transform_name,
            'metadata': self.metadata_
        }
    
    def set_metadata(self, key: str, value: Any):
        """Store metadata (e.g., SKU ID, fold, fit time)"""
        self.metadata_[key] = value


class TransformMixin:
    """Mixin for handling SURD transforms"""
    
    TRANSFORMS = {
        'identity': (lambda x: x, lambda x: x),
        'log': (lambda x: np.log(np.maximum(x, 1e-6)), np.exp),
        'log1p': (np.log1p, np.expm1),
        'sqrt': (lambda x: np.sqrt(np.maximum(x, 0)), np.square),
        'cbrt': (np.cbrt, lambda x: np.power(x, 3)),
    }
    
    @staticmethod
    def apply_transform(y: pd.Series, transform_name: str) -> pd.Series:
        """Apply forward transform"""
        if transform_name not in TransformMixin.TRANSFORMS:
            raise ValueError(f"Unknown transform: {transform_name}")
        forward_fn, _ = TransformMixin.TRANSFORMS[transform_name]
        return pd.Series(forward_fn(y.values), index=y.index)
    
    @staticmethod
    def inverse_transform(y: pd.Series, transform_name: str) -> pd.Series:
        """Apply inverse transform"""
        if transform_name not in TransformMixin.TRANSFORMS:
            raise ValueError(f"Unknown transform: {transform_name}")
        _, inverse_fn = TransformMixin.TRANSFORMS[transform_name]
        return pd.Series(inverse_fn(y.values), index=y.index)
    
    @staticmethod
    def bias_correction(
        y_transformed: np.ndarray, 
        transform_name: str, 
        variance: float
    ) -> np.ndarray:
        """
        Apply bias correction for Jensen's inequality when back-transforming.
        
        For lognormal: E[exp(X)] = exp(μ + σ²/2) not exp(E[X])
        """
        if transform_name in ['log', 'log1p']:
            correction = np.exp(variance / 2)
            return y_transformed * correction
        return y_transformed

