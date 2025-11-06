"""
LightGBM Quantile Regression for density forecasting.

Trains separate models for each quantile using LightGBM's quantile objective.
Fast, handles mixed patterns, and works well with intermittent demand.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from .base import BaseForecaster, ForecastConfig

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class LightGBMQuantileForecaster(BaseForecaster):
    """
    LightGBM-based quantile regression forecaster.
    
    Trains one model per quantile level, using engineered features.
    """
    
    def __init__(
        self,
        config: ForecastConfig,
        max_depth: int = 6,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        n_estimators: int = 100,
        min_data_in_leaf: int = 20
    ):
        super().__init__(config, name="LightGBM_Quantile")
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")
        
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_data_in_leaf = min_data_in_leaf
        
        self.models_= {}  # Dict of quantile -> trained model
        self.feature_names_ = None
    
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> 'LightGBMQuantileForecaster':
        """
        Fit separate LightGBM models for each quantile.
        
        Args:
            y: Time series of demand
            X: Optional feature matrix (if None, will create basic features)
        """
        # Build features if not provided
        if X is None or X.empty:
            X, y = self._prepare_data_from_series(y)
        else:
            # Use provided features, ensure alignment
            common_idx = X.index.intersection(y.index)
            if len(common_idx) == 0:
                raise ValueError("No common indices between X and y")
            X = X.loc[common_idx].copy()
            y = y.loc[common_idx].copy()
            
            # Remove duplicate indices (keep last occurrence)
            if not X.index.is_unique:
                X = X[~X.index.duplicated(keep='last')]
            if not y.index.is_unique:
                y = y[~y.index.duplicated(keep='last')]
            
            # Final alignment after deduplication
            final_idx = X.index.intersection(y.index)
            X = X.loc[final_idx]
            y = y.loc[final_idx]
        
        if len(X) < self.min_data_in_leaf * 2:
            # Not enough data to train
            self.models_ = {}
            self._y_train = y  # Store for fallback
            return self
        
        self.feature_names_ = X.columns.tolist()
        self._y_train = y  # Store for fallback quantiles
        
        # Train a model for each quantile
        for q in self.config.quantiles:
            params = {
                'objective': 'quantile',
                'alpha': q,
                'metric': 'quantile',
                'max_depth': self.max_depth,
                'num_leaves': self.num_leaves,
                'learning_rate': self.learning_rate,
                'min_data_in_leaf': self.min_data_in_leaf,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 1,
                'verbosity': -1,
                'seed': self.config.random_state,
                'num_threads': 1  # Parallelize at SKU level, not within model
            }
            
            try:
                model = lgb.LGBMRegressor(
                    **params,
                    n_estimators=self.n_estimators
                )
                model.fit(X, y, eval_set=[(X, y)], verbose=False)
                self.models_[q] = model
            except Exception as e:
                # If training fails for this quantile, skip it
                continue
        
        return self
    
    def _prepare_data_from_series(self, y: pd.Series) -> tuple:
        """
        Build feature matrix and target from a time series.
        
        Args:
            y: Time series of demand
            
        Returns:
            X: Feature DataFrame
            y: Target Series (cleaned)
        """
        y = y.copy()
        X = pd.DataFrame(index=y.index)
        
        # Lag features
        for lag in [1, 2, 4, 8, 13, 52]:
            if len(y) > lag:
                X[f'lag_{lag}'] = y.shift(lag)
        
        # Rolling statistics
        for window in [4, 8, 13]:
            if len(y) > window:
                X[f'rolling_mean_{window}'] = y.rolling(window, min_periods=1).mean()
                X[f'rolling_std_{window}'] = y.rolling(window, min_periods=1).std().fillna(0)
                X[f'rolling_cv_{window}'] = (X[f'rolling_std_{window}'] / 
                                              (X[f'rolling_mean_{window}'] + 1e-9))
        
        # Intermittency features
        X['is_zero'] = (y == 0).astype(int)
        X['zeros_pct_13'] = X['is_zero'].rolling(13, min_periods=1).mean()
        
        # Trend
        X['time_idx'] = np.arange(len(y))
        
        # Drop NaN rows
        valid_idx = X.notna().all(axis=1)
        X = X[valid_idx].fillna(0)
        y = y[valid_idx]
        
        return X, y
    
    def predict_quantiles(
        self, 
        steps: int = 2, 
        X_future: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate quantile forecasts for each horizon.
        
        Args:
            steps: Number of steps ahead to forecast
            X_future: Future features (required for LightGBM)
            
        Returns:
            DataFrame with index=steps and columns=quantiles
        """
        # Calculate fallback values from historical data (avoid pure zeros)
        if hasattr(self, '_y_train') and self._y_train is not None:
            # Use historical quantiles as fallback
            fallback_quantiles = np.quantile(self._y_train[self._y_train > 0], self.config.quantiles) if len(self._y_train[self._y_train > 0]) > 0 else np.ones(len(self.config.quantiles)) * 0.1
            # Ensure minimum values to prevent zero-cost predictions
            fallback_quantiles = np.maximum(fallback_quantiles, 0.1)
        else:
            # Conservative fallback: assume at least some demand
            fallback_quantiles = np.maximum(np.array(self.config.quantiles) * 2.0, 0.1)
        
        if not self.models_:
            # Model not trained, use fallback
            result = pd.DataFrame(
                [fallback_quantiles] * steps, 
                index=range(1, steps + 1), 
                columns=self.config.quantiles
            )
            result.index.name = 'step'
            return result
        
        if X_future is None or len(X_future) < steps:
            # No features provided, use fallback
            result = pd.DataFrame(
                [fallback_quantiles] * steps, 
                index=range(1, steps + 1), 
                columns=self.config.quantiles
            )
            result.index.name = 'step'
            return result
        
        # Predict for each step using provided features
        forecasts = []
        for step_idx in range(steps):
            X_step = X_future.iloc[step_idx:step_idx+1]
            quantile_values = []
            for i, q in enumerate(self.config.quantiles):
                if q in self.models_:
                    pred = self.models_[q].predict(X_step)[0]
                    # Apply minimum variance floor and ensure monotonicity
                    pred = max(pred, fallback_quantiles[i] * 0.1)  # At least 10% of fallback
                    quantile_values.append(pred)
                else:
                    quantile_values.append(fallback_quantiles[i])
            
            # Ensure monotonicity: q_i <= q_j for i < j
            quantile_values = np.array(quantile_values)
            for i in range(1, len(quantile_values)):
                quantile_values[i] = max(quantile_values[i], quantile_values[i-1])
            
            forecasts.append(quantile_values)
        
        result = pd.DataFrame(
            forecasts,
            index=range(1, steps + 1),
            columns=self.config.quantiles
        )
        result.index.name = 'step'
        return result

