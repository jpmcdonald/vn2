"""
LightGBM Point Forecaster (deterministic MSE objective).

For Jensen's inequality comparison: point forecast + service level policy.
"""

import numpy as np
import pandas as pd
from typing import Optional
import lightgbm as lgb

from .base import BaseForecaster, ForecastConfig


class LightGBMPointForecaster(BaseForecaster):
    """
    LightGBM point forecaster using MSE objective.
    
    Returns median (0.5 quantile) as point forecast, with symmetric
    prediction intervals derived from residuals for density evaluation.
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
        super().__init__(config, name="LightGBM_Point")
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_data_in_leaf = min_data_in_leaf
        
        self.model = None
        self.residual_std = None
        
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> 'LightGBMPointForecaster':
        """Fit LightGBM with MSE objective"""
        
        # Prepare features
        if X is None or len(X) == 0:
            # Create simple lag features from y
            X = pd.DataFrame(index=y.index)
            for lag in [1, 2, 4, 8, 13, 26, 52]:
                X[f'lag_{lag}'] = y.shift(lag)
            X = X.fillna(0)
        
        # Align indices
        if len(X) != len(y):
            X = X.loc[y.index]
            if not X.index.is_unique:
                X = X[~X.index.duplicated(keep='last')]
        
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: X={len(X)}, y={len(y)}")
        
        X_vals = X.fillna(X.mean()).values
        X_vals = np.nan_to_num(X_vals, nan=0.0)
        y_vals = y.values
        
        # Train LightGBM with MSE
        train_data = lgb.Dataset(X_vals, label=y_vals)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'learning_rate': self.learning_rate,
            'min_data_in_leaf': self.min_data_in_leaf,
            'verbosity': -1,
            'force_col_wise': True
        }
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[train_data],
            callbacks=[lgb.log_evaluation(period=0)]
        )
        
        # Compute residual std for PI construction
        y_pred = self.model.predict(X_vals)
        residuals = y_vals - y_pred
        self.residual_std = np.std(residuals)
        
        self.is_fitted_ = True
        return self
    
    def predict_quantiles(
        self,
        steps: int = 2,
        X_future: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate quantiles from point forecast + Normal residuals.
        
        Returns symmetric quantiles around point prediction using
        residual standard deviation.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted")
        
        if X_future is None or len(X_future) < steps:
            raise ValueError("X_future required with at least 'steps' rows")
        
        quantiles_dict = {}
        
        for step in range(1, steps + 1):
            idx = min(step - 1, len(X_future) - 1)
            X_step = X_future.iloc[idx:idx+1].fillna(X_future.mean()).values
            X_step = np.nan_to_num(X_step, nan=0.0)
            
            # Point prediction
            y_pred = self.model.predict(X_step)[0]
            y_pred = max(0, y_pred)  # Non-negative
            
            # Symmetric quantiles using Normal approximation
            from scipy import stats
            quantile_values = []
            for q in self.config.quantiles:
                z = stats.norm.ppf(q)
                q_val = y_pred + z * self.residual_std
                q_val = max(0, q_val)  # Non-negative
                quantile_values.append(q_val)
            
            quantiles_dict[step] = quantile_values
        
        df = pd.DataFrame(quantiles_dict, index=self.config.quantiles).T
        df.index.name = 'step'
        
        return df

