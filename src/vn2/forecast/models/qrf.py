"""
Quantile Random Forest forecaster.

Uses sklearn RandomForestRegressor with quantile extraction from leaf nodes.
"""

import numpy as np
import pandas as pd
from typing import Optional
from sklearn.ensemble import RandomForestRegressor

from .base import BaseForecaster, ForecastConfig


class QRFForecaster(BaseForecaster):
    """
    Quantile Random Forest forecaster.
    
    Uses RandomForestRegressor and extracts quantiles from the
    distribution of predictions across trees.
    """
    
    def __init__(
        self,
        config: ForecastConfig,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
        random_state: int = 42
    ):
        super().__init__(config, name="QRF")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        self.model = None
        
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> 'QRFForecaster':
        """Fit Random Forest"""
        
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
        
        # Train Random Forest
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=1  # Single-threaded per model (parallelism at pipeline level)
        )
        
        self.model.fit(X_vals, y_vals)
        
        self.is_fitted_ = True
        return self
    
    def predict_quantiles(
        self,
        steps: int = 2,
        X_future: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate quantiles from tree predictions.
        
        Each tree gives a prediction; quantiles are computed from
        the distribution of these predictions.
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
            
            # Get predictions from all trees
            tree_predictions = np.array([
                tree.predict(X_step)[0]
                for tree in self.model.estimators_
            ])
            
            # Ensure non-negative
            tree_predictions = np.maximum(tree_predictions, 0)
            
            # Compute quantiles from tree predictions
            quantile_values = np.quantile(tree_predictions, self.config.quantiles)
            quantile_values = np.maximum(quantile_values, 0)  # Ensure non-negative
            
            quantiles_dict[step] = quantile_values
        
        df = pd.DataFrame(quantiles_dict, index=self.config.quantiles).T
        df.index.name = 'step'
        
        return df

