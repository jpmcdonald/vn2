"""
k-NN Profile forecaster (Case-Based Reasoning).

Uses k-nearest neighbors to find similar historical demand profiles
and bootstraps from their outcomes. Leverages the profile matching
infrastructure from stockout imputation.
"""

import numpy as np
import pandas as pd
from typing import Optional
from sklearn.neighbors import NearestNeighbors
from .base import BaseForecaster, ForecastConfig


class KNNProfileForecaster(BaseForecaster):
    """
    k-NN Profile forecaster using case-based reasoning.
    
    Finds similar historical patterns and samples from their future outcomes.
    Similar to SLURP Bootstrap but with explicit profile matching.
    """
    
    def __init__(
        self, 
        config: ForecastConfig,
        n_neighbors: int = 20,
        lookback_weeks: int = 13
    ):
        """
        Args:
            config: Forecast configuration
            n_neighbors: Number of similar profiles to use
            lookback_weeks: Number of weeks to use for profile comparison
        """
        super().__init__(config, name="KNN_Profile")
        self.n_neighbors = n_neighbors
        self.lookback_weeks = lookback_weeks
        self.history_y = None
        self.history_profiles = None
        self.future_outcomes = None
        self.nn_model = None
        
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> 'KNNProfileForecaster':
        """
        Fit by creating historical profiles and outcomes.
        
        Args:
            y: Time series of demand
            X: Optional features (not used, relies on y patterns)
        """
        y_vals = y.values
        n = len(y_vals)
        
        # Build historical profiles and outcomes
        profiles = []
        outcomes = []
        
        # For each historical point (leaving room for lookback and future)
        for t in range(self.lookback_weeks, n - 2):
            # Profile: last lookback_weeks of demand
            profile = y_vals[t - self.lookback_weeks:t]
            
            # Outcome: next 2 weeks
            outcome = y_vals[t:t + 2]
            
            profiles.append(profile)
            outcomes.append(outcome)
        
        if len(profiles) == 0:
            # Not enough history
            self.history_y = y_vals
            self.history_profiles = None
            self.nn_model = None
            self.is_fitted_ = True
            return self
        
        self.history_profiles = np.array(profiles)
        self.future_outcomes = np.array(outcomes)
        
        # Fit k-NN model on profiles
        self.nn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(profiles)),
            metric='euclidean'
        )
        self.nn_model.fit(self.history_profiles)
        
        # Store full history for fallback
        self.history_y = y_vals
        
        self.is_fitted_ = True
        return self
    
    def predict_quantiles(
        self, 
        steps: int = 2, 
        X_future: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate quantile forecasts via profile matching.
        
        Args:
            steps: Number of steps ahead (1-2)
            X_future: Future features (not used)
            
        Returns:
            DataFrame with index=steps and columns=quantiles
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call .fit() first.")
        
        if self.nn_model is None or len(self.history_y) < self.lookback_weeks + 2:
            # Not enough history, use simple bootstrap
            rng = np.random.default_rng(self.config.random_state)
            
            quantiles_dict = {}
            for step in range(1, steps + 1):
                bootstrap_samples = rng.choice(
                    self.history_y,
                    size=1000,
                    replace=True
                )
                bootstrap_samples = np.maximum(bootstrap_samples, 0)
                quantile_values = np.quantile(bootstrap_samples, self.config.quantiles)
                quantiles_dict[step] = quantile_values
            
            df = pd.DataFrame(quantiles_dict, index=self.config.quantiles).T
            df.index.name = 'step'
            return df
        
        # Create current profile from most recent data
        current_profile = self.history_y[-self.lookback_weeks:]
        current_profile = current_profile.reshape(1, -1)
        
        # Find k-nearest neighbors
        distances, indices = self.nn_model.kneighbors(current_profile)
        neighbor_indices = indices[0]
        
        # Get their future outcomes
        neighbor_outcomes = self.future_outcomes[neighbor_indices]
        
        # Compute quantiles from neighbor outcomes
        quantiles_dict = {}
        
        for step in range(1, min(steps + 1, 3)):  # Max 2 steps
            if step <= neighbor_outcomes.shape[1]:
                step_outcomes = neighbor_outcomes[:, step - 1]
                # Ensure non-negative
                step_outcomes = np.maximum(step_outcomes, 0)
                quantile_values = np.quantile(step_outcomes, self.config.quantiles)
            else:
                # If step exceeds available outcomes, use last step
                step_outcomes = neighbor_outcomes[:, -1]
                step_outcomes = np.maximum(step_outcomes, 0)
                quantile_values = np.quantile(step_outcomes, self.config.quantiles)
            
            quantiles_dict[step] = quantile_values
        
        # Convert to DataFrame
        df = pd.DataFrame(quantiles_dict, index=self.config.quantiles).T
        df.index.name = 'step'
        
        return df

