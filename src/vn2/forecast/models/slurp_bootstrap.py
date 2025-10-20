"""
SLURP Conditional Bootstrap forecaster.

Leverages the SLURP infrastructure for feature-conditional resampling.
Samples from similar historical observations based on features like:
- Week of year (seasonality)
- Rolling CV (volatility)
- Trend
- Recent demand patterns
"""

import numpy as np
import pandas as pd
from typing import Optional
from sklearn.neighbors import NearestNeighbors
from .base import BaseForecaster, ForecastConfig


class SLURPBootstrapForecaster(BaseForecaster):
    """
    SLURP Conditional Bootstrap forecaster.
    
    Uses k-nearest neighbors in feature space to bootstrap from similar
    historical observations, preserving the stochastic relationships.
    
    Supports stockout-aware mode: treats stockouts as censored observations
    and samples from the conditional distribution of positive demand.
    """
    
    def __init__(
        self, 
        config: ForecastConfig,
        n_neighbors: int = 50,
        n_bootstrap: int = 1000,
        stockout_aware: bool = False
    ):
        """
        Args:
            config: Forecast configuration
            n_neighbors: Number of similar historical observations to sample from
            n_bootstrap: Number of bootstrap samples for quantile estimation
            stockout_aware: If True, handle stockouts as censored observations
        """
        super().__init__(config, name="SLURP_Bootstrap")
        self.n_neighbors = n_neighbors
        self.n_bootstrap = n_bootstrap
        self.stockout_aware = stockout_aware
        self.history_y = None
        self.history_X = None
        self.history_in_stock = None
        self.nn_model = None
        
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> 'SLURPBootstrapForecaster':
        """
        Fit by storing historical observations for bootstrap sampling.
        
        Args:
            y: Time series of demand
            X: Feature matrix (calendar, lags, rolling stats)
                If stockout_aware=True, X must contain 'in_stock' column
        """
        self.history_y = y.values
        
        # Store stockout indicators if available
        if self.stockout_aware and X is not None and 'in_stock' in X.columns:
            # Align in_stock with y
            if len(X) != len(y):
                X = X.loc[y.index]
                if not X.index.is_unique:
                    X = X[~X.index.duplicated(keep='last')]
            
            self.history_in_stock = X['in_stock'].values
        else:
            self.history_in_stock = None
        
        if X is not None and len(X) > 0:
            # CRITICAL: Ensure X and y have the same length
            # X might have duplicate indices or more rows due to how features are created
            # We need to align X with y's index
            if len(X) != len(y):
                # If lengths don't match, take only the rows corresponding to y's index
                X = X.loc[y.index]
                
                # If X still has duplicates (due to feature creation), drop them
                if not X.index.is_unique:
                    # Keep the last occurrence of each duplicate index
                    X = X[~X.index.duplicated(keep='last')]
            
            # Ensure we now have matching lengths
            if len(X) != len(y):
                raise ValueError(f"X and y length mismatch after alignment: X={len(X)}, y={len(y)}")

            
            # Use features for conditional sampling
            # Handle missing values by filling with column means
            X_vals = X.fillna(X.mean()).values
            
            # If still NaN (all-NaN columns), fill with 0
            X_vals = np.nan_to_num(X_vals, nan=0.0)
            
            self.history_X = X_vals
            
            # Fit k-NN model for finding similar observations
            self.nn_model = NearestNeighbors(
                n_neighbors=min(self.n_neighbors, len(y)),
                metric='euclidean'
            )
            self.nn_model.fit(self.history_X)
        else:
            # Fallback: unconditional bootstrap
            self.history_X = None
            self.nn_model = None
        
        self.is_fitted_ = True
        return self
    
    def _stockout_aware_sample(
        self, 
        candidate_indices: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Bootstrap sample with stockout awareness.
        
        For stockout observations, sample from the conditional distribution
        of observed positive demand instead of using the (censored) zero value.
        
        Args:
            candidate_indices: Indices to sample from (neighbors or all history)
            rng: Random number generator
            
        Returns:
            Bootstrap samples (n_bootstrap,)
        """
        bootstrap_samples = []
        
        # Get in-stock observations from candidates
        in_stock_mask = self.history_in_stock[candidate_indices]
        in_stock_indices = candidate_indices[in_stock_mask]
        stockout_indices = candidate_indices[~in_stock_mask]
        
        # Count of each type in candidate pool
        n_in_stock = len(in_stock_indices)
        n_stockout = len(stockout_indices)
        
        if n_in_stock == 0:
            # No in-stock observations in candidates, fall back to simple sampling
            return rng.choice(self.history_y[candidate_indices], size=self.n_bootstrap, replace=True)
        
        # Sample maintaining the stockout rate
        stockout_rate = n_stockout / len(candidate_indices) if len(candidate_indices) > 0 else 0
        n_stockout_samples = int(self.n_bootstrap * stockout_rate)
        n_instock_samples = self.n_bootstrap - n_stockout_samples
        
        # Sample in-stock observations normally
        if n_instock_samples > 0:
            instock_samples = rng.choice(
                self.history_y[in_stock_indices],
                size=n_instock_samples,
                replace=True
            )
            bootstrap_samples.extend(instock_samples)
        
        # For stockouts, sample from positive observed demand
        if n_stockout_samples > 0:
            # Get positive in-stock values from candidates (or all history if too few)
            positive_instock = self.history_y[in_stock_indices]
            positive_instock = positive_instock[positive_instock > 0]
            
            if len(positive_instock) == 0:
                # No positive values, use all in-stock (including zeros)
                positive_instock = self.history_y[in_stock_indices]
            
            if len(positive_instock) > 0:
                stockout_samples = rng.choice(
                    positive_instock,
                    size=n_stockout_samples,
                    replace=True
                )
                bootstrap_samples.extend(stockout_samples)
            else:
                # Fallback: add zeros
                bootstrap_samples.extend([0] * n_stockout_samples)
        
        return np.array(bootstrap_samples)
    
    def predict_quantiles(
        self, 
        steps: int = 2, 
        X_future: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate quantile forecasts via conditional bootstrap.
        
        Args:
            steps: Number of steps ahead (1-2)
            X_future: Future features for conditional sampling
            
        Returns:
            DataFrame with index=steps and columns=quantiles
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call .fit() first.")
        
        rng = np.random.default_rng(self.config.random_state)
        
        quantiles_dict = {}
        
        for step in range(1, steps + 1):
            if self.nn_model is not None and X_future is not None and len(X_future) >= step:
                # Conditional bootstrap: find similar historical observations
                # Use the available feature row (might be less than step if X_future is short)
                idx = min(step - 1, len(X_future) - 1)
                future_features = X_future.iloc[idx:idx+1].fillna(X_future.mean()).values
                
                # Handle remaining NaNs
                future_features = np.nan_to_num(future_features, nan=0.0)
                
                # Find k-nearest neighbors
                distances, indices = self.nn_model.kneighbors(future_features)
                neighbor_indices = indices[0]
                
                # Sample with replacement from neighbors
                if self.stockout_aware and self.history_in_stock is not None:
                    # Stockout-aware sampling
                    bootstrap_samples = self._stockout_aware_sample(
                        neighbor_indices, rng
                    )
                else:
                    # Standard sampling
                    bootstrap_samples = rng.choice(
                        self.history_y[neighbor_indices],
                        size=self.n_bootstrap,
                        replace=True
                    )
            else:
                # Unconditional bootstrap: sample from all history
                if self.stockout_aware and self.history_in_stock is not None:
                    # Stockout-aware sampling from all history
                    bootstrap_samples = self._stockout_aware_sample(
                        np.arange(len(self.history_y)), rng
                    )
                else:
                    # Standard sampling
                    bootstrap_samples = rng.choice(
                        self.history_y,
                        size=self.n_bootstrap,
                        replace=True
                    )
            
            # Ensure non-negative
            bootstrap_samples = np.maximum(bootstrap_samples, 0)
            
            # Compute quantiles
            quantile_values = np.quantile(bootstrap_samples, self.config.quantiles)
            quantiles_dict[step] = quantile_values
        
        # Convert to DataFrame
        df = pd.DataFrame(quantiles_dict, index=self.config.quantiles).T
        df.index.name = 'step'
        
        return df

