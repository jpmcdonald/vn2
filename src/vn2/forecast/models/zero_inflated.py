"""
Zero-Inflated Poisson and Negative Binomial models for intermittent demand.

These models explicitly handle excess zeros by modeling:
1. Probability of zero (logistic regression)
2. Count distribution for non-zeros (Poisson or NegBin)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from .base import BaseForecaster, ForecastConfig

try:
    from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class ZeroInflatedForecaster(BaseForecaster):
    """Base class for Zero-Inflated models."""
    
    def __init__(self, config: ForecastConfig, model_type: str = 'poisson'):
        super().__init__(config, name=f"ZI_{model_type}")
        self.model_type = model_type
        self.fitted_model = None
        self.last_features = None
        
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for Zero-Inflated models. Install with: pip install statsmodels")
    
    def fit(self, history: pd.DataFrame, master_df: Optional[pd.DataFrame] = None, 
            surd_df: Optional[pd.DataFrame] = None) -> 'ZeroInflatedForecaster':
        """
        Fit Zero-Inflated model with exogenous features.
        
        Args:
            history: DataFrame with 'sales' and feature columns
            master_df: Not used
            surd_df: Not used
        """
        y = history['sales'].values
        
        # Build simple feature set from history
        # Use lags, rolling stats, and calendar features if available
        X = self._build_features(history)
        
        if len(X) < 10:  # Need minimum observations
            # Fallback to simple model with just intercept
            X = pd.DataFrame({'const': np.ones(len(y))}, index=history.index)
        
        # Align y and X
        y = y[X.index]
        
        try:
            if self.model_type == 'poisson':
                self.fitted_model = ZeroInflatedPoisson(
                    endog=y,
                    exog=X,
                    exog_infl=X  # Same features for inflation model
                ).fit(disp=False, maxiter=100)
            else:  # negbin
                self.fitted_model = ZeroInflatedNegativeBinomialP(
                    endog=y,
                    exog=X,
                    exog_infl=X,
                    p=2  # NB2 parameterization
                ).fit(disp=False, maxiter=100)
            
            self.last_features = X.iloc[-1].values
            
        except Exception as e:
            # If fitting fails, store None and handle in predict
            self.fitted_model = None
            self.last_features = None
        
        return self
    
    def _build_features(self, history: pd.DataFrame) -> pd.DataFrame:
        """Build simple feature matrix from history."""
        features = pd.DataFrame(index=history.index)
        features['const'] = 1.0
        
        sales = history['sales'].values
        
        # Lag features (if enough data)
        if len(sales) > 1:
            features['lag_1'] = pd.Series(sales).shift(1).fillna(0).values
        if len(sales) > 4:
            features['lag_4'] = pd.Series(sales).shift(4).fillna(0).values
        if len(sales) > 8:
            features['lag_8'] = pd.Series(sales).shift(8).fillna(0).values
        
        # Rolling mean (if enough data)
        if len(sales) > 4:
            features['rolling_mean_4'] = pd.Series(sales).rolling(4, min_periods=1).mean().values
        
        # Zero indicator
        features['is_zero'] = (sales == 0).astype(float)
        
        # Calendar features if available in history
        if 'week_of_year' in history.columns:
            # Cyclical encoding
            woy = history['week_of_year'].values
            features['week_sin'] = np.sin(2 * np.pi * woy / 52)
            features['week_cos'] = np.cos(2 * np.pi * woy / 52)
        
        # Drop rows with NaN from lag creation
        features = features.dropna()
        
        return features
    
    def predict_quantiles(self, forecast_origin_week: int, horizon_weeks: int) -> Dict[int, pd.Series]:
        """
        Generate quantile forecasts by simulating from the fitted distribution.
        """
        if self.fitted_model is None or self.last_features is None:
            # Model failed to fit, return zeros
            return {h: pd.Series(0.0, index=self.config.quantiles) for h in range(1, horizon_weeks + 1)}
        
        forecasts = {}
        
        # Use last observation's features for all horizons
        # In production, you'd iterate and update features per horizon
        X_future = self.last_features.reshape(1, -1)
        
        for h in range(1, horizon_weeks + 1):
            # Predict parameters
            mu = self.fitted_model.predict(X_future)[0]  # Mean count
            
            # Get inflation probability (probability of structural zero)
            if hasattr(self.fitted_model, 'predict_prob'):
                prob_zero = self.fitted_model.predict_prob(X_future)[0]
            else:
                # Fallback: use model's inflation model
                prob_zero = 0.0  # Conservative
            
            # Simulate from the distribution
            n_sims = 10000
            rng = np.random.default_rng(self.config.random_state + h)
            
            # Simulate from Zero-Inflated distribution
            # Step 1: Draw structural zeros
            is_structural_zero = rng.random(n_sims) < prob_zero
            
            # Step 2: Draw from count distribution for non-structural zeros
            if self.model_type == 'poisson':
                counts = rng.poisson(lam=max(mu, 0.01), size=n_sims)
            else:  # negbin
                # Approximate NegBin with Gamma-Poisson mixture
                # For simplicity, use overdispersed Poisson
                variance = mu * 1.5  # Assume some overdispersion
                counts = rng.negative_binomial(
                    n=max(mu**2 / (variance - mu + 1e-9), 0.1),
                    p=max(mu / (variance + 1e-9), 0.01),
                    size=n_sims
                )
            
            # Combine: structural zeros override counts
            simulated = np.where(is_structural_zero, 0, counts)
            
            # Compute quantiles
            quantile_values = np.quantile(simulated, self.config.quantiles)
            forecasts[h] = pd.Series(quantile_values, index=self.config.quantiles)
        
        return forecasts


class ZIPForecaster(ZeroInflatedForecaster):
    """Zero-Inflated Poisson forecaster."""
    def __init__(self, config: ForecastConfig):
        super().__init__(config, model_type='poisson')


class ZINBForecaster(ZeroInflatedForecaster):
    """Zero-Inflated Negative Binomial forecaster."""
    def __init__(self, config: ForecastConfig):
        super().__init__(config, model_type='negbin')

