"""
GLM Count Models (Poisson and Negative Binomial) for demand forecasting.

Parametric count models that return quantiles via CDF/PPF.
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings

from .base import BaseForecaster, ForecastConfig


class GLMCountForecaster(BaseForecaster):
    """
    Generalized Linear Model for count data (Poisson or NegativeBinomial).
    
    Uses statsmodels GLM with log link. Returns quantiles via fitted
    distribution's PPF (percent point function).
    """
    
    def __init__(
        self,
        config: ForecastConfig,
        family: str = 'poisson',
        name: str = None
    ):
        """
        Args:
            config: Forecast configuration
            family: 'poisson' or 'negbin' (negative binomial)
            name: Model name (auto-generated if None)
        """
        if name is None:
            name = f"GLM-{family.capitalize()}"
        super().__init__(config, name=name)
        
        self.family = family.lower()
        if self.family not in ['poisson', 'negbin']:
            raise ValueError(f"family must be 'poisson' or 'negbin', got {family}")
        
        self.model = None
        self.dispersion = 1.0  # For Poisson; estimated for NegBin
        
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> 'GLMCountForecaster':
        """Fit GLM with Poisson or NegBin family"""
        
        try:
            import statsmodels.api as sm
            from statsmodels.genmod.families import Poisson, NegativeBinomial
        except ImportError:
            raise ImportError("statsmodels required for GLM models. Install with: pip install statsmodels")
        
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
        X_vals = sm.add_constant(X_vals)  # Add intercept
        
        y_vals = y.values
        y_vals = np.maximum(y_vals, 0)  # Ensure non-negative
        y_vals = np.round(y_vals).astype(int)  # Count data
        
        # Fit GLM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if self.family == 'poisson':
                family = Poisson()
                self.model = sm.GLM(y_vals, X_vals, family=family).fit()
                self.dispersion = 1.0  # Poisson has fixed dispersion
            else:  # negbin
                # Use Poisson as initial fit, then estimate dispersion
                family = Poisson()
                poisson_model = sm.GLM(y_vals, X_vals, family=family).fit()
                
                # Estimate dispersion from Poisson residuals
                mu = poisson_model.predict(X_vals)
                residuals = y_vals - mu
                self.dispersion = np.var(residuals) / np.mean(mu) if np.mean(mu) > 0 else 1.0
                self.dispersion = max(1.0, self.dispersion)  # NegBin requires dispersion >= 1
                
                # Refit with NegBin (alpha = 1/dispersion)
                alpha = 1.0 / self.dispersion
                family = NegativeBinomial(alpha=alpha)
                self.model = sm.GLM(y_vals, X_vals, family=family).fit()
        
        self.is_fitted_ = True
        return self
    
    def predict_quantiles(
        self,
        steps: int = 2,
        X_future: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate quantiles from fitted Poisson/NegBin distribution.
        
        Uses PPF (inverse CDF) of the fitted distribution.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted")
        
        if X_future is None or len(X_future) < steps:
            raise ValueError("X_future required with at least 'steps' rows")
        
        import statsmodels.api as sm
        from scipy import stats
        
        quantiles_dict = {}
        
        for step in range(1, steps + 1):
            idx = min(step - 1, len(X_future) - 1)
            X_step = X_future.iloc[idx:idx+1].fillna(X_future.mean()).values
            X_step = np.nan_to_num(X_step, nan=0.0)
            X_step = sm.add_constant(X_step)
            
            # Predict mean (lambda for Poisson, mu for NegBin)
            mu = self.model.predict(X_step)[0]
            mu = max(0.1, mu)  # Avoid zero mean
            
            # Generate quantiles from distribution
            quantile_values = []
            for q in self.config.quantiles:
                if self.family == 'poisson':
                    q_val = stats.poisson.ppf(q, mu)
                else:  # negbin
                    # NegBin parameterization: n = mu^2 / (var - mu)
                    # var = mu * dispersion for our parameterization
                    var = mu * self.dispersion
                    if var <= mu:
                        # Fallback to Poisson if dispersion too low
                        q_val = stats.poisson.ppf(q, mu)
                    else:
                        n = mu ** 2 / (var - mu)
                        p = mu / var
                        q_val = stats.nbinom.ppf(q, n, p)
                
                q_val = max(0, q_val)  # Non-negative
                quantile_values.append(float(q_val))
            
            quantiles_dict[step] = quantile_values
        
        df = pd.DataFrame(quantiles_dict, index=self.config.quantiles).T
        df.index.name = 'step'
        
        return df

