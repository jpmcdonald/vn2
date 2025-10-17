"""
Zero-Inflated Poisson (ZIP) and Zero-Inflated Negative Binomial (ZINB) forecasters.

These models explicitly handle intermittent demand by modeling:
1. The probability of zero demand (logistic regression)
2. The count distribution for non-zero demand (Poisson or Negative Binomial)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
from .base import BaseForecaster, ForecastConfig


class ZeroInflatedForecaster(BaseForecaster):
    """
    Zero-Inflated Poisson (ZIP) or Zero-Inflated Negative Binomial (ZINB) forecaster.
    """
    
    def __init__(self, config: ForecastConfig, model_type: str = 'poisson'):
        """
        Args:
            config: Forecast configuration
            model_type: 'poisson' for ZIP, 'negbin' for ZINB
        """
        name = f"ZI{model_type.upper()}"
        super().__init__(config, name=name)
        self.model_type = model_type
        self.fitted_model = None
    
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> 'ZeroInflatedForecaster':
        """
        Fit Zero-Inflated model to time series.
        
        Args:
            y: Time series of demand
            X: Optional exogenous features (not used in this simple implementation)
        """
        try:
            # For now, fit a simple model without exogenous features
            # statsmodels ZIP/ZINB requires exog parameter (can be None for intercept-only)
            # We need to create a constant (intercept) term
            import numpy as np
            exog_const = np.ones((len(y), 1))
            
            if self.model_type == 'poisson':
                model = ZeroInflatedPoisson(endog=y, exog=exog_const, exog_infl=exog_const)
            elif self.model_type == 'negbin':
                model = ZeroInflatedNegativeBinomialP(endog=y, exog=exog_const, exog_infl=exog_const)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            self.fitted_model = model.fit(disp=False, maxiter=100)
            self.exog_const = exog_const  # Store for prediction
            self.is_fitted_ = True
            
        except Exception as e:
            # If fitting fails, fall back to naive approach
            self.fitted_model = None
            self.is_fitted_ = False
            raise RuntimeError(f"Failed to fit {self.name} model: {e}")
        
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
        
        # Extract model parameters
        params = self.fitted_model.params
        
        if self.model_type == 'poisson':
            # params[0] = inflate_logit, params[1] = count_log_lambda
            p_zero = 1 / (1 + np.exp(-params[0]))  # Inverse logit
            count_lambda = np.exp(params[1])  # Inverse log
            
            # Simulate demand
            rng = np.random.default_rng(self.config.random_state)
            n_sims = 10000
            
            # For each step, generate simulations
            quantiles_dict = {}
            for step in range(1, steps + 1):
                # Bernoulli draw for zero-inflation
                is_zero = rng.binomial(1, p_zero, n_sims)
                
                # Poisson draws for non-zero
                nonzero_demands = rng.poisson(count_lambda, n_sims)
                
                # Combine: zero if is_zero==1, else nonzero_demand
                simulated_demands = np.where(is_zero, 0, nonzero_demands)
                
                # Compute quantiles
                quantile_values = np.quantile(simulated_demands, self.config.quantiles)
                quantiles_dict[step] = quantile_values
            
        elif self.model_type == 'negbin':
            # params[0] = inflate_logit, params[1] = count_log_mu, params[2] = count_log_alpha
            p_zero = 1 / (1 + np.exp(-params[0]))
            count_mu = np.exp(params[1])
            count_alpha = np.exp(params[2])
            
            # Convert to n, p for numpy negative_binomial
            # For NB parameterization: n = 1/alpha, p = 1/(1+mu*alpha)
            n_nb = 1 / count_alpha
            p_nb = 1 / (1 + count_mu * count_alpha)
            
            rng = np.random.default_rng(self.config.random_state)
            n_sims = 10000
            
            quantiles_dict = {}
            for step in range(1, steps + 1):
                is_zero = rng.binomial(1, p_zero, n_sims)
                nonzero_demands = rng.negative_binomial(n_nb, p_nb, n_sims)
                simulated_demands = np.where(is_zero, 0, nonzero_demands)
                quantile_values = np.quantile(simulated_demands, self.config.quantiles)
                quantiles_dict[step] = quantile_values
        
        # Convert to DataFrame
        df = pd.DataFrame(quantiles_dict, index=self.config.quantiles).T
        df.index.name = 'step'
        
        return df


class ZIPForecaster(ZeroInflatedForecaster):
    """Zero-Inflated Poisson forecaster (convenience wrapper)."""
    
    def __init__(self, config: ForecastConfig):
        super().__init__(config, model_type='poisson')


class ZINBForecaster(ZeroInflatedForecaster):
    """Zero-Inflated Negative Binomial forecaster (convenience wrapper)."""
    
    def __init__(self, config: ForecastConfig):
        super().__init__(config, model_type='negbin')
