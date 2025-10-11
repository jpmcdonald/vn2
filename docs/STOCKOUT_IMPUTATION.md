# Stockout Imputation: Profile-Based SIP Replacement

## Overview

This module implements **full density function replacement** for stockout-censored demand, going beyond traditional point imputation to reconstruct entire prediction intervals (SIPs).

### The Problem

When inventory stock-outs occur, observed sales are censored:

```
Sales_observed = Min(Demand_true, Stock)
```

This creates three critical issues:
1. **Downward bias** in mean/quantile estimates
2. **Artificially tight prediction intervals** (truncated variance)
3. **Missing tail information** (no data about unsatisfied demand)

Traditional approaches (EM algorithm, Tobit regression) fit parametric models. Our approach **directly reconstructs the predictive distribution** using similar non-stockout profiles.

---

## The "Odd Method": Transform-Aware Tail Splicing

### Key Innovation

Instead of fitting a global censored regression model, we:
1. **Find similar profiles** from non-stockout periods (matching on seasonality, trend, volatility)
2. **Work in transform space** (log/sqrt/cbrt) for variance stabilization
3. **Splice quantile functions**: observed below stock + neighbor tails above
4. **Back-transform with bias correction** (Jensen's inequality)

This preserves the **full uncertainty** needed for Monte Carlo inventory optimization.

---

## Mathematical Framework

### Censored Density

For a stockout week with stock level `s`:

**Observed (censored):**
```
f_obs(x) = { f_D(x)              if x < s
           { P(D ≥ s) · δ(x-s)   if x = s
```

**Goal:** Recover uncensored `f_D(x)` for all x

### Quantile-Based Reconstruction

1. **Below stock**: Use empirical quantiles from observed data
   ```
   Q_obs(p) = quantile(sales[sales < s], p)  for p where Q(p) < s
   ```

2. **Above stock**: Extract tails from neighbor profiles
   ```
   Q_neighbor(p) for p where Q_neighbor(p) ≥ s
   ```

3. **Splice at continuity point**:
   ```
   p* = F(s) ← inverse CDF at stock level
   Q_reconstructed(p) = { Q_obs(p)              if p < p*
                        { aggregate(Q_neighbors(p))  if p ≥ p*
   ```

### Transform-Space Operations

For **log-space** (most common from SURD analysis):

```python
log(Sales_obs) = min(log(D), log(s))  # Additive censoring

# Impute in log-space
log_D_imputed = splice(log_Q_obs, log_Q_neighbors, log(s))

# Back-transform with bias correction
D_imputed = exp(log_D_imputed) × exp(σ²/2)
```

This is more stable than working in original space because multiplicative noise becomes additive.

---

## Implementation

### 1. Profile Matching

Find `k` nearest non-stockout profiles using weighted distance:

```python
from vn2.uncertainty.stockout_imputation import find_neighbor_profiles

neighbors = find_neighbor_profiles(
    target_sku=(store, product),
    target_week=week,
    df=historical_data,
    n_neighbors=20,
    retail_week_window=2  # Seasonal matching
)
```

**Features used for matching:**
- `rolling_mean_4`: Recent demand level
- `rolling_cv_4`: Recent volatility
- `seasonal_mean`: Historical demand for this retail week
- `trend_slope`: Recent trajectory
- `product_group`: Hierarchy match bonus

### 2. Full SIP Imputation

```python
from vn2.uncertainty.stockout_imputation import impute_stockout_sip

q_levels = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 
                     0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

imputed_sip = impute_stockout_sip(
    sku_id=(store, product),
    week=week_with_stockout,
    stock_level=observed_sales,  # = stock for stockouts
    q_levels=q_levels,
    df=historical_data,
    transform_name='log',  # From SURD analysis
    n_neighbors=20
)

# Result: pd.Series indexed by quantile levels
# Ready for Monte Carlo sampling!
```

### 3. Training Data Pipeline

```python
from vn2.forecast.imputation_pipeline import create_imputed_training_data

# Replace stockout observations with imputed demand
df_imputed = create_imputed_training_data(
    df=demand_data,
    surd_transforms=sku_transforms,  # From EDA
    q_levels=q_levels,
    n_neighbors=20
)

# Use df_imputed for model training
# Stockout weeks now have 'imputed' flag = True
# 'sales' column contains median of imputed SIP
```

---

## Usage

### CLI Command

```bash
# After running EDA notebook
source activate.sh
./go impute --config configs/uncertainty.yaml --n-neighbors 20
```

**Output:**
- `data/processed/demand_imputed.parquet` — Training data with imputed values
- `data/processed/imputed_sips.parquet` — Full SIPs for all stockout weeks
- `data/processed/imputation_summary.csv` — Statistics

### Python API

```python
# Single SKU imputation
from vn2.uncertainty.stockout_imputation import impute_stockout_sip

sip = impute_stockout_sip(
    sku_id=(1, 101),
    week=50,
    stock_level=15.0,
    q_levels=quantiles,
    df=data,
    transform_name='log'
)

print(f"Censored sales: 15.0")
print(f"Imputed median: {sip.loc[0.5]:.2f}")
print(f"90% PI: [{sip.loc[0.05]:.2f}, {sip.loc[0.95]:.2f}]")
```

```python
# Bulk imputation
from vn2.uncertainty.stockout_imputation import impute_all_stockouts

imputed_sips = impute_all_stockouts(
    df=data,
    surd_transforms=transforms,
    q_levels=quantiles,
    n_neighbors=20
)

# Returns: Dict[(store, product, week) -> pd.Series]
```

---

## Validation

### 1. Self-Consistency

For non-stockout weeks (stock >> sales):
- Imputation should recover observed demand ✓
- Difference ~ 0% when stock is ample

### 2. Cross-Validation

Artificially censor high-stock weeks:
```python
# Censor at various thresholds
artificial_stock = observed_sales.quantile([0.3, 0.5, 0.7])

# Impute and compare to true uncensored quantiles
coverage = check_quantile_coverage(imputed, true, levels=[0.8, 0.9])
```

### 3. Downstream Performance

Run Monte Carlo inventory optimization with/without imputation:
- Compare realized costs on validation weeks
- Imputed SIPs should reduce shortage costs

---

## Advantages Over Traditional Methods

| Approach | Parametric? | Point/Full? | Transform-Aware? | Cross-SKU? |
|----------|-------------|-------------|------------------|------------|
| EM/Tobit | Yes (Normal/Lognormal) | Point + variance | No | No |
| Censored Quantile Regression | No | Quantiles | Partial | No |
| **Our Method** | **No (nonparametric)** | **Full SIP** | **Yes** | **Yes (profiles)** |

**Key Benefits:**
1. **Preserves full uncertainty** for downstream optimization
2. **No distributional assumptions** (empirical quantiles)
3. **Respects variance stabilization** from SURD
4. **Borrows strength** across similar SKUs
5. **Directly usable** in Monte Carlo simulation

---

## Integration with Forecasting

### Workflow

```
1. EDA → SURD transforms per SKU
2. Impute stockouts → demand_imputed.parquet
3. Train models on imputed data
4. Generate forecast SIPs (h=1..8 weeks)
5. For stockout weeks in history:
   - Use imputed SIPs as "pseudo-observations"
   - Or: flag and down-weight in training
6. Monte Carlo optimization over forecast SIPs
```

### Example: Seasonal ARIMA with Imputation

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load imputed data
df = pd.read_parquet('data/processed/demand_imputed.parquet')

# Train model per SKU
for sku in skus:
    sku_data = df[df['SKU'] == sku].sort_values('week')
    
    # Down-weight imputed observations (optional)
    weights = np.where(sku_data['imputed'], 0.5, 1.0)
    
    # Fit in transform space
    transform = surd_transforms.loc[sku, 'best_transform']
    y = apply_transform(sku_data['sales'], transform)
    
    model = SARIMAX(y, order=(1,0,1), seasonal_order=(1,0,1,52))
    fit = model.fit(cov_type='robust', weights=weights)
    
    # Forecast quantiles
    forecast_sip = generate_sip_from_arima(fit, horizon=8, q_levels)
```

---

## References

### Theory
- **Censored regression**: Tobit (1958), Powell (1984) censored quantile regression
- **Variance stabilization**: Box-Cox, SURD analysis
- **Profile matching**: k-NN regression, local smoothing
- **Jensen's inequality**: Bias correction for nonlinear transforms

### Related Work
- Syntetos & Boylan (2005): Intermittent demand forecasting
- Hyndman et al. (2008): Forecast reconciliation
- Gneiting & Raftery (2007): Strictly proper scoring rules (CRPS)

### Our Innovation
- **Full SIP replacement** vs point imputation
- **Transform-aware splicing** in variance-stabilized space
- **Cross-sectional profile matching** for tail extrapolation
- **Direct integration** with Monte Carlo inventory optimization

---

## Future Enhancements

### Phase 3: Hierarchical Bayesian (Advanced)

```python
# Model stockout probability jointly with demand level
# Borrow strength across product hierarchy

import pymc as pm

with pm.Model() as hierarchical_model:
    # Hierarchy: Department → ProductGroup → SKU
    dept_effect = pm.Normal('dept', 0, 1, shape=n_depts)
    pg_effect = pm.Normal('pg', dept_effect[dept_idx], 0.5, shape=n_pgs)
    
    # Demand level (log-space)
    mu = pm.Normal('mu', pg_effect[pg_idx], sigma_within)
    
    # Stockout probability (logit-space)
    p_stockout = pm.Beta('p_so', alpha, beta)
    
    # Censored likelihood
    obs = pm.Censored('obs', pm.LogNormal.dist(mu, sigma), 
                      lower=0, upper=stock, observed=sales)
```

### Other Ideas
- **Dynamic neighbor weighting** by distance (not just median)
- **Conformal prediction** for calibrated intervals post-imputation
- **Multi-period dependencies** (stock-outs often cluster)
- **Price/promotion interactions** (if data available)

---

## Contact & Contributions

This is a competition-specific implementation. For questions:
- See `notebooks/03_stockout_imputation_demo.ipynb` for examples
- Check `src/vn2/uncertainty/stockout_imputation.py` for implementation
- Review EDA results in `data/processed/surd_transforms.parquet`

**Status**: ✅ Implemented and tested (Phase 2)

**Next**: Integrate with forecasting models and validate via backtest

