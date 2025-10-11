## Optimization plan

Objective: choose order-up-to or service-level decisions using SIP-based demand uncertainty, validated via simulation.

### Inputs

- Density forecasts / SIP samples per SKU
- Costs, lead/review weeks from config

### Approaches

- Base-stock grid search over upto levels or service levels
- Monte Carlo evaluation using SIP samples (existing pipeline)
- Segment-aware caps to bound grid size

### Metrics

- Total cost (holding + shortage)
- Service level / fill rate
- Sensitivity to lead time and cost parameters

### Checkpoint/resume

- Unit of work = SKU (Store, Product)
- Persist per-SKU frontier to `artifacts/opt/frontiers.parquet`
- Manifest `artifacts/opt/manifest.parquet` with SKU status and runtime
- Resume by skipping completed SKUs

### Validation

- Simulate candidate policies on backtest horizon; retain Pareto frontier


