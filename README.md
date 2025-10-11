# VN2 Inventory Planning Challenge

Supply chain optimization using SIP/SLURP uncertainty quantification and Monte Carlo optimization to close Jensen's gap.

## Overview

This repository implements a novel approach to inventory planning that:
- **Quantifies uncertainty** using SIP (Stochastic Information Packets) and SLURP (Stochastic Library Units with Relationships Preserved)
- **Optimizes under uncertainty** via Monte Carlo simulation over full predictive distributions
- **Closes Jensen's gap** by optimizing expected cost directly rather than collapsing to moments
- **Imputes stock-outs** nonparametrically using quantile functions

## Quick Start

```bash
# Setup environment
./activate.sh

# Run EDA notebook first
jupyter notebook notebooks/02_comprehensive_time_series_eda.ipynb

# Impute stockout-censored demand
./go impute --config configs/uncertainty.yaml --n-neighbors 20

# Ingest data
./go ingest --raw data/raw --out data/interim

# Run base-stock simulation
./go simulate --config configs/base.yaml --out data/submissions/W1.csv

# Run Monte Carlo optimization
./go optimize-mc --config configs/uncertainty.yaml --out data/submissions/W1_mc.csv

# Analyze specific SKU
./go analyze --raw data/raw --store 60 --product 126

# Demo SIP/SLURP
./go slurp-demo --out scratch/demo.xml
```

## Project Structure

```
vn2/
├── src/vn2/          # Core modules
│   ├── sim/          # Simulation engine (2-week lead time, costs)
│   ├── policy/       # Inventory policies (base-stock, MC optimization)
│   ├── uncertainty/  # SIP/SLURP, quantile I/O, stock-out imputation
│   ├── data/         # Loaders and validation
│   ├── submit/       # Submission builder
│   ├── analyze/      # EDA and segmentation
│   └── cli.py        # Command-line interface
├── configs/          # YAML configuration
├── data/
│   ├── raw/          # Competition data (immutable)
│   ├── interim/      # Cleaned artifacts
│   ├── processed/    # Forecasts and quantiles
│   └── submissions/  # Competition submissions
├── notebooks/        # Jupyter notebooks for exploration
├── test/             # Unit tests
├── go                # CLI wrapper
└── activate.sh       # Environment setup
```

## Key Concepts

### SIP/SLURP
Translated from R implementations in `docs/SIPS_SLURPS/`. A SLURP preserves correlations across variables by sampling full scenarios (rows) rather than independent marginals.

### Jensen's Gap
Traditional methods optimize E[f(X)] ≈ f(E[X]) using point forecasts. Our approach computes E[cost] via Monte Carlo over the full predictive distribution, preserving nonlinear interactions between uncertainty and costs.

### Stock-out Imputation
Profile-based **full SIP replacement** for censored demand. Instead of point imputation, we reconstruct entire quantile functions by splicing observed data below stock with neighbor tails above. Works in variance-stabilized transform space (log/sqrt/cbrt) for stability. See [docs/STOCKOUT_IMPUTATION.md](docs/STOCKOUT_IMPUTATION.md).

### SURD Analysis
Systematic Unsupervised Representation Discovery for variance stabilization and feature importance. Discovers optimal transforms per SKU to tighten prediction intervals. Reference: [ALD-Lab/SURD](https://github.com/ALD-Lab/SURD)

## Competition Details

- **Goal**: Minimize holding + shortage costs over 6 ordering weeks (W1-W6) + 2 delivery weeks (W7-W8)
- **Lead time**: 2 weeks (order at end of W, receive start of W+3)
- **Costs**: Holding 0.2€/unit/week, shortage 1.0€/unit
- **Constraints**: 599 SKUs, no backorders, no waste
- **Timeline**: Oct 21 - Nov 5, 2025

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest test/

# Format code
black src/ test/
ruff check src/ test/
```

## CLI Commands

- `go ingest` - Ingest and clean raw data
- `go impute` - **Impute stockout-censored demand** (run after EDA)
- `go simulate` - Run simulation with base-stock policy
- `go optimize-mc` - Monte Carlo optimization over SIP samples
- `go submit` - Build submission file
- `go analyze` - Ad-hoc SKU analysis
- `go segment` - ABC/XYZ segmentation
- `go slurp-demo` - Demonstrate SIP/SLURP functionality

## References

- Competition: [VN2 Inventory Planning Challenge](docs/Instructions.md)
- SURD: [ALD-Lab/SURD](https://github.com/ALD-Lab/SURD) - Synergistic-Unique-Redundant Decomposition of causality
- SIP/SLURP: See `docs/SIPS_SLURPS/` for original R implementations

## License

MIT

