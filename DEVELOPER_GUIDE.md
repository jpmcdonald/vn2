# Developer Guide: VN2 Inventory Planning Challenge

Welcome! This guide will help you understand the codebase structure and where to begin.

## 📋 Table of Contents

1. [What is This Project?](#what-is-this-project)
2. [Project Overview](#project-overview)
3. [Where to Start](#where-to-start)
4. [Core Architecture](#core-architecture)
5. [Key Concepts](#key-concepts)
6. [Directory Structure](#directory-structure)
7. [Development Workflow](#development-workflow)
8. [Key Files to Understand](#key-files-to-understand)

---

## What is This Project?

**VN2** is a supply chain inventory optimization competition solution. The goal is to minimize total cost (holding + shortage) by optimally ordering inventory for 599 SKUs across multiple stores over 6 ordering weeks.

### Competition Rules
- **Lead Time**: 2 weeks (order at end of week X, receive at start of week X+3)
- **Costs**: 0.2€/unit/week holding cost, 1.0€/unit shortage cost
- **Objective**: Minimize total cost over 8 weeks (6 ordering + 2 delivery weeks)
- **Data**: 599 SKUs with ~3 years of historical weekly sales and stock data

---

## Project Overview

This codebase implements a sophisticated inventory planning system that:

1. **Quantifies Uncertainty**: Uses SIP (Stochastic Information Packets) and SLURP to model full predictive distributions
2. **Optimizes Under Uncertainty**: Monte Carlo simulation over full distributions (not just point forecasts)
3. **Closes Jensen's Gap**: Optimizes expected cost directly rather than using E[f(X)] ≈ f(E[X])
4. **Handles Stockouts**: Nonparametric imputation of stockout-censored demand using quantile functions

---

## Where to Start

### Step 1: Read the README
Start with `README.md` - it provides the high-level overview and quick start commands.

### Step 2: Understand the Problem
Read `docs/Instructions.md` to understand the competition rules and requirements.

### Step 3: Explore the Data
Run the EDA notebook to understand the data structure:
```bash
jupyter notebook notebooks/02_comprehensive_time_series_eda.ipynb
```

### Step 4: Understand the Core Flow
Follow this learning path:

1. **Simulation Engine** → `src/vn2/sim/core.py`
   - How the inventory system works
   - Lead time mechanics
   - Cost calculations

2. **Data Loading** → `src/vn2/data/loaders.py`
   - How data is ingested
   - State representation
   - Index structure

3. **Forecasting** → `src/vn2/forecast/`
   - Multiple forecasting models
   - Quantile predictions
   - Uncertainty quantification

4. **Policy/Optimization** → `src/vn2/policy/`
   - Base-stock policy
   - Monte Carlo optimization
   - How orders are determined

5. **CLI Interface** → `src/vn2/cli.py`
   - How everything is orchestrated
   - Command structure

### Step 5: Run a Simple Example
```bash
# Activate environment
./activate.sh

# Ingest data
./go ingest --raw data/raw --out data/interim

# Run a simple simulation
./go simulate --config configs/base.yaml --out data/submissions/test.csv
```

---

## Core Architecture

```
┌─────────────┐
│   Raw Data  │ (Competition data)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Ingest    │ (Clean & validate)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Forecast   │ (Generate predictions + uncertainty)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Optimize   │ (MC optimization over SIP/SLURP)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Simulate   │ (Test policy)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Submit     │ (Generate submission CSV)
└─────────────┘
```

---

## Key Concepts

### 1. SIP/SLURP (Stochastic Information Packets)
- **SIP**: Represents uncertainty as a collection of scenarios (quantiles)
- **SLURP**: Preserves correlations by sampling full scenarios (rows) rather than independent marginals
- Located in: `src/vn2/uncertainty/sip_slurp.py`
- Documentation: `docs/SIPS_SLURPS/`

### 2. Jensen's Gap
Traditional methods approximate:
```
E[cost(demand)] ≈ cost(E[demand])
```

This project computes the true expected cost:
```
E[cost(demand)] = ∫ cost(x) p(x) dx
```
via Monte Carlo simulation over the full predictive distribution.

### 3. Stockout Imputation
When inventory runs out, observed sales are censored (underestimate true demand).
- Profile-based full SIP replacement
- Reconstructs quantile functions by splicing observed data with neighbor tails
- Works in variance-stabilized transform space
- See: `docs/STOCKOUT_IMPUTATION.md` and `src/vn2/uncertainty/stockout_imputation.py`

### 4. SURD Analysis
Systematic Unsupervised Representation Discovery for variance stabilization.
- Discovers optimal transforms per SKU
- Tightens prediction intervals
- See: `docs/SURD/`

---

## Directory Structure

```
vn2/
├── src/vn2/              # Main source code
│   ├── cli.py            # ⭐ Command-line interface (start here)
│   ├── sim/              # Simulation engine
│   │   └── core.py       # ⭐ Core simulator (understand this first)
│   ├── data/             # Data loading and validation
│   │   └── loaders.py    # ⭐ Data ingestion
│   ├── forecast/         # Forecasting models
│   │   ├── pipeline.py   # Forecast orchestration
│   │   └── models/       # Individual forecast models
│   ├── policy/           # Inventory policies
│   │   ├── base_stock.py # Base-stock policy
│   │   └── bruteforce_mc.py # MC optimization
│   ├── uncertainty/      # Uncertainty quantification
│   │   ├── sip_slurp.py  # SIP/SLURP implementation
│   │   └── stockout_imputation.py # Stockout handling
│   ├── analyze/          # Analysis and evaluation
│   ├── backtest/         # Backtesting strategies
│   ├── submit/           # Submission file generation
│   └── utils/            # Utilities
│
├── configs/              # YAML configuration files
│   ├── base.yaml         # ⭐ Basic configuration
│   ├── uncertainty.yaml  # Uncertainty-based config
│   └── ...
│
├── data/
│   ├── raw/              # Competition data (immutable)
│   ├── interim/          # Cleaned data
│   ├── processed/        # Forecasts and quantiles
│   └── submissions/      # Generated submission files
│
├── notebooks/            # Jupyter notebooks
│   ├── 01_getting_started.ipynb
│   └── 02_comprehensive_time_series_eda.ipynb  # ⭐ Start here
│
├── docs/                 # Documentation
│   ├── Instructions.md   # Competition rules
│   ├── STOCKOUT_IMPUTATION.md
│   └── ...
│
├── reports/              # Generated reports and analysis
├── test/                 # Unit tests
├── scripts/              # Utility scripts
├── go                    # CLI wrapper script
└── activate.sh           # Environment setup
```

---

## Development Workflow

### 1. Setup Environment
```bash
./activate.sh
```

### 2. Understand the Data
```bash
# Run EDA notebook
jupyter notebook notebooks/02_comprehensive_time_series_eda.ipynb
```

### 3. Run Stockout Imputation
```bash
./go impute --config configs/uncertainty.yaml --n-neighbors 20
```

### 4. Ingest Data
```bash
./go ingest --raw data/raw --out data/interim
```

### 5. Generate Forecasts
Forecasts are typically generated via the forecast pipeline (see `src/vn2/forecast/pipeline.py`)

### 6. Optimize Orders
```bash
./go optimize-mc --config configs/uncertainty.yaml --out data/submissions/W1_mc.csv
```

### 7. Analyze Results
```bash
./go analyze --raw data/raw --store 60 --product 126
```

---

## Key Files to Understand

### Essential Files (Read First)

1. **`README.md`** - Project overview and quick start
2. **`src/vn2/sim/core.py`** - Simulation engine (inventory dynamics)
3. **`src/vn2/data/loaders.py`** - Data loading and state management
4. **`src/vn2/cli.py`** - Command-line interface (orchestration)
5. **`configs/base.yaml`** - Configuration structure

### Important Modules

6. **`src/vn2/forecast/pipeline.py`** - Forecast orchestration
7. **`src/vn2/policy/base_stock.py`** - Base-stock policy
8. **`src/vn2/policy/bruteforce_mc.py`** - Monte Carlo optimization
9. **`src/vn2/uncertainty/sip_slurp.py`** - SIP/SLURP implementation
10. **`src/vn2/uncertainty/stockout_imputation.py`** - Stockout handling

### Documentation Files

- **`docs/Instructions.md`** - Competition rules
- **`docs/STOCKOUT_IMPUTATION.md`** - Stockout imputation methodology
- **`docs/SURD/SURD_Paper.md`** - SURD analysis details

---

## Common Tasks

### Adding a New Forecasting Model
1. Create model class in `src/vn2/forecast/models/`
2. Inherit from base forecaster interface
3. Register in forecast pipeline
4. See examples: `lightgbm_quantile.py`, `ets.py`, etc.

### Modifying the Optimization Policy
1. Edit `src/vn2/policy/bruteforce_mc.py` or create new policy
2. Update CLI command in `src/vn2/cli.py`
3. Test with `./go optimize-mc`

### Running Backtests
See `src/vn2/analyze/sequential_backtest.py` and `scripts/run_strategy_backtest.py`

### Evaluating Models
See `src/vn2/analyze/model_eval.py` and related evaluation modules

---

## Testing

```bash
# Run all tests
pytest test/

# Run specific test
pytest test/test_sim.py

# Run with coverage
pytest test/ --cov=src/vn2
```

---

## Code Style

```bash
# Format code
black src/ test/

# Lint code
ruff check src/ test/
```

---

## Getting Help

1. Check existing documentation in `docs/`
2. Review markdown files in root directory (e.g., `IMPLEMENTATION_SUMMARY.md`)
3. Look at example notebooks in `notebooks/`
4. Review recent reports in `reports/` and `models/results/`

---

## Next Steps

1. ✅ Read this guide
2. ✅ Run the EDA notebook
3. ✅ Understand the simulator (`src/vn2/sim/core.py`)
4. ✅ Run a simple simulation
5. ✅ Explore the forecast models
6. ✅ Understand the optimization policy
7. ✅ Review the uncertainty quantification approach

Welcome to the team! 🚀

