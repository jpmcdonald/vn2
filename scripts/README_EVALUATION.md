# Sequential Backtest Evaluation Scripts

This directory contains scripts for running, analyzing, and validating the 12-week sequential backtest evaluation.

## Scripts Overview

### 1. `run_seq12_eval.sh`
**Purpose:** Run the complete 12-week sequential backtest evaluation

**Usage:**
```bash
./run_seq12_eval.sh
```

**What it does:**
- Activates Python environment
- Runs evaluation across all models and SKUs
- Uses 12 workers for parallel processing
- Saves results with timestamp
- Logs output to `logs/seq12_eval_*.log`

**Output files:**
- `models/results/sequential_results_{timestamp}.parquet`
- `models/results/model_totals_{timestamp}.parquet`
- `models/results/selector_map_{timestamp}.parquet`
- `models/results/leaderboard_{timestamp}.md`

**Runtime:** ~1-2 minutes on 12 cores

---

### 2. `analyze_submitted_order.py`
**Purpose:** Analyze submitted orders with cost expectations and confidence intervals

**Usage:**
```bash
python scripts/analyze_submitted_order.py
```

**What it does:**
- Compares submitted orders vs selector recommendations
- Calculates expected cost at decision time with 5th/95th percentiles
- Optionally calculates realized cost after observing demand
- Provides comprehensive cost analysis with uncertainty quantification

**Output:**
- Console summary with order comparison and cost analysis
- `models/results/order_comparison_detailed.csv`

**Key Questions Answered:**
1. By how much did our order change compared to what was submitted?
2. What was the expected cost when we submitted (with 5th/95th percentiles)?
3. What is the realized cost now that we have Week 1 data?

**Example Output:**
```
Order Statistics:
  Submitted - Mean: 3.85, Median: 1.00, Total: 2309
  Recommended - Mean: 3.85, Median: 1.00, Total: 2309

SKU: Store 2, Product 124
  Expected cost: 3.78
  90% CI: [0.60, 11.00]
```

---

### 3. `compare_evaluation_runs.py`
**Purpose:** Compare portfolio costs between two evaluation runs

**Usage:**
```bash
python scripts/compare_evaluation_runs.py
```

**What it does:**
- Loads previous and current evaluation results
- Compares portfolio costs across models
- Calculates percentage changes
- Validates implementation consistency

**Output:**
- Console summary with comparison table
- `models/results/run_comparison.csv`

**Use case:** Validate that code changes don't break results

---

### 4. `compare_week1_orders.py`
**Purpose:** Compare actual Week 1 submission with selector recommendations

**Usage:**
```bash
python scripts/compare_week1_orders.py
```

**What it does:**
- Loads actual submission from `data/submissions/`
- Extracts selector's Week 1 recommendations
- Computes agreement metrics
- Identifies largest differences

**Output:**
- Console summary with agreement statistics
- `models/results/week1_order_comparison.csv`

**Use case:** Validate selector recommendations against real submission

---

### 5. `generate_evaluation_summary.py`
**Purpose:** Generate comprehensive summary of evaluation results

**Usage:**
```bash
python scripts/generate_evaluation_summary.py
```

**What it does:**
- Loads latest evaluation results
- Prints model rankings
- Analyzes selector performance
- Shows cost breakdowns
- Reports data quality metrics

**Output:**
- Console summary (can redirect to file)

**Use case:** Quick overview of evaluation results

---

## Complete Workflow

### Initial Evaluation

1. **Backup data:**
   ```bash
   mkdir -p backups/pre_eval_$(date +%Y%m%d)
   cp -r data/processed/*.parquet backups/pre_eval_$(date +%Y%m%d)/
   cp -r models/results/*.parquet backups/pre_eval_$(date +%Y%m%d)/
   ```

2. **Run evaluation:**
   ```bash
   ./run_seq12_eval.sh
   ```

3. **Generate summary:**
   ```bash
   python scripts/generate_evaluation_summary.py > EVALUATION_SUMMARY_$(date +%Y%m%d).txt
   ```

### Validation

4. **Compare with previous run:**
   ```bash
   python scripts/compare_evaluation_runs.py
   ```

5. **Validate against actual submission:**
   ```bash
   python scripts/compare_week1_orders.py
   ```

### Analysis

6. **View leaderboard:**
   ```bash
   cat models/results/leaderboard_seq12_*.md | tail -30
   ```

7. **Analyze per-SKU results:**
   ```python
   import pandas as pd
   results = pd.read_parquet('models/results/sequential_results_seq12_*.parquet')
   # Custom analysis...
   ```

---

## Configuration

All scripts use these default paths:
- **Checkpoints:** `models/checkpoints/`
- **Data:** `data/processed/demand_long.parquet`, `data/interim/state.parquet`
- **Output:** `models/results/`
- **Submissions:** `data/submissions/`

To modify, edit the scripts or pass parameters (if supported).

---

## Troubleshooting

### Script fails with "No results found"
**Solution:** Ensure evaluation has been run at least once:
```bash
./run_seq12_eval.sh
```

### Week 1 comparison shows 0% agreement
**Problem:** Using wrong submission file or results file

**Solution:** Check that files exist:
```bash
ls data/submissions/orders_selector_wide_2024-04-15.csv
ls models/results/sequential_results_seq12_*.parquet
```

### Evaluation runs slowly
**Problem:** Not enough workers or large PMF grain

**Solution:** Edit `run_seq12_eval.sh`:
- Increase `--n-jobs` (up to CPU count)
- Decrease `--sip-grain` to 300 (faster but less precise)

### Out of memory
**Problem:** Too many parallel workers

**Solution:** Reduce `--n-jobs` in `run_seq12_eval.sh`

---

## Output Files Reference

### Sequential Results (`sequential_results_*.parquet`)
Per-(model, SKU) detailed results:
- `store`, `product`, `model_name`
- `total_cost`, `total_cost_excl_w1`
- `total_expected_cost`, `total_expected_cost_excl_w1`
- `n_missing`, `n_weeks`
- `orders` (list of 12 integers)
- `costs_by_week` (list of 12 floats)

### Model Totals (`model_totals_*.parquet`)
Aggregated by model:
- `model_name`
- `portfolio_cost` (sum across SKUs)
- `mean_sku_cost`, `std_sku_cost`
- `n_skus`
- `p05_sku`, `p50_sku`, `p95_sku`
- `total_missing`

### Selector Map (`selector_map_*.parquet`)
Per-SKU best model:
- `store`, `product`
- `model_name` (selected model)
- `total_cost` (best cost for this SKU)

### Leaderboard (`leaderboard_*.md`)
Markdown table with model rankings

---

## Reproducibility

All scripts are deterministic and reproducible:
1. Same input data → same results
2. All random seeds are fixed
3. PMF operations are exact (FFT-based)
4. No stochastic components in optimization

To reproduce results:
```bash
# Use exact same data
git checkout <commit-hash>

# Run evaluation
./run_seq12_eval.sh

# Results should match within floating-point precision
```

---

## Dependencies

All scripts require:
- Python 3.10+
- pandas
- numpy
- pyarrow (for parquet files)

Installed via:
```bash
pip install -r requirements.txt
```

---

## Related Documentation

- **Implementation:** `SEQUENTIAL_BACKTEST_SUMMARY.md`
- **Results:** `EVALUATION_RESULTS_SUMMARY.md`
- **Validation:** `COMPARISON_SUMMARY.md`
- **Runbook:** `docs/pipelines/sequential_backtest.md`
- **Tests:** `test/test_sequential_backtest.py`

---

## Maintenance

### Adding New Models

1. Train model and save checkpoints to `models/checkpoints/{model_name}/`
2. Run evaluation: `./run_seq12_eval.sh`
3. New model will be automatically included

### Changing Cost Parameters

Edit `run_seq12_eval.sh`:
```bash
--cu 1.5 \  # Change shortage cost
--co 0.3 \  # Change holding cost
```

### Changing PMF Grain

Edit `run_seq12_eval.sh`:
```bash
--sip-grain 1000 \  # Increase precision (slower)
```

---

## Questions?

See main documentation or open an issue.

