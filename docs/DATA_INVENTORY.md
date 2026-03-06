# VN2 Data Inventory

This document tracks data files required for backtesting and model training, their purpose, and current status in the repo.

**Last validated**: See plan implementation date. Re-run validation when adding Week 1–8 Sales or leaderboard files. When Week 1–8 Sales CSVs are present, run `build_demand_long.py` and use `full_L3_simulation.py --max-weeks 8` and `run_rolling_benchmark_8week.py` for full 8-week backtest.

---

## Data inventory table

| File or pattern | Purpose | Status | Notes |
|-----------------|---------|--------|--------|
| `data/raw/Week 0 - 2024-04-08 - Initial State.csv` | Week 0 inventory and in-transit | Present | Required for all sims |
| `data/raw/Week 0 - 2024-04-08 - Sales.csv` | Historical sales (wide; through 2024-04-08) | Present | Used for training and benchmark |
| `data/raw/Week 0 - In Stock.csv` | Stock availability by week | Present | Used for stockout imputation |
| `data/raw/Week 0 - Master.csv` | Store/product metadata | Present | Optional for enrichment |
| `data/raw/Week 0 - Submission Template.csv` | Submission format | Present | Reference only |
| `data/raw/Week 1 - 2024-04-15 - Sales.csv` | Week 1 actuals | **Missing** | Add when available |
| `data/raw/Week 2 - 2024-04-22 - Sales.csv` | Week 2 actuals | **Missing** | Add when available |
| `data/raw/Week 3 - 2024-04-29 - Sales.csv` | Week 3 actuals | **Missing** | Add when available |
| `data/raw/Week 4 - 2024-05-06 - Sales.csv` | Week 4 actuals | **Missing** | Add when available |
| `data/raw/Week 5 - 2024-05-13 - Sales.csv` | Week 5 actuals | **Missing** | Add when available |
| `data/raw/Week 6 - 2024-05-20 - Sales.csv` | Week 6 actuals | **Missing** | Add when available |
| `data/raw/Week 7 - 2024-05-27 - Sales.csv` | Week 7 actuals | **Missing** | Add when available |
| `data/raw/Week 8 - 2024-06-03 - Sales.csv` | Week 8 actuals | **Missing** | Add when available |
| `data/raw/leaderboards/Week1.txt` … `Week8.txt` | Per-week leaderboard (rank, name, order_cost, cumulative_cost) | Partial | Week3.txt, Week4.txt present |
| `data/raw/leaderboards/cumulative-leaderboard.txt` | Web paste of final/cumulative leaderboard | Present | See Leaderboard format below |
| `data/raw/leaderboards/FinalScore.txt` | Machine-readable final 8-week scores (e.g. name, score, rank) | **Missing** | Optional; use if exported |
| `data/processed/demand_long.parquet` | Long-format demand (Store, Product, week, demand/sales) | **Missing** | Built by script; see Data preparation |
| `data/processed/surd_transforms.parquet` | SURD transform per SKU | Optional | From EDA/transform pipeline |
| `data/states/state*.csv` | Weekly state files | Optional | If you have competition state snapshots |

---

## Leaderboard format

- **Week N.txt**: Parsed by `src/vn2/competition/leaderboard_parser.py`. Expected: lines of rank (digit), blank, competitor name, then tab-separated line with `apply_count`, `order_cost`, `cumulative_cost`, `entries`, `last_seen`, team. Parser uses `Week*.txt` glob.
- **cumulative-leaderboard.txt**: Web paste from competition site. Contains section headers (e.g. "Order #1 Rankings (Order Cost)", "Cumulative") and rows with rank, name, and a tab-separated data line (country, apply, order cost, cumulative cost, entries, last). For final 8-week comparison, use the **Cumulative** section; our entry is "Patrick McDonald", benchmark is "Benchmark Benchmark". A parser for this format is provided in `src/vn2/competition/leaderboard_parser.py` (`parse_cumulative_leaderboard`) for programmatic use.
- **FinalScore.txt**: If available, use for machine-readable final scores (e.g. tab-sep: name, score, rank). Scripts such as `scripts/simulate_L3_costs.py` reference `data/raw/leaderboards/FinalScore.txt` by default.

---

## Data preparation: demand_long.parquet

Many scripts (training, evaluation, backtest) expect `data/processed/demand_long.parquet` with columns: `Store`, `Product`, `week` (datetime or week index), and `demand` (or `sales`).

**How to produce it**

1. **Script (recommended)**: Run `python scripts/build_demand_long.py --raw-dir data/raw --out data/processed/demand_long.parquet`. This script:
   - Reads `Week 0 - 2024-04-08 - Sales.csv` (historical columns).
   - Optionally appends Week 1–8 Sales CSVs if present (exact names as in `scripts/simulate_L3_costs.py`).
   - Melts to long format (Store, Product, week, demand), optionally merges `Week 0 - In Stock.csv` for an `in_stock` column.
   - Writes `data/processed/demand_long.parquet`.

2. **Notebook**: The EDA notebook `notebooks/02_comprehensive_time_series_eda.ipynb` melts Week 0 Sales and related data; it does not write `demand_long.parquet` by default. For a single source of truth, use the script above and document that in this section.

**If the pipeline doesn’t exist**: The plan included creating `scripts/build_demand_long.py`; that script is provided in the repo. After adding Week 1–8 Sales files, run it to regenerate `demand_long.parquet` including competition actuals.
