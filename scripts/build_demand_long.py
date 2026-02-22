#!/usr/bin/env python3
"""
Build demand_long.parquet from Week 0 Sales and optional Week 1-8 Sales CSVs.

Reads:
- data/raw/Week 0 - 2024-04-08 - Sales.csv (historical weekly columns)
- data/raw/Week N - YYYY-MM-DD - Sales.csv for N in 1..8 if present

Writes long-format parquet with columns: Store, Product, week (datetime), demand.
Optionally merges Week 0 - In Stock.csv for in_stock column.

Usage:
    python scripts/build_demand_long.py --raw-dir data/raw --out data/processed/demand_long.parquet
    python scripts/build_demand_long.py --raw-dir data/raw --out data/processed/demand_long.parquet --with-in-stock
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Week N -> (filename, date column name) for competition weeks 1-8
WEEK_SALES_FILES = {
    1: ("Week 1 - 2024-04-15 - Sales.csv", "2024-04-15"),
    2: ("Week 2 - 2024-04-22 - Sales.csv", "2024-04-22"),
    3: ("Week 3 - 2024-04-29 - Sales.csv", "2024-04-29"),
    4: ("Week 4 - 2024-05-06 - Sales.csv", "2024-05-06"),
    5: ("Week 5 - 2024-05-13 - Sales.csv", "2024-05-13"),
    6: ("Week 6 - 2024-05-20 - Sales.csv", "2024-05-20"),
    7: ("Week 7 - 2024-05-27 - Sales.csv", "2024-05-27"),
    8: ("Week 8 - 2024-06-03 - Sales.csv", "2024-06-03"),
}

WEEK0_SALES = "Week 0 - 2024-04-08 - Sales.csv"
WEEK0_IN_STOCK = "Week 0 - In Stock.csv"


def build_demand_long(
    raw_dir: Path,
    output_path: Path,
    with_in_stock: bool = False,
) -> None:
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)

    # 1) Week 0 historical sales (wide -> long)
    week0_path = raw_dir / WEEK0_SALES
    if not week0_path.exists():
        raise FileNotFoundError(f"Required file not found: {week0_path}")

    sales0 = pd.read_csv(week0_path)
    sales0.columns = [c.strip() for c in sales0.columns]
    id_cols = ["Store", "Product"]
    if not all(c in sales0.columns for c in id_cols):
        raise ValueError(f"Week 0 Sales must have columns {id_cols}")

    date_columns = [c for c in sales0.columns if c not in id_cols]
    long = sales0.melt(
        id_vars=id_cols,
        value_vars=date_columns,
        var_name="week",
        value_name="demand",
    )
    long["week"] = pd.to_datetime(long["week"])
    long["demand"] = pd.to_numeric(long["demand"], errors="coerce").fillna(0).astype(int)

    # 2) Append Week 1-8 if files exist
    for week_num in sorted(WEEK_SALES_FILES.keys()):
        fname, date_col = WEEK_SALES_FILES[week_num]
        path = raw_dir / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        if date_col not in df.columns:
            continue
        block = df[id_cols + [date_col]].copy()
        block = block.rename(columns={date_col: "demand"})
        block["week"] = pd.to_datetime(date_col)
        block["demand"] = pd.to_numeric(block["demand"], errors="coerce").fillna(0).astype(int)
        long = pd.concat([long, block], ignore_index=True)

    long = long.sort_values(["Store", "Product", "week"]).reset_index(drop=True)

    # 3) Optional: merge in_stock from Week 0 - In Stock.csv
    if with_in_stock:
        in_stock_path = raw_dir / WEEK0_IN_STOCK
        if in_stock_path.exists():
            stock0 = pd.read_csv(in_stock_path)
            stock0.columns = [c.strip() for c in stock0.columns]
            stock_long = stock0.melt(
                id_vars=id_cols,
                value_vars=[c for c in stock0.columns if c not in id_cols],
                var_name="week",
                value_name="in_stock",
            )
            stock_long["week"] = pd.to_datetime(stock_long["week"])
            stock_long["in_stock"] = stock_long["in_stock"].fillna(False).astype(bool)
            long = long.merge(
                stock_long,
                on=id_cols + ["week"],
                how="left",
            )
            long["in_stock"] = long["in_stock"].fillna(True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(output_path, index=False)
    print(f"Wrote {len(long):,} rows to {output_path}")


def main():
    p = argparse.ArgumentParser(description="Build demand_long.parquet from Week 0 + Week 1-8 Sales.")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Raw data directory")
    p.add_argument("--out", type=Path, default=Path("data/processed/demand_long.parquet"), help="Output parquet path")
    p.add_argument("--with-in-stock", action="store_true", help="Merge Week 0 In Stock for in_stock column")
    args = p.parse_args()
    build_demand_long(args.raw_dir, args.out, args.with_in_stock)


if __name__ == "__main__":
    main()
