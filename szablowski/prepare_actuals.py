"""
Prepare actuals parquet from weekly Sales CSVs for phi calibration.

Reads Week 1..8 Sales CSVs and produces a tidy parquet with columns:
Store, Product, week (int 1-indexed), actual_demand.
"""

import argparse
from pathlib import Path

import pandas as pd


WEEK_FILES = {
    1: "Week 1 - 2024-04-15 - Sales.csv",
    2: "Week 2 - 2024-04-22 - Sales.csv",
    3: "Week 3 - 2024-04-29 - Sales.csv",
    4: "Week 4 - 2024-05-06 - Sales.csv",
    5: "Week 5 - 2024-05-13 - Sales.csv",
    6: "Week 6 - 2024-05-20 - Sales.csv",
    7: "Week 7 - 2024-05-27 - Sales.csv",
    8: "Week 8 - 2024-06-03 - Sales.csv",
}

WEEK_DATE_COLS = {
    1: "2024-04-15",
    2: "2024-04-22",
    3: "2024-04-29",
    4: "2024-05-06",
    5: "2024-05-13",
    6: "2024-05-20",
    7: "2024-05-27",
    8: "2024-06-03",
}


def prepare_actuals(sales_dir: Path, output_path: Path) -> pd.DataFrame:
    rows = []
    for week_num, filename in sorted(WEEK_FILES.items()):
        path = sales_dir / filename
        if not path.exists():
            print(f"  Warning: {path} not found, skipping week {week_num}")
            continue

        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]

        date_col = WEEK_DATE_COLS[week_num]
        if date_col not in df.columns:
            possible = [c for c in df.columns if c not in ("Store", "Product")]
            if possible:
                date_col = possible[0]
            else:
                print(f"  Warning: no sales column found in week {week_num}")
                continue

        for _, row in df.iterrows():
            rows.append({
                "Store": int(row["Store"]),
                "Product": int(row["Product"]),
                "week": week_num,
                "actual_demand": int(row[date_col]),
            })

    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"Saved {len(result)} rows to {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare actuals from weekly Sales CSVs")
    parser.add_argument("--sales-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("models/szablowski/actuals.parquet"))
    args = parser.parse_args()

    prepare_actuals(args.sales_dir, args.output)


if __name__ == "__main__":
    main()
