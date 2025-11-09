#!/usr/bin/env python3
"""Revenue and profit analysis across state snapshots.

Assumptions:
- Each unit sells for €2.00
- Gross margin per unit is €1.00 (50%)
- Holding costs reduce net profit but not revenue or gross profit
- Shortage costs represent direct penalties that reduce net profit

The script scans state CSV files (state1.csv, state2.csv, ...) and computes
weekly and cumulative metrics including revenue, gross profit, and net profit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

DEFAULT_PRICE_PER_UNIT = 2.0
DEFAULT_MARGIN_PER_UNIT = 1.0
STATE_FILE_PATTERN = "state{}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute revenue/gross/net profit metrics from state files"
    )
    parser.add_argument(
        "--states-dir",
        type=Path,
        default=Path("data/states"),
        help="Directory containing stateN.csv files",
    )
    parser.add_argument(
        "--price",
        type=float,
        default=DEFAULT_PRICE_PER_UNIT,
        help="Selling price per unit (default: 2.0)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN_PER_UNIT,
        help="Gross margin per unit (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the weekly summary as CSV",
    )
    return parser.parse_args()


def discover_state_files(states_dir: Path) -> Dict[int, Path]:
    """Find state files named stateN.csv and return mapping of week -> path."""
    mapping: Dict[int, Path] = {}
    if not states_dir.exists():
        return mapping

    for path in sorted(states_dir.glob("state*.csv")):
        name = path.stem  # e.g., "state4"
        suffix = name.replace("state", "")
        if not suffix.isdigit():
            continue
        week = int(suffix)
        mapping[week] = path
    return dict(sorted(mapping.items()))


def compute_week_metrics(
    df: pd.DataFrame,
    week: int,
    price_per_unit: float,
    margin_per_unit: float,
) -> Dict[str, float]:
    sales_units = df["Sales"].sum()
    missed_units = df["Missed Sales"].sum()
    holding_cost = df.get("Holding Cost", pd.Series(dtype=float)).sum()
    shortage_cost = df.get("Shortage Cost", pd.Series(dtype=float)).sum()

    revenue = sales_units * price_per_unit
    potential_revenue = (sales_units + missed_units) * price_per_unit
    lost_revenue = potential_revenue - revenue

    gross_profit = sales_units * margin_per_unit
    potential_gross = (sales_units + missed_units) * margin_per_unit
    lost_gross = potential_gross - gross_profit

    total_costs = holding_cost + shortage_cost
    net_profit = gross_profit - total_costs

    cost_to_revenue = total_costs / revenue if revenue else float("nan")
    cost_to_gross = total_costs / gross_profit if gross_profit else float("nan")

    return {
        "week": week,
        "sales_units": sales_units,
        "missed_units": missed_units,
        "revenue": revenue,
        "potential_revenue": potential_revenue,
        "lost_revenue": lost_revenue,
        "gross_profit": gross_profit,
        "potential_gross": potential_gross,
        "lost_gross": lost_gross,
        "holding_cost": holding_cost,
        "shortage_cost": shortage_cost,
        "total_costs": total_costs,
        "net_profit": net_profit,
        "cost_to_revenue": cost_to_revenue,
        "cost_to_gross": cost_to_gross,
    }


def format_currency(value: float) -> str:
    return f"€{value:,.2f}"


def format_ratio(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def main() -> int:
    args = parse_args()
    state_files = discover_state_files(args.states_dir)

    if not state_files:
        print(f"No state files found in {args.states_dir}")
        return 1

    weekly_records: List[Dict[str, float]] = []
    for week, path in state_files.items():
        df = pd.read_csv(path)
        metrics = compute_week_metrics(df, week, args.price, args.margin)
        weekly_records.append(metrics)

    weekly_df = pd.DataFrame(weekly_records)
    weekly_df.set_index("week", inplace=True)

    totals = weekly_df.sum(numeric_only=True)
    totals.name = "Total"

    # Print weekly summary
    print("=== Weekly Revenue & Profit Summary ===")
    display_cols = [
        "revenue",
        "gross_profit",
        "net_profit",
        "holding_cost",
        "shortage_cost",
        "total_costs",
        "cost_to_revenue",
        "cost_to_gross",
    ]

    printable = weekly_df[display_cols].copy()
    printable["cost_to_revenue"] = printable["cost_to_revenue"].apply(format_ratio)
    printable["cost_to_gross"] = printable["cost_to_gross"].apply(format_ratio)
    printable["revenue"] = printable["revenue"].apply(format_currency)
    printable["gross_profit"] = printable["gross_profit"].apply(format_currency)
    printable["net_profit"] = printable["net_profit"].apply(format_currency)
    printable["holding_cost"] = printable["holding_cost"].apply(format_currency)
    printable["shortage_cost"] = printable["shortage_cost"].apply(format_currency)
    printable["total_costs"] = printable["total_costs"].apply(format_currency)

    print(printable.to_string())

    # Totals and comparisons
    total_revenue = weekly_df["revenue"].sum()
    total_gross = weekly_df["gross_profit"].sum()
    total_net = weekly_df["net_profit"].sum()
    total_costs = weekly_df["total_costs"].sum()

    print("\n=== Totals ===")
    print(f"Revenue: {format_currency(total_revenue)}")
    print(f"Gross Profit: {format_currency(total_gross)}")
    print(f"Net Profit: {format_currency(total_net)}")
    print(f"Total Costs (holding + shortage): {format_currency(total_costs)}")

    if total_revenue:
        print(
            f"Cost share vs revenue: {total_costs / total_revenue * 100:.2f}%"
        )
    if total_gross:
        print(
            f"Cost share vs gross profit: {total_costs / total_gross * 100:.2f}%"
        )

    print("\n=== Lost Opportunity ===")
    total_potential_rev = weekly_df["potential_revenue"].sum()
    total_lost_rev = weekly_df["lost_revenue"].sum()
    total_potential_gross = weekly_df["potential_gross"].sum()
    total_lost_gross = weekly_df["lost_gross"].sum()
    print(f"Potential Revenue (no stockouts): {format_currency(total_potential_rev)}")
    print(f"Lost Revenue from stockouts: {format_currency(total_lost_rev)}")
    print(f"Potential Gross Profit: {format_currency(total_potential_gross)}")
    print(f"Lost Gross Profit: {format_currency(total_lost_gross)}")

    if args.output:
        weekly_df.to_csv(args.output)
        print(f"\nSaved weekly summary to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
