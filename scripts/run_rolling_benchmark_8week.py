#!/usr/bin/env python3
"""
Rolling 8-week benchmark: Official Benchmark methodology with no leakage.

At each decision week t (0..5), uses only sales through week t (Week 0 + Weeks 1..t)
to compute seasonal MA forecast and order-up-to order. Simulates weeks 1..8 with
actual demand; L=3 lead time. Outputs total 8-week cost for comparison to our pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

INDEX = ["Store", "Product"]

# Week number -> (filename, date column) for competition weeks 1-8
WEEK_FILES = {
    1: ("Week 1 - 2024-04-15 - Sales.csv", "2024-04-15"),
    2: ("Week 2 - 2024-04-22 - Sales.csv", "2024-04-22"),
    3: ("Week 3 - 2024-04-29 - Sales.csv", "2024-04-29"),
    4: ("Week 4 - 2024-05-06 - Sales.csv", "2024-05-06"),
    5: ("Week 5 - 2024-05-13 - Sales.csv", "2024-05-13"),
    6: ("Week 6 - 2024-05-20 - Sales.csv", "2024-05-20"),
    7: ("Week 7 - 2024-05-27 - Sales.csv", "2024-05-27"),
    8: ("Week 8 - 2024-06-03 - Sales.csv", "2024-06-03"),
}


def load_week0(raw_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Week 0 Sales, In Stock, Initial State."""
    sales = pd.read_csv(raw_dir / "Week 0 - 2024-04-08 - Sales.csv").set_index(INDEX)
    in_stock = pd.read_csv(raw_dir / "Week 0 - In Stock.csv").set_index(INDEX)
    state = pd.read_csv(raw_dir / "Week 0 - 2024-04-08 - Initial State.csv").set_index(INDEX)
    sales.columns = pd.to_datetime(sales.columns)
    in_stock.columns = pd.to_datetime(in_stock.columns)
    sales[~in_stock] = np.nan
    return sales, in_stock, state


def load_week_sales(raw_dir: Path, week: int) -> pd.DataFrame:
    """Load one week's sales as DataFrame with single date column; index Store, Product."""
    if week not in WEEK_FILES:
        return pd.DataFrame()
    fname, date_col = WEEK_FILES[week]
    path = raw_dir / fname
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.set_index(INDEX)[[date_col]].copy()
    df.columns = [pd.Timestamp(date_col)]
    return df


def build_history_through_week(raw_dir: Path, through_week: int) -> pd.DataFrame:
    """Build sales DataFrame with columns from Week 0 through through_week (no future)."""
    sales0, in_stock0, _ = load_week0(raw_dir)
    out = sales0.copy()
    for w in range(1, through_week + 1):
        sw = load_week_sales(raw_dir, w)
        if sw.empty:
            break
        for c in sw.columns:
            out[c] = sw[c]
    return out


def benchmark_forecast_and_order(
    sales_history: pd.DataFrame,
    state: pd.DataFrame,
    in_stock: pd.DataFrame,
) -> pd.Series:
    """
    Seasonal MA + order-up-to 4 weeks. Uses only sales_history (no future).
    Returns order qty per (Store, Product).
    """
    if sales_history.empty or sales_history.shape[1] < 2:
        return pd.Series(0.0, index=sales_history.index)
    sales = sales_history.copy()
    sales.columns = pd.to_datetime(sales.columns)
    in_stock_cols = in_stock.reindex(columns=sales.columns, fill_value=True)
    sales[~in_stock_cols] = np.nan

    # Seasonal factors: mean per calendar week (across SKUs), then normalize
    season = sales.mean(axis=0).rename("Demand").to_frame()
    season["WeekNumber"] = season.index.isocalendar().week
    season = season.groupby("WeekNumber")["Demand"].mean()
    season = season / season.mean()

    # De-seasonalize
    sales_weeks = sales.columns.isocalendar().week
    sales_no_season = sales / season.loc[sales_weeks.values].values.reshape(1, -1)

    # Last 13 weeks (or all if fewer)
    n_use = min(13, sales_no_season.shape[1])
    base_forecast = sales_no_season.iloc[:, -n_use:].mean(axis=1)

    # Next 4 weeks: use last date + 1,2,3,4 weeks
    last_date = sales.columns[-1]
    f_periods = pd.date_range(start=last_date, periods=5, inclusive="neither", freq="W-MON")[:4]
    if len(f_periods) == 0:
        return pd.Series(0.0, index=sales.index)
    forecast = base_forecast.to_frame().assign(**{str(d): base_forecast for d in f_periods})
    forecast = forecast[[str(d) for d in f_periods]]
    f_weeks = f_periods.isocalendar().week
    forecast = forecast * season.reindex(f_weeks).values.reshape(1, -1)

    order_up_to = forecast.sum(axis=1)
    net_inventory = state["In Transit W+1"].fillna(0) + state["In Transit W+2"].fillna(0) + state["End Inventory"].fillna(0)
    order = (order_up_to - net_inventory).clip(lower=0).round(0).astype(int)
    return order


def run_rolling_benchmark_8week(
    raw_dir: Path,
    cu: float = 1.0,
    co: float = 0.2,
) -> Tuple[float, pd.DataFrame]:
    """
    Run rolling benchmark: 6 orders (end of weeks 0..5), simulate 8 weeks with actuals.
    Returns (total_8week_cost, per_week_dataframe).
    """
    raw_dir = Path(raw_dir)
    sales0, in_stock0, state_df = load_week0(raw_dir)

    # State: index (Store, Product), on_hand, pipeline = [arrives_this_week, arrives_next, in_2_weeks]
    on_hand = state_df["End Inventory"].fillna(0).astype(int)
    it1 = state_df["In Transit W+1"].fillna(0).astype(int)
    it2 = state_df["In Transit W+2"].fillna(0).astype(int)
    pipeline = [it1, it2, 0]

    # Place Order 1 at end of week 0 (history = Week 0 only). Order 1 arrives start of week 3.
    state0 = state_df.copy()
    state0["End Inventory"] = on_hand
    state0["In Transit W+1"] = it1
    state0["In Transit W+2"] = it2
    history0 = build_history_through_week(raw_dir, 0)
    order1 = benchmark_forecast_and_order(history0, state0, in_stock0)
    order1 = order1.reindex(on_hand.index, fill_value=0).astype(int)
    pipeline[2] = order1

    results = []
    for week in range(1, 9):
        # Load actual demand for this week
        demand_df = load_week_sales(raw_dir, week)
        if demand_df.empty:
            demand_series = pd.Series(0, index=on_hand.index)
        else:
            demand_series = demand_df.iloc[:, 0].reindex(on_hand.index, fill_value=0).astype(int)

        # Arrivals at start of week
        arriving = pipeline[0]
        available = on_hand + arriving

        # Costs
        leftover = (available - demand_series).clip(lower=0)
        shortage = (demand_series - available).clip(lower=0)
        holding_cost = (co * leftover).sum()
        shortage_cost = (cu * shortage).sum()
        total_cost = holding_cost + shortage_cost

        # Update state
        on_hand = leftover
        pipeline = [pipeline[1], pipeline[2], 0]

        # Place next order at END of week (Order 2 after week 1, ..., Order 6 after week 5)
        if week <= 5:
            history = build_history_through_week(raw_dir, week)
            state_now = pd.DataFrame({
                "In Transit W+1": pipeline[0],
                "In Transit W+2": pipeline[1],
                "End Inventory": on_hand,
            }, index=on_hand.index)
            order = benchmark_forecast_and_order(history, state_now, in_stock0)
            order = order.reindex(on_hand.index, fill_value=0).astype(int)
            pipeline[2] = order

        results.append({
            "week": week,
            "holding_cost": holding_cost,
            "shortage_cost": shortage_cost,
            "total_cost": total_cost,
        })

    df = pd.DataFrame(results)
    total_8week = df["total_cost"].sum()
    return total_8week, df


def main():
    p = argparse.ArgumentParser(description="Rolling 8-week benchmark (no leakage).")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--cu", type=float, default=1.0)
    p.add_argument("--co", type=float, default=0.2)
    p.add_argument("--output", type=Path, default=None, help="Optional: write per-week CSV")
    args = p.parse_args()

    total, per_week = run_rolling_benchmark_8week(args.raw_dir, cu=args.cu, co=args.co)
    print(f"Benchmark 8-week total cost: €{total:.2f}")
    print(per_week.to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        per_week.to_csv(args.output, index=False)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
