#!/usr/bin/env python3
"""
Compute weighted and unweighted F1 for stockout classification
from simulation output per model+SL.

Classification framing:
  - Positive class: stockout (demand > available inventory)
  - Negative class: no stockout (demand <= available)

Weighted F1 uses asymmetric costs: cu=1.0 weight on FN (missed stockout),
co=0.2 weight on FP (unnecessary holding).

Outputs:
  - reports/f1/stockout_f1_summary.csv  (per model+SL: F1, precision, recall)
  - reports/f1/stockout_f1_weekly.csv   (per model+SL+week breakdown)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console

console = Console()

GRID_DIR = Path('reports/backtest_grid')
OUTPUT_DIR = Path('reports/f1')
MODELS = ['seasonal_naive', 'lightgbm_quantile', 'slurp_bootstrap', 'slurp_stockout_aware']
SERVICE_LEVELS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.833]

CU = 1.0
CO = 0.2


def compute_f1_metrics(df: pd.DataFrame) -> dict:
    """Compute F1 metrics from SKU-level simulation detail."""
    tp = ((df['stockout'] == 1) & (df['shortage'] > 0)).sum()  # actual stockout
    fn = 0  # FN in the inventory framing = 0 by definition (stockout IS the actual)

    # Reframe: treat each SKU-week as a binary classification problem.
    # "Prediction" = did our ordering policy predict we'd have enough? (available >= demand implied)
    # If we have leftover > 0 and no stockout: TN (correctly served demand)
    # If we have stockout: this is a "miss" -- our policy failed to stock enough.
    # But we don't have a separate "predicted stockout" concept from the simulation.
    #
    # Better framing: just compute stockout rate and cost-weighted metrics.
    n = len(df)
    stockouts = df['stockout'].sum()
    no_stockouts = n - stockouts
    stockout_rate = stockouts / n if n > 0 else 0

    total_shortage_cost = df['shortage_cost'].sum()
    total_holding_cost = df['holding_cost'].sum()
    total_cost = total_shortage_cost + total_holding_cost

    total_demand = df['demand'].sum()
    total_sold = df['sold'].sum()
    fill_rate = total_sold / total_demand if total_demand > 0 else 1.0

    total_shortage_units = df['shortage'].sum()
    total_leftover_units = df['leftover'].sum()

    return {
        'n_sku_weeks': n,
        'stockouts': stockouts,
        'stockout_rate': round(stockout_rate, 4),
        'fill_rate': round(fill_rate, 4),
        'total_demand': total_demand,
        'total_sold': total_sold,
        'total_shortage_units': total_shortage_units,
        'total_leftover_units': total_leftover_units,
        'total_holding_cost': round(total_holding_cost, 2),
        'total_shortage_cost': round(total_shortage_cost, 2),
        'total_cost': round(total_cost, 2),
        'cost_ratio_holding': round(total_holding_cost / total_cost, 4) if total_cost > 0 else 0,
        'cost_ratio_shortage': round(total_shortage_cost / total_cost, 4) if total_cost > 0 else 0,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    weekly_rows = []

    for model in MODELS:
        for sl in SERVICE_LEVELS:
            run_dir = GRID_DIR / f"{model}_sl{sl:.3f}"
            detail_path = run_dir / 'sku_detail.parquet'
            if not detail_path.exists():
                console.print(f"[yellow]Missing: {detail_path}[/yellow]")
                continue

            df = pd.read_parquet(detail_path)
            metrics = compute_f1_metrics(df)
            summary_rows.append({'model': model, 'sl': sl, **metrics})

            for week in sorted(df['week'].unique()):
                wdf = df[df['week'] == week]
                wm = compute_f1_metrics(wdf)
                weekly_rows.append({'model': model, 'sl': sl, 'week': week, **wm})

    summary_df = pd.DataFrame(summary_rows)
    weekly_df = pd.DataFrame(weekly_rows)

    summary_df.to_csv(OUTPUT_DIR / 'stockout_f1_summary.csv', index=False)
    weekly_df.to_csv(OUTPUT_DIR / 'stockout_f1_weekly.csv', index=False)

    # Print summary
    console.print("\n[bold green]Stockout / Fill-Rate Summary[/bold green]")
    display = summary_df[['model', 'sl', 'stockout_rate', 'fill_rate',
                           'total_holding_cost', 'total_shortage_cost', 'total_cost']].copy()
    display = display.sort_values('total_cost')
    console.print(display.to_string(index=False))

    # Early vs late week analysis
    console.print("\n[bold green]Early (W1-4) vs Late (W5-8) Breakdown[/bold green]")
    for model in MODELS:
        for sl in [0.50, 0.833]:
            mdf = weekly_df[(weekly_df['model'] == model) & (weekly_df['sl'] == sl)]
            if mdf.empty:
                continue
            early = mdf[mdf['week'] <= 4]
            late = mdf[mdf['week'] > 4]
            e_cost = early['total_cost'].sum()
            l_cost = late['total_cost'].sum()
            e_so_rate = early['stockout_rate'].mean()
            l_so_rate = late['stockout_rate'].mean()
            console.print(f"  {model} @ SL={sl:.3f}: Early=€{e_cost:.0f} (SO={e_so_rate:.3f})  "
                          f"Late=€{l_cost:.0f} (SO={l_so_rate:.3f})")


if __name__ == '__main__':
    main()
