#!/usr/bin/env python3
"""
Build model selector maps using distributional quality metrics.

Produces:
  1. Static per-SKU selector (same model for all weeks) based on mean composite/CRPS/Wasserstein
  2. Weekly per-SKU selector (potentially different model each week) based on per-week metrics
  3. Comparison backtest of all selectors

Usage:
    uv run python scripts/build_dynamic_selector.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def build_static_selector(
    detail: pd.DataFrame,
    metric: str = 'composite',
    exclude_models: list | None = None,
) -> pd.DataFrame:
    """
    For each SKU, pick the model with lowest mean of `metric` across all weeks.
    Returns DataFrame with (store, product, model_name) columns.
    """
    df = detail.copy()
    if exclude_models:
        df = df[~df['model'].isin(exclude_models)]

    agg = df.groupby(['Store', 'Product', 'model'])[metric].mean().reset_index()
    best = agg.loc[agg.groupby(['Store', 'Product'])[metric].idxmin()]
    result = best[['Store', 'Product', 'model']].rename(columns={
        'Store': 'store', 'Product': 'product', 'model': 'model_name'
    })
    return result.reset_index(drop=True)


def build_weekly_selector(
    detail: pd.DataFrame,
    metric: str = 'composite',
    exclude_models: list | None = None,
) -> pd.DataFrame:
    """
    For each (SKU, week), pick the model with lowest `metric`.
    Returns DataFrame with (store, product, week, model_name) columns.
    """
    df = detail.copy()
    if exclude_models:
        df = df[~df['model'].isin(exclude_models)]

    best_idx = df.groupby(['Store', 'Product', 'week'])[metric].idxmin()
    best = df.loc[best_idx, ['Store', 'Product', 'week', 'model']].rename(columns={
        'Store': 'store', 'Product': 'product', 'model': 'model_name'
    })
    return best.reset_index(drop=True)


def run_backtest(
    label: str,
    sl: float = 0.833,
    selector_path: Path | None = None,
    weekly_selector_path: Path | None = None,
) -> dict | None:
    """Run full_L3_simulation.py with a given selector map and return costs."""
    import re
    output_dir = Path('reports/dynamic_selector') / label
    cmd = [
        sys.executable, 'scripts/full_L3_simulation.py',
        '--max-weeks', '8',
        '--service-level', str(sl),
        '--output-dir', str(output_dir),
    ]
    if selector_path:
        cmd += ['--selector-map', str(selector_path)]
    if weekly_selector_path:
        cmd += ['--weekly-selector-map', str(weekly_selector_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        console.print(f"  [red]FAILED[/red]: {result.stderr[-300:]}")
        return None
    combined = result.stdout + result.stderr
    holding = shortage = total = None
    for line in combined.splitlines():
        m = re.search(r"Total Holding: €([\d,.]+)", line)
        if m:
            holding = float(m.group(1).replace(",", ""))
        m = re.search(r"Total Shortage: €([\d,.]+)", line)
        if m:
            shortage = float(m.group(1).replace(",", ""))
        m = re.search(r"Total Cost: €([\d,.]+)", line)
        if m:
            total = float(m.group(1).replace(",", ""))
    return {'label': label, 'holding': holding, 'shortage': shortage, 'total': total}


def main():
    parser = argparse.ArgumentParser(description='Build dynamic model selectors')
    parser.add_argument('--detail-path', type=Path,
                        default=Path('reports/bias/per_sku_week_detail.parquet'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports/dynamic_selector'))
    parser.add_argument('--sl', type=float, default=0.833)
    parser.add_argument('--skip-backtest', action='store_true')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Loading per-SKU-week detail...[/bold]")
    detail = pd.read_parquet(args.detail_path)
    console.print(f"  {len(detail)} rows, models: {detail['model'].unique().tolist()}")

    # Exclude seasonal_naive from selection (too poor; use others)
    exclude = ['seasonal_naive']

    # --- Build selectors ---
    metrics = ['composite', 'crps', 'wasserstein']
    selectors = {}

    for metric in metrics:
        console.print(f"\n[bold cyan]Static selector (metric={metric})[/bold cyan]")
        sel = build_static_selector(detail, metric=metric, exclude_models=exclude)
        path = args.output_dir / f'static_{metric}_selector.parquet'
        sel.to_parquet(path, index=False)
        console.print(f"  Model distribution:")
        console.print(sel['model_name'].value_counts().to_string())
        selectors[f'static_{metric}'] = path

    for metric in metrics:
        console.print(f"\n[bold cyan]Weekly selector (metric={metric})[/bold cyan]")
        sel = build_weekly_selector(detail, metric=metric, exclude_models=exclude)
        path = args.output_dir / f'weekly_{metric}_selector.parquet'
        sel.to_parquet(path, index=False)
        model_counts = sel['model_name'].value_counts()
        console.print(f"  Model distribution (across all SKU-weeks):")
        console.print(model_counts.to_string())
        selectors[f'weekly_{metric}'] = path

    if args.skip_backtest:
        console.print("\n[yellow]Skipping backtests[/yellow]")
        return

    # --- Run backtests ---
    console.print(f"\n[bold]Running backtests at SL={args.sl}...[/bold]")
    results = []

    # Static selectors
    for label, path in selectors.items():
        if 'weekly' in label:
            continue
        console.print(f"\n  {label}...")
        r = run_backtest(label, sl=args.sl, selector_path=path)
        if r:
            results.append(r)
            console.print(f"  → €{r['total']:,.2f}")

    # Weekly selectors (oracle: uses same-week metrics for selection)
    for label, path in selectors.items():
        if 'weekly' not in label:
            continue
        console.print(f"\n  {label} (oracle)...")
        r = run_backtest(label, sl=args.sl, weekly_selector_path=path)
        if r:
            results.append(r)
            console.print(f"  → €{r['total']:,.2f}")

    # --- Summary ---
    if results:
        rdf = pd.DataFrame(results)
        rdf.to_csv(args.output_dir / 'selector_comparison.csv', index=False)

        console.print("\n" + "=" * 70)
        table = Table(title="Selector Comparison")
        table.add_column("Selector", style="bold")
        table.add_column("Holding", justify="right")
        table.add_column("Shortage", justify="right")
        table.add_column("Total", justify="right", style="bold")
        for r in sorted(results, key=lambda x: x['total'] or 999999):
            table.add_row(
                r['label'],
                f"€{r['holding']:,.2f}" if r['holding'] else "N/A",
                f"€{r['shortage']:,.2f}" if r['shortage'] else "N/A",
                f"€{r['total']:,.2f}" if r['total'] else "N/A",
            )
        console.print(table)
        console.print(f"\nReference: slurp_bootstrap @ SL=0.833 (single model) = €5,169")
        console.print(f"Reference: benchmark = €5,248, winner = €4,677")

    console.print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
