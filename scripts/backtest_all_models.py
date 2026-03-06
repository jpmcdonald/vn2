#!/usr/bin/env python3
"""
Backtest all trained models across a range of service levels.
Saves per-run weekly results, per-SKU diagnostics, and a summary grid to CSV.

Uses the corrected full_L3_simulation.py with separate eval_costs.
"""

import subprocess
import sys
import re
import csv
from pathlib import Path

MODELS = ['seasonal_naive', 'lightgbm_quantile', 'slurp_bootstrap', 'slurp_stockout_aware', 'deepar']
SERVICE_LEVELS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.833]

OUTPUT_DIR = Path('reports/backtest_grid')

BASE_CMD = [sys.executable, "scripts/full_L3_simulation.py", "--max-weeks", "8"]


def extract_cost(output: str) -> dict:
    holding = shortage = total = None
    for line in output.splitlines():
        m = re.search(r"Total Holding: €([\d,.]+)", line)
        if m:
            holding = float(m.group(1).replace(",", ""))
        m = re.search(r"Total Shortage: €([\d,.]+)", line)
        if m:
            shortage = float(m.group(1).replace(",", ""))
        m = re.search(r"Total Cost: €([\d,.]+)", line)
        if m:
            total = float(m.group(1).replace(",", ""))
    return {"holding": holding, "shortage": shortage, "total": total}


def run_sim(model: str, sl: float) -> dict:
    run_dir = OUTPUT_DIR / f"{model}_sl{sl:.3f}"
    cmd = BASE_CMD + [
        "--default-model", model,
        "--service-level", str(sl),
        "--output-dir", str(run_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-300:]}")
        return {"holding": None, "shortage": None, "total": None}
    return extract_cost(result.stdout + result.stderr)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    print("=" * 80)
    print("Model x Service-Level Backtest Grid  (eval_costs: cu=1.0, co=0.2)")
    print("=" * 80)

    for model in MODELS:
        for sl in SERVICE_LEVELS:
            print(f"\n{model} @ SL={sl:.3f} ...")
            costs = run_sim(model, sl)
            results.append({"model": model, "sl": sl, **costs})
            if costs["total"] is not None:
                print(f"  Holding=€{costs['holding']:.2f}  Shortage=€{costs['shortage']:.2f}  Total=€{costs['total']:.2f}")

    # Save grid CSV
    csv_path = OUTPUT_DIR / "grid_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "sl", "holding", "shortage", "total"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nGrid saved to {csv_path}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Model':<25} {'SL':>6} {'Holding':>10} {'Shortage':>10} {'Total':>10}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x['total'] or 999999):
        if r["total"] is not None:
            print(f"{r['model']:<25} {r['sl']:>6.3f} €{r['holding']:>9,.2f} €{r['shortage']:>9,.2f} €{r['total']:>9,.2f}")
    print("-" * 80)
    print(f"{'Benchmark (ref)':<25} {'':>6} {'':>10} {'':>10} €{'5,247.80':>9}")
    print(f"{'Winner (ref)':<25} {'':>6} {'':>10} {'':>10} €{'4,677.00':>9}")
    print("=" * 80)

    valid = [r for r in results if r["total"] is not None]
    if valid:
        best = min(valid, key=lambda r: r["total"])
        print(f"\nBest: {best['model']} @ SL={best['sl']:.3f}  Total=€{best['total']:,.2f}")

        if best["total"] < 5247.80:
            print("  -> BEATS the benchmark!")
        if best["total"] < 4677.00:
            print("  -> BEATS the winner!")


if __name__ == "__main__":
    main()
