#!/usr/bin/env python3
"""
Compare train-once (static folds) vs sequential-refit (fold per order) strategies.

Runs slurp_bootstrap at the best service level (0.833) under both strategies,
then extends to all models for the most interesting SLs.
"""

import subprocess
import sys
import re
import csv
from pathlib import Path

OUTPUT_DIR = Path('reports/train_strategy')
BASE_CMD = [sys.executable, "scripts/full_L3_simulation.py", "--max-weeks", "8"]

PILOT_MODEL = 'slurp_bootstrap'
PILOT_SL = 0.833

ALL_MODELS = ['seasonal_naive', 'lightgbm_quantile', 'slurp_bootstrap', 'slurp_stockout_aware', 'deepar']
COMPARE_SLS = [0.50, 0.70, 0.833]


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


def run_sim(model: str, sl: float, static_folds: bool) -> dict:
    strategy = "static" if static_folds else "sequential"
    run_dir = OUTPUT_DIR / f"{model}_sl{sl:.3f}_{strategy}"
    cmd = BASE_CMD + [
        "--default-model", model,
        "--service-level", str(sl),
        "--output-dir", str(run_dir),
    ]
    if static_folds:
        cmd.append("--static-folds")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-300:]}")
        return {"holding": None, "shortage": None, "total": None}
    return extract_cost(result.stdout + result.stderr)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    print("=" * 90)
    print("Train-Once (static fold_0) vs Sequential-Refit (fold per order)")
    print("=" * 90)

    # Full comparison
    for model in ALL_MODELS:
        for sl in COMPARE_SLS:
            for static in [True, False]:
                strategy = "static" if static else "sequential"
                print(f"\n{model} @ SL={sl:.3f}  [{strategy}] ...")
                costs = run_sim(model, sl, static)
                results.append({
                    "model": model, "sl": sl, "strategy": strategy, **costs
                })
                if costs["total"] is not None:
                    print(f"  Holding=€{costs['holding']:.2f}  "
                          f"Shortage=€{costs['shortage']:.2f}  "
                          f"Total=€{costs['total']:.2f}")

    # Save CSV
    csv_path = OUTPUT_DIR / "strategy_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "sl", "strategy", "holding", "shortage", "total"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {csv_path}")

    # Delta table
    print("\n" + "=" * 90)
    print(f"{'Model':<25} {'SL':>6} {'Static':>10} {'Sequential':>10} {'Delta':>10} {'Winner':>12}")
    print("-" * 90)

    for model in ALL_MODELS:
        for sl in COMPARE_SLS:
            static_r = next((r for r in results if r['model'] == model
                             and r['sl'] == sl and r['strategy'] == 'static'), None)
            seq_r = next((r for r in results if r['model'] == model
                          and r['sl'] == sl and r['strategy'] == 'sequential'), None)
            if static_r and seq_r and static_r['total'] and seq_r['total']:
                delta = seq_r['total'] - static_r['total']
                winner = "static" if delta >= 0 else "sequential"
                print(f"{model:<25} {sl:>6.3f} €{static_r['total']:>9,.2f} €{seq_r['total']:>9,.2f} "
                      f"€{delta:>+9,.2f} {winner:>12}")
    print("-" * 90)
    print("Positive delta = static is cheaper; Negative delta = sequential is cheaper")
    print("=" * 90)


if __name__ == "__main__":
    main()
