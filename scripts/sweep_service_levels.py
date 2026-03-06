#!/usr/bin/env python3
"""
Sweep flat service levels and schedules through the L3 simulation.
Collects total 8-week cost for each configuration.

Usage:
    python scripts/sweep_service_levels.py
"""

import subprocess
import sys
import json
import re

FLAT_LEVELS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.833]

SCHEDULES = {
    "A_ramp": {1: 0.60, 2: 0.65, 3: 0.70, 4: 0.75, 5: 0.80, 6: 0.83},
    "B_conservative": {1: 0.50, 2: 0.60, 3: 0.70, 4: 0.75, 5: 0.80, 6: 0.83},
    "C_low_flat_ramp": {1: 0.50, 2: 0.55, 3: 0.60, 4: 0.65, 5: 0.70, 6: 0.75},
}

BASE_CMD = [
    sys.executable, "scripts/full_L3_simulation.py",
    "--max-weeks", "8",
]


def extract_cost(output: str) -> dict:
    """Extract holding, shortage, total from simulation output."""
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


def run_sim(extra_args: list) -> dict:
    cmd = BASE_CMD + extra_args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-200:]}")
        return {"holding": None, "shortage": None, "total": None}
    return extract_cost(result.stdout)


def main():
    results = []

    print("=" * 70)
    print("Service-Level Sweep (seasonal_naive h=3)")
    print("=" * 70)

    # Flat levels
    for sl in FLAT_LEVELS:
        print(f"\nRunning flat SL={sl:.3f} ...")
        costs = run_sim(["--service-level", str(sl)])
        results.append({"config": f"flat_{sl:.3f}", "sl": sl, **costs})
        print(f"  Holding=€{costs['holding']:.2f}  Shortage=€{costs['shortage']:.2f}  Total=€{costs['total']:.2f}")

    # Schedules
    for name, sched in SCHEDULES.items():
        sched_json = json.dumps({str(k): v for k, v in sched.items()})
        print(f"\nRunning schedule {name}: {sched} ...")
        costs = run_sim(["--service-level-schedule", sched_json])
        results.append({"config": f"sched_{name}", "sl": "schedule", **costs})
        print(f"  Holding=€{costs['holding']:.2f}  Shortage=€{costs['shortage']:.2f}  Total=€{costs['total']:.2f}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Config':<25} {'Holding':>10} {'Shortage':>10} {'Total':>10}")
    print("-" * 70)
    for r in results:
        if r["total"] is not None:
            print(f"{r['config']:<25} €{r['holding']:>9,.2f} €{r['shortage']:>9,.2f} €{r['total']:>9,.2f}")
    print("-" * 70)
    print(f"{'Benchmark (ref)':<25} {'':>10} {'':>10} €{'5,247.80':>9}")
    print(f"{'Winner (ref)':<25} {'':>10} {'':>10} €{'4,677.00':>9}")
    print("=" * 70)

    # Find best
    valid = [r for r in results if r["total"] is not None]
    if valid:
        best = min(valid, key=lambda r: r["total"])
        print(f"\nBest: {best['config']}  Total=€{best['total']:,.2f}")


if __name__ == "__main__":
    main()
