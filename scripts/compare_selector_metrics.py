#!/usr/bin/env python3
"""
Compare backtest outcome: composite (pinball x Wasserstein) vs cost-based selector.

Builds a cost-based selector from eval_folds, runs full_L3_simulation with both
selector maps under identical settings, and writes a comparison summary.

Usage:
    uv run python scripts/compare_selector_metrics.py
    uv run python scripts/compare_selector_metrics.py --eval-folds path/to/eval_folds.parquet
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from vn2.analyze.model_selector import select_per_sku_from_folds


def run_backtest(
    selector_path: Path,
    output_dir: Path,
    service_level: float = 0.833,
    checkpoints_dir: Path = Path("models/checkpoints_h3"),
    timeout: int = 1800,
) -> dict | None:
    """Run full_L3_simulation.py with the given selector map; return holding, shortage, total."""
    cmd = [
        sys.executable,
        "scripts/full_L3_simulation.py",
        "--max-weeks",
        "8",
        "--service-level",
        str(service_level),
        "--checkpoints-dir",
        str(checkpoints_dir),
        "--selector-map",
        str(selector_path),
        "--output-dir",
        str(output_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=Path(__file__).resolve().parent.parent)
    if result.returncode != 0:
        print(f"Backtest FAILED: {result.stderr[-500:]}")
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
    return {"holding": holding, "shortage": shortage, "total": total}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare composite vs cost-based selector backtest outcomes"
    )
    parser.add_argument(
        "--eval-folds",
        type=Path,
        default=Path("models/results/eval_folds_v4_sip.parquet"),
        help="Path to eval_folds parquet with sip_realized_cost_w2",
    )
    parser.add_argument(
        "--composite-selector",
        type=Path,
        default=Path("reports/dynamic_selector/static_composite_selector.parquet"),
        help="Path to composite (pinball x Wasserstein) selector map",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/selector_comparison"),
        help="Directory for cost-based map, backtest outputs, and summary",
    )
    parser.add_argument(
        "--service-level",
        type=float,
        default=0.833,
        help="Service level for backtest (default 0.833)",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("models/checkpoints_h3"),
        help="Checkpoints dir for full_L3_simulation",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Only build cost-based selector and skip running backtests",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout in seconds per backtest (default 1800 = 30 min). Increase if full_L3_simulation is slow.",
    )
    parser.add_argument(
        "--no-parallel-backtests",
        action="store_true",
        help="Run the two backtests sequentially instead of in parallel",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    args.output_dir = root / args.output_dir
    args.eval_folds = root / args.eval_folds
    args.composite_selector = root / args.composite_selector
    args.checkpoints_dir = root / args.checkpoints_dir

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Ensure inputs exist ---
    if not args.eval_folds.exists():
        print(f"ERROR: eval_folds not found: {args.eval_folds}")
        print("Run the eval pipeline that produces eval_folds (e.g. eval_folds_v4_sip.parquet) first.")
        sys.exit(1)
    df_folds = pd.read_parquet(args.eval_folds)
    if "sip_realized_cost_w2" not in df_folds.columns:
        print(f"ERROR: eval_folds missing column 'sip_realized_cost_w2'. Available: {list(df_folds.columns)}")
        sys.exit(1)

    if not args.composite_selector.exists():
        print(f"WARNING: composite selector not found: {args.composite_selector}")
        print("Run compute_bias_analysis.py then build_dynamic_selector.py to create it.")
        print("Will run only cost-based backtest.")
        composite_map_exists = False
    else:
        composite_map_exists = True

    # --- 2. Build cost-based selector map (backtest format: store, product, model_name) ---
    cost_based_path = args.output_dir / "cost_based_selector.parquet"
    print("Building cost-based selector from eval_folds...")
    top1 = select_per_sku_from_folds(
        args.eval_folds,
        cost_col="sip_realized_cost_w2",
        fold_window=8,
        output_path=None,
    )
    # full_L3_simulation expects column model_name
    backtest_map = top1[["store", "product", "selected_model"]].rename(
        columns={"selected_model": "model_name"}
    )
    backtest_map.to_parquet(cost_based_path, index=False)
    print(f"Saved backtest-format map: {cost_based_path} ({len(backtest_map)} SKUs)")

    if args.skip_backtest:
        print("Skipping backtests (--skip-backtest).")
        return

    # --- 3. Run same backtest for both selectors (optionally in parallel) ---
    results = {}
    timeout_sec = getattr(args, "timeout", 1800)

    def _run_composite():
        out_composite = args.output_dir / "backtest_composite"
        return run_backtest(
            args.composite_selector,
            output_dir=out_composite,
            service_level=args.service_level,
            checkpoints_dir=args.checkpoints_dir,
            timeout=timeout_sec,
        )

    def _run_cost_based():
        out_cost = args.output_dir / "backtest_cost_based"
        return run_backtest(
            cost_based_path,
            output_dir=out_cost,
            service_level=args.service_level,
            checkpoints_dir=args.checkpoints_dir,
            timeout=timeout_sec,
        )

    if args.no_parallel_backtests:
        if composite_map_exists:
            print("\nRunning backtest: composite selector...")
            results["composite"] = _run_composite()
            if results.get("composite"):
                print(f"  Total Cost: €{results['composite']['total']:.2f}")
        print("\nRunning backtest: cost-based selector...")
        results["cost_based"] = _run_cost_based()
        if results.get("cost_based"):
            print(f"  Total Cost: €{results['cost_based']['total']:.2f}")
    else:
        print(f"\nRunning both backtests in parallel (timeout={timeout_sec}s each)...")
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = {}
            if composite_map_exists:
                futures["composite"] = ex.submit(_run_composite)
            futures["cost_based"] = ex.submit(_run_cost_based)
            for name, fut in futures.items():
                try:
                    results[name] = fut.result()
                    if results[name]:
                        print(f"  {name}: Total Cost €{results[name]['total']:.2f}")
                except Exception as e:
                    print(f"  {name}: FAILED - {e}")
                    results[name] = None

    # --- 4. Compare and report ---
    composite_df = pd.read_parquet(args.composite_selector) if args.composite_selector.exists() else None
    cost_counts = backtest_map["model_name"].value_counts().sort_index()

    summary_path = args.output_dir / "comparison_summary.md"
    with open(summary_path, "w") as f:
        f.write("# Selector comparison: composite vs cost-based\n\n")
        f.write("Same backtest (full_L3_simulation, 8 weeks, service-level=0.833).\n\n")
        f.write("## Results\n\n")
        f.write("| Selector       | Holding   | Shortage  | Total     |\n")
        f.write("|----------------|----------:|----------:|----------:|\n")
        if composite_map_exists and results.get("composite"):
            r = results["composite"]
            f.write(f"| Composite      | €{r['holding']:,.2f} | €{r['shortage']:,.2f} | €{r['total']:,.2f} |\n")
        else:
            f.write("| Composite      | —         | —         | —         |\n")
        if results.get("cost_based"):
            r = results["cost_based"]
            f.write(f"| Cost-based     | €{r['holding']:,.2f} | €{r['shortage']:,.2f} | €{r['total']:,.2f} |\n")
        else:
            f.write("| Cost-based     | —         | —         | —         |\n")
        if composite_map_exists and results.get("composite") and results.get("cost_based"):
            diff = results["cost_based"]["total"] - results["composite"]["total"]
            f.write(f"\n**Difference (cost_based − composite):** €{diff:,.2f}\n")
        f.write("\n## Model selection counts (cost-based selector)\n\n")
        f.write("| Model | SKUs |\n")
        f.write("|-------|-----:|\n")
        for model, count in cost_counts.items():
            f.write(f"| {model} | {count} |\n")
        if composite_df is not None and "model_name" in composite_df.columns:
            comp_counts = composite_df["model_name"].value_counts().sort_index()
            f.write("\n## Model selection counts (composite selector)\n\n")
            f.write("| Model | SKUs |\n")
            f.write("|-------|-----:|\n")
            for model, count in comp_counts.items():
                f.write(f"| {model} | {count} |\n")
        f.write("\n---\n")
        f.write("Note: Cost-based selection minimizes the same 8-week realized cost that the backtest reports; composite uses pinball×Wasserstein and may generalize differently out-of-sample.\n")

    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
