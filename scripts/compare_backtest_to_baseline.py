#!/usr/bin/env python3
"""
Compare current backtest grid results to a baseline run.

Loads baseline and current grid_results.csv, reports:
- Reproducibility: for (model, sl) in both, baseline vs new total cost and delta.
- New results: rows only in current run (e.g. SURD models) with cost and vs best baseline at that SL.

Usage:
  python scripts/compare_backtest_to_baseline.py
  python scripts/compare_backtest_to_baseline.py --baseline reports/backtest_grid_baseline_20260305 --current reports/backtest_grid
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Compare backtest grid to baseline")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("reports/backtest_grid_baseline_20260305"),
        help="Baseline run directory containing grid_results.csv",
    )
    parser.add_argument(
        "--current-dir",
        type=Path,
        default=Path("reports/backtest_grid"),
        help="Current run directory containing grid_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write comparison artifacts here (default: current-dir)",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.current_dir

    baseline_csv = args.baseline_dir / "grid_results.csv"
    current_csv = args.current_dir / "grid_results.csv"

    if not baseline_csv.exists():
        print(f"Baseline not found: {baseline_csv}")
        return 1
    if not current_csv.exists():
        print(f"Current grid not found: {current_csv}")
        return 1

    base = pd.read_csv(baseline_csv)
    cur = pd.read_csv(current_csv)

    base = base.rename(columns={"total": "baseline_total", "holding": "baseline_holding", "shortage": "baseline_shortage"})
    cur = cur.rename(columns={"total": "current_total", "holding": "current_holding", "shortage": "current_shortage"})

    merged = cur.merge(
        base[["model", "sl", "baseline_total", "baseline_holding", "baseline_shortage"]],
        on=["model", "sl"],
        how="outer",
    )

    # Reproducibility: both have the row
    both = merged.loc[merged["baseline_total"].notna() & merged["current_total"].notna()].copy()
    both["delta_total"] = both["current_total"] - both["baseline_total"]
    both["repro"] = both["delta_total"].abs() < 0.01

    # New only (e.g. SURD models)
    new_only = merged.loc[merged["baseline_total"].isna() & merged["current_total"].notna()].copy()

    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV: full comparison
    comparison_rows = []
    for _, row in both.iterrows():
        comparison_rows.append({
            "model": row["model"],
            "sl": row["sl"],
            "baseline_total": row["baseline_total"],
            "current_total": row["current_total"],
            "delta_total": row["delta_total"],
            "reproducible": row["repro"],
        })
    for _, row in new_only.iterrows():
        comparison_rows.append({
            "model": row["model"],
            "sl": row["sl"],
            "baseline_total": None,
            "current_total": row["current_total"],
            "delta_total": None,
            "reproducible": None,
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv = output_dir / "comparison_to_baseline.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"Wrote {comparison_csv}")

    # Markdown summary
    md_lines = [
        "# Backtest Grid: Comparison to Baseline",
        "",
        f"Baseline: `{args.baseline_dir}`",
        f"Current:  `{args.current_dir}`",
        "",
        "## Reproducibility (models in both runs)",
        "",
    ]
    if not both.empty:
        n_repro = both["repro"].sum()
        n_both = len(both)
        md_lines.append(f"Same (model, sl) rows: {n_both}. Within €0.01: {n_repro}.")
        md_lines.append("")
        md_lines.append("| Model | SL | Baseline | Current | Delta |")
        md_lines.append("|-------|-----|--------:|--------:|------:|")
        for _, row in both.sort_values(["model", "sl"]).iterrows():
            md_lines.append(
                f"| {row['model']} | {row['sl']:.3f} | €{row['baseline_total']:,.2f} | €{row['current_total']:,.2f} | €{row['delta_total']:+,.2f} |"
            )
    else:
        md_lines.append("No overlapping (model, sl) rows.")
    md_lines.append("")
    md_lines.append("## New results (current run only)")
    md_lines.append("")
    if not new_only.empty:
        # Best baseline per SL for reference
        best_baseline = base.groupby("sl")["baseline_total"].min().to_dict()
        md_lines.append("| Model | SL | Current | Best baseline at this SL |")
        md_lines.append("|-------|-----|--------:|--------------------------|")
        for _, row in new_only.sort_values(["model", "sl"]).iterrows():
            best_at_sl = best_baseline.get(row["sl"], float("nan"))
            md_lines.append(
                f"| {row['model']} | {row['sl']:.3f} | €{row['current_total']:,.2f} | €{best_at_sl:,.2f} |"
            )
    else:
        md_lines.append("No rows only in current run.")
    md_lines.append("")

    comparison_md = output_dir / "comparison_to_baseline.md"
    comparison_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {comparison_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
