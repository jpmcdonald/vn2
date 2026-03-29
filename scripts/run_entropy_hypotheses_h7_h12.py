#!/usr/bin/env python3
"""
Run summary analyses for entropy hypotheses H7–H12 from metrics parquet.

Reads output of scripts/compute_entropy_regime_metrics.py (or eval_folds with entropy columns).
Writes CSV + Markdown under reports/entropy_hypotheses/.

Usage:
  uv run python scripts/run_entropy_hypotheses_h7_h12.py \\
    --metrics models/results/entropy_regime_metrics.parquet \\
    --eval-folds models/results/eval_folds.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument(
        "--eval-folds",
        type=Path,
        default=None,
        help="Optional eval_folds parquet to merge sip_realized_cost_w2 / composite",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("reports/entropy_hypotheses"))
    args = ap.parse_args()

    _ensure_dir(args.out_dir)
    m = pd.read_parquet(args.metrics)

    if args.eval_folds and args.eval_folds.exists():
        ev = pd.read_parquet(args.eval_folds)
        merge_cols = ["model_name", "store", "product", "fold_idx"]
        extra = [c for c in ("sip_realized_cost_w2", "composite", "pinball_loss") if c in ev.columns]
        if extra:
            m = m.merge(ev[merge_cols + extra], on=merge_cols, how="left")

    lines = ["# Entropy hypotheses H7–H12 (automated summary)\n", f"Source: `{args.metrics}`\n"]

    # --- H9: entropy gap (demand) vs Jensen gap ---
    h9_cols = ["entropy_gap_demand_h2", "jensen_gap_w2"]
    if all(c in m.columns for c in h9_cols):
        sub = m[h9_cols].dropna()
        if len(sub) > 5:
            rho = sub["entropy_gap_demand_h2"].corr(sub["jensen_gap_w2"], method="spearman")
            h9 = pd.DataFrame([{"hypothesis": "H9", "spearman_rho": rho, "n": len(sub)}])
            h9.to_csv(args.out_dir / "h9_entropy_gap_vs_jensen.csv", index=False)
            lines.append(f"## H9: Entropy gap (Gaussian ref − H) vs Jensen gap\n")
            lines.append(f"Spearman ρ = **{rho:.4f}**, n = {len(sub)}\n")
    else:
        lines.append("## H9: skipped (missing columns)\n")

    # --- H7 / H8: outcome vs demand entropy; Jensen summary ---
    if "H_outcome_w2" in m.columns and "H_demand_h2" in m.columns:
        sub = m[["H_outcome_w2", "H_demand_h2"]].dropna()
        if len(sub) > 2:
            rho_oo_dd = sub["H_outcome_w2"].corr(sub["H_demand_h2"], method="spearman")
            h7 = pd.DataFrame(
                [{"metric": "spearman_H_outcome_vs_H_demand", "value": rho_oo_dd, "n": len(sub)}]
            )
            h7.to_csv(args.out_dir / "h7_h8_entropy_correlation.csv", index=False)
            lines.append("## H7 / H8: Demand vs outcome entropy\n")
            lines.append(f"Spearman(H_outcome, H_demand) = **{rho_oo_dd:.4f}**\n")

    if "jensen_gap_w2" in m.columns:
        j = m["jensen_gap_w2"].dropna()
        if len(j) > 0:
            h8 = pd.DataFrame(
                [
                    {
                        "hypothesis": "H8",
                        "mean_jensen_gap": float(j.mean()),
                        "std_jensen_gap": float(j.std()),
                        "n": len(j),
                    }
                ]
            )
            h8.to_csv(args.out_dir / "h8_jensen_gap_summary.csv", index=False)
            lines.append("## H8: Jensen gap (joint − point plug-in)\n")
            lines.append(f"Mean = **{j.mean():.4f}**, std = **{j.std():.4f}**, n = {len(j)}\n")

    # --- H10: sensitivity by regime ---
    if "sensitivity_ratio" in m.columns and "entropy_regime" in m.columns:
        sub = m[np.isfinite(m["sensitivity_ratio"])]
        if len(sub) > 0:
            g = sub.groupby("entropy_regime")["sensitivity_ratio"].agg(["mean", "std", "count"])
            g.to_csv(args.out_dir / "h10_sensitivity_by_regime.csv")
            lines.append("## H10: Sensitivity ratio by entropy_regime\n")
            lines.append("```\n" + g.to_string() + "\n```\n")

    # --- H11: scalar proxies vs outcome entropy for explaining cost ---
    if "sip_realized_cost_w2" in m.columns:
        cost = m["sip_realized_cost_w2"]
        rows = []
        for name, col in [
            ("H_outcome_w2", "H_outcome_w2"),
            ("H_demand_h2", "H_demand_h2"),
            ("composite", "composite"),
        ]:
            if col in m.columns:
                pair = pd.DataFrame({"c": cost, "x": m[col]}).dropna()
                if len(pair) > 10:
                    rows.append(
                        {
                            "signal": name,
                            "spearman_vs_sip_cost": pair["x"].corr(pair["c"], method="spearman"),
                            "n": len(pair),
                        }
                    )
        if rows:
            h11 = pd.DataFrame(rows)
            h11.to_csv(args.out_dir / "h11_signal_vs_cost_correlation.csv", index=False)
            lines.append("## H11: |Spearman| with sip_realized_cost_w2 (higher magnitude ⇒ stronger linear rank link)\n")
            lines.append("```\n" + h11.to_string(index=False) + "\n```\n")

    # --- H12: KL(emp || model) vs cost ---
    if "kl_empirical_vs_model_h2" in m.columns and "sip_realized_cost_w2" in m.columns:
        pair = m[["kl_empirical_vs_model_h2", "sip_realized_cost_w2"]].dropna()
        if len(pair) > 10:
            rho = pair["kl_empirical_vs_model_h2"].corr(pair["sip_realized_cost_w2"], method="spearman")
            h12 = pd.DataFrame([{"spearman_kl_vs_cost": rho, "n": len(pair)}])
            h12.to_csv(args.out_dir / "h12_kl_vs_realized_cost.csv", index=False)
            lines.append("## H12: KL divergence proxy vs realized SIP cost\n")
            lines.append(
                f"Spearman(KL(emp||model), sip_realized_cost_w2) = **{rho:.4f}** "
                f"(interpret cautiously; selector backtest is separate).\n"
            )

    summary_path = args.out_dir / "summary_h7_h12.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
