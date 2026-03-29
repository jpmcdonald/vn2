#!/usr/bin/env python3
"""
Add entropy regime labels + optional EDA join (Task 4).

Reads a metrics parquet (e.g. from compute_entropy_regime_metrics.py without labels),
applies add_sensitivity_ratio_column / add_entropy_regime_labels, optionally merges
EDA artifacts via szablowski.harness.join_eda_artifacts.

Usage:
  uv run python scripts/compute_entropy_regime_labels.py \\
    --input models/results/entropy_regime_metrics.parquet \\
    --output models/results/entropy_regime_with_eda.parquet \\
    --eda-processed-dir data/processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vn2.analyze.entropy_regime import (  # noqa: E402
    add_entropy_regime_labels,
    add_sensitivity_ratio_column,
)
from szablowski.harness import join_eda_artifacts  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--eda-processed-dir",
        type=Path,
        default=None,
        help="If set, merge summary_statistics / stationarity parquets from this directory",
    )
    args = p.parse_args()

    df = pd.read_parquet(args.input)
    need = {"store", "product", "model_name", "fold_idx", "H_outcome_w2", "H_demand_h2"}
    if not need.issubset(df.columns):
        raise SystemExit(f"Input must contain columns {need}; run compute_entropy_regime_metrics first.")

    if "sensitivity_ratio" not in df.columns:
        df = add_sensitivity_ratio_column(df)
    if "entropy_regime" not in df.columns:
        df = add_entropy_regime_labels(df)

    if args.eda_processed_dir and args.eda_processed_dir.exists():
        eda_df = df.rename(columns={"store": "Store", "product": "Product"})
        eda_df = join_eda_artifacts(eda_df, args.eda_processed_dir)
        df = eda_df.rename(columns={"Store": "store", "Product": "product"})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
