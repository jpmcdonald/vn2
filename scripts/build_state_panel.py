#!/usr/bin/env python3
"""
Build state_panel.parquet: SIP (on_hand, intransit_1, intransit_2) at each rolling
fold origin, by replaying L=3 dynamics from Initial State with **zero** historical
orders (see module docstring in vn2.data.state_panel).

Usage:
  uv run python scripts/build_state_panel.py \\
    --demand-path data/processed/demand_imputed.parquet \\
    --raw-dir data/raw \\
    --output data/processed/state_panel.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vn2.data.state_panel import build_state_panel, state_panel_to_parquet


def main() -> None:
    p = argparse.ArgumentParser(description="Build (store, product, fold_idx) state panel")
    p.add_argument(
        "--demand-path",
        type=Path,
        default=Path("data/processed/demand_imputed.parquet"),
    )
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/state_panel.parquet"),
    )
    p.add_argument("--holdout", type=int, default=8)
    p.add_argument("--min-train-rows", type=int, default=52)
    args = p.parse_args()

    demand_df = pd.read_parquet(args.demand_path)
    panel = build_state_panel(
        demand_df,
        args.raw_dir,
        holdout_weeks=args.holdout,
        min_train_rows=args.min_train_rows,
    )
    state_panel_to_parquet(panel, args.output)
    print(f"Wrote {len(panel)} rows to {args.output}")


if __name__ == "__main__":
    main()
