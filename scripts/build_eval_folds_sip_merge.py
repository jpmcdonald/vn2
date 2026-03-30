#!/usr/bin/env python3
"""
Build a minimal eval_folds-style parquet for merging into entropy hypotheses (H11/H12).

Computes only SIP week-2 realized cost and pinball loss per (model, store, product, fold_idx),
using the same state panel + checkpoints as compute_entropy_regime_metrics.py — without
Monte Carlo inventory simulation or shape metrics (much faster than full model_eval).

Usage:
  uv run python scripts/build_eval_folds_sip_merge.py \\
    --checkpoint-dir models/checkpoints \\
    --demand-path data/processed/demand_imputed.parquet \\
    --state-path data/processed/state_panel.parquet \\
    --output models/results/eval_folds_sip_merge.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vn2.analyze.model_eval import (  # noqa: E402
    EvalTask,
    compute_sip_cost_metric,
    is_degenerate_forecast,
    load_checkpoint,
)
from vn2.analyze.state_resolve import load_state_parquet, resolve_sip_state_row  # noqa: E402
from vn2.forecast.evaluation import average_pinball_loss  # noqa: E402
from vn2.forecast.features import prepare_train_test_split  # noqa: E402
from vn2.sim import Costs  # noqa: E402


def _one_row(
    task: EvalTask,
    df: pd.DataFrame,
    master_df: pd.DataFrame | None,
    checkpoint_dir: Path,
    holdout_weeks: int,
    state_df,
    costs: Costs,
    sip_grain: int,
) -> dict | None:
    path = checkpoint_dir / task.model_name / f"{task.store}_{task.product}" / f"fold_{task.fold_idx}.pkl"
    ck = load_checkpoint(path)
    if ck is None:
        return None
    qdf = ck.get("quantiles")
    if qdf is None or qdf.empty or is_degenerate_forecast(qdf):
        return None
    try:
        y_train, _, y_test, _ = prepare_train_test_split(
            df,
            (task.store, task.product),
            holdout_weeks=holdout_weeks,
            fold_idx=task.fold_idx,
            master_df=master_df,
        )
    except Exception:
        return None
    if y_train is None or y_test is None or len(y_test) < 2:
        return None
    y_true = y_test.values.astype(float).ravel()
    i0, q1, q2 = resolve_sip_state_row(state_df, task.store, task.product, task.fold_idx)
    initial_state = pd.DataFrame(
        {"on_hand": [i0], "intransit_1": [q1], "intransit_2": [q2]},
        index=[(task.store, task.product)],
    )
    sip = compute_sip_cost_metric(
        y_true, qdf, initial_state, costs, sip_grain, exclude_week1=True
    )
    if sip.get("sip_realized_cost_w2") is None:
        return None
    pinball = float(average_pinball_loss(y_true, qdf))
    return {
        "model_name": task.model_name,
        "store": task.store,
        "product": task.product,
        "fold_idx": task.fold_idx,
        "sip_realized_cost_w2": sip["sip_realized_cost_w2"],
        "pinball_loss": pinball,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Minimal SIP columns for entropy H11/H12 merge")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("models/checkpoints"))
    p.add_argument("--demand-path", type=Path, default=Path("data/processed/demand_imputed.parquet"))
    p.add_argument("--master-path", type=Path, default=Path("data/processed/master.parquet"))
    p.add_argument(
        "--state-path",
        type=Path,
        default=Path("data/processed/state_panel.parquet"),
    )
    p.add_argument("--output", type=Path, default=Path("models/results/eval_folds_sip_merge.parquet"))
    p.add_argument("--holdout", type=int, default=8)
    p.add_argument("--holding", type=float, default=0.2)
    p.add_argument("--shortage", type=float, default=1.0)
    p.add_argument("--sip-grain", type=int, default=1000)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    df = pd.read_parquet(args.demand_path)
    master_df = pd.read_parquet(args.master_path) if args.master_path.exists() else None
    state_df = load_state_parquet(args.state_path) if args.state_path.exists() else None
    if state_df is None:
        print("ERROR: state parquet missing or empty; need fold-specific state for SIP cost.")
        sys.exit(1)

    costs = Costs(holding=args.holding, shortage=args.shortage)
    tasks: list[EvalTask] = []
    for model_dir in sorted(args.checkpoint_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for sku_dir in model_dir.iterdir():
            if not sku_dir.is_dir():
                continue
            try:
                store, product = map(int, sku_dir.name.split("_"))
            except ValueError:
                continue
            for fold_idx in range(args.holdout):
                tasks.append(EvalTask(model_dir.name, store, product, fold_idx))
    if args.limit > 0:
        tasks = tasks[: args.limit]
    print(f"Tasks: {len(tasks)}")

    rows = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(_one_row)(t, df, master_df, args.checkpoint_dir, args.holdout, state_df, costs, args.sip_grain)
        for t in tasks
    )
    valid = [r for r in rows if r is not None]
    out = pd.DataFrame(valid)
    if out.empty:
        print("No rows computed.")
        sys.exit(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
