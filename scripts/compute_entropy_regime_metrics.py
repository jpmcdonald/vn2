#!/usr/bin/env python3
"""
Batch-compute entropy / outcome-PMF metrics per (model, SKU, fold).

Writes a parquet suitable for Task 3 time series and H7–H12 analyses.
Can merge with full eval_folds via keys (model_name, store, product, fold_idx).

Usage:
  uv run python scripts/compute_entropy_regime_metrics.py \\
    --checkpoint-dir models/checkpoints \\
    --demand-path data/processed/demand_imputed.parquet \\
    --state-path data/processed/state.parquet \\
    --output models/results/entropy_regime_metrics.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vn2.analyze.model_eval import (  # noqa: E402
    EvalTask,
    ProgressTracker,
    is_degenerate_forecast,
    load_checkpoint,
)
from vn2.analyze.sip_opt import Costs as SIPCosts, quantiles_to_pmf
from vn2.analyze.entropy_metrics import (
    full_week2_entropy_metrics,
    empirical_demand_pmf_from_counts,
    entropy_gap_model_vs_empirical,
)
from vn2.analyze.entropy_regime import add_entropy_regime_labels, add_sensitivity_ratio_column
from vn2.forecast.features import prepare_train_test_split  # noqa: E402


def _q_step(qdf: pd.DataFrame, step: int) -> np.ndarray | None:
    if step in qdf.index:
        return qdf.loc[step].values
    fs = float(step)
    if fs in qdf.index:
        return qdf.loc[fs].values
    return None


def _one_row(
    task: EvalTask,
    df: pd.DataFrame,
    master_df: pd.DataFrame | None,
    checkpoint_dir: Path,
    holdout_weeks: int,
    sip_costs: SIPCosts,
    state_df: pd.DataFrame | None,
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
    if y_train is None or y_test is None:
        return None
    r1 = _q_step(qdf, 1)
    r2 = _q_step(qdf, 2)
    if r1 is None or r2 is None:
        return None

    qcols = qdf.columns.values.astype(float)
    pmf1 = quantiles_to_pmf(r1, qcols, grain=sip_grain)
    pmf2 = quantiles_to_pmf(r2, qcols, grain=sip_grain)

    if state_df is not None:
        key = (task.store, task.product, task.fold_idx)
        if key in state_df.index:
            sr = state_df.loc[key]
            I0 = int(sr["on_hand"])
            Q1 = int(sr.get("intransit_1", 0))
            Q2 = int(sr.get("intransit_2", 0))
        else:
            I0, Q1, Q2 = 0, 0, 0
    else:
        I0, Q1, Q2 = 0, 0, 0

    em = full_week2_entropy_metrics(I0, Q1, Q2, pmf1, pmf2, sip_costs, max_Q=sip_grain)
    y_hist = y_train.values.astype(float).ravel()
    emp = empirical_demand_pmf_from_counts(y_hist, sip_grain)
    kl = float(entropy_gap_model_vs_empirical(pmf2, emp))

    row = {
        "model_name": task.model_name,
        "store": task.store,
        "product": task.product,
        "fold_idx": task.fold_idx,
        "kl_empirical_vs_model_h2": kl,
        **em,
    }
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="Compute entropy regime metrics batch")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("models/checkpoints"))
    p.add_argument("--demand-path", type=Path, default=Path("data/processed/demand_imputed.parquet"))
    p.add_argument("--master-path", type=Path, default=Path("data/processed/master.parquet"))
    p.add_argument("--state-path", type=Path, default=Path("data/processed/state.parquet"))
    p.add_argument("--output", type=Path, default=Path("models/results/entropy_regime_metrics.parquet"))
    p.add_argument("--holdout", type=int, default=8)
    p.add_argument("--sip-grain", type=int, default=1000)
    p.add_argument("--holding", type=float, default=0.2)
    p.add_argument("--shortage", type=float, default=1.0)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--models", type=str, default="", help="Comma-separated model names; default all")
    p.add_argument("--limit", type=int, default=0, help="Max tasks (debug)")
    p.add_argument("--no-regime-labels", action="store_true", help="Skip sensitivity + regime columns")
    args = p.parse_args()

    df = pd.read_parquet(args.demand_path)
    master_df = pd.read_parquet(args.master_path) if args.master_path.exists() else None

    state_df = None
    if args.state_path.exists():
        state_df = pd.read_parquet(args.state_path)
        if not isinstance(state_df.index, pd.MultiIndex):
            state_df = state_df.set_index(["store", "product", "week"])

    sip_c = SIPCosts(holding=args.holding, shortage=args.shortage)
    progress_path = args.output.parent / f"{args.output.stem}.progress.json"
    progress = ProgressTracker(progress_path)
    tasks = []
    model_filter = [m.strip() for m in args.models.split(",") if m.strip()]
    for model_dir in sorted(args.checkpoint_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_filter and model_dir.name not in model_filter:
            continue
        for sku_dir in model_dir.iterdir():
            if not sku_dir.is_dir():
                continue
            try:
                store, product = map(int, sku_dir.name.split("_"))
            except ValueError:
                continue
            for fold_idx in range(args.holdout):
                t = EvalTask(model_dir.name, store, product, fold_idx)
                if not progress.is_complete(t):
                    tasks.append(t)

    if args.limit > 0:
        tasks = tasks[: args.limit]

    print(f"Tasks to run: {len(tasks)}")

    rows = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(_one_row)(t, df, master_df, args.checkpoint_dir, args.holdout, sip_c, state_df, args.sip_grain)
        for t in tasks
    )
    valid = [r for r in rows if r is not None]
    for t, r in zip(tasks, rows):
        if r is not None:
            progress.mark_complete(t)
    progress.save()

    out = pd.DataFrame(valid)
    if out.empty:
        print("No rows computed.")
        return

    if not args.no_regime_labels:
        out = add_sensitivity_ratio_column(out)
        out = add_entropy_regime_labels(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output)
    print(f"Wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
