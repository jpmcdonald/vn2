"""
Build (store, product, fold_idx) SIP state at rolling-origin test starts.

Replays L=3-style weekly dynamics with zero orders (documented assumption) from
competition Initial State through each SKU's demand history, aligned with
``prepare_train_test_split`` split indices.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from vn2.data.loaders import load_initial_state, submission_index


def _step_l3(
    on_hand: int,
    in_transit: List[int],
    demand: int,
    order_q: int = 0,
) -> Tuple[int, List[int]]:
    """One week: same transition as szablowski.harness.simulate_week_l3 (single SKU)."""
    arriving = in_transit[0]
    available = on_hand + arriving
    d = max(0, int(demand))
    sold = min(available, d)
    leftover = max(0, available - d)
    new_it = [in_transit[1], in_transit[2], int(order_q)]
    return int(leftover), new_it


def state_at_split(
    demands: np.ndarray,
    split_idx: int,
    i0: int,
    it0: int,
    it1: int,
) -> Tuple[int, int, int]:
    """
    State at **start** of row ``split_idx`` (first test week), after training rows
    ``0 .. split_idx-1`` have been processed.
    """
    on_hand = int(i0)
    in_transit = [int(it0), int(it1), 0]
    for t in range(split_idx):
        d = int(demands[t]) if t < len(demands) else 0
        on_hand, in_transit = _step_l3(on_hand, in_transit, d, order_q=0)
    i0_out = on_hand
    q1_out = in_transit[0]
    q2_out = in_transit[1]
    return i0_out, q1_out, q2_out


def build_state_panel(
    demand_df: pd.DataFrame,
    raw_dir: Path,
    holdout_weeks: int = 8,
    min_train_rows: int = 52,
) -> pd.DataFrame:
    """
    Returns long DataFrame with columns store, product, fold_idx, on_hand,
    intransit_1, intransit_2.
    """
    raw_dir = Path(raw_dir)
    idx = submission_index(str(raw_dir))
    base = load_initial_state(str(raw_dir), idx)
    init_map = {
        (int(s), int(p)): (
            int(round(base.loc[(s, p), "on_hand"])),
            int(round(base.loc[(s, p), "intransit_1"])),
            int(round(base.loc[(s, p), "intransit_2"])),
        )
        for s, p in idx
    }

    dc = "week_date" if "week_date" in demand_df.columns else "week"
    sc = "sales" if "sales" in demand_df.columns else "demand"

    rows = []
    for (store, product), sku_df in demand_df.groupby(["Store", "Product"], sort=False):
        sku_df = sku_df.sort_values(dc)
        n = len(sku_df)
        if (store, product) not in init_map:
            continue
        i0, it1, it2 = init_map[(store, product)]
        demands = sku_df[sc].fillna(0).values.astype(float)
        demands = np.round(demands).astype(int)

        for fold_idx in range(holdout_weeks):
            split_idx = n - (holdout_weeks - fold_idx)
            if split_idx < min_train_rows:
                continue
            oh, q1, q2 = state_at_split(demands, split_idx, i0, it1, it2)
            rows.append(
                {
                    "store": int(store),
                    "product": int(product),
                    "fold_idx": int(fold_idx),
                    "on_hand": oh,
                    "intransit_1": q1,
                    "intransit_2": q2,
                }
            )

    return pd.DataFrame(rows)


def state_panel_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write panel with MultiIndex (store, product, fold_idx) for fast lookup."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.set_index(["store", "product", "fold_idx"]).sort_index()
    out.to_parquet(path)
