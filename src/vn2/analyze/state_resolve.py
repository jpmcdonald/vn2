"""
Resolve SIP initial row from state DataFrame (2-level week-0 or 3-level panel).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd


def resolve_sip_state_row(
    state_df: Optional[pd.DataFrame],
    store: int,
    product: int,
    fold_idx: int,
) -> Tuple[int, int, int]:
    """
    Return (on_hand, intransit_1, intransit_2). Falls back to zeros if missing.
    """
    if state_df is None or state_df.empty:
        return 0, 0, 0

    df = state_df
    idx = df.index

    if isinstance(idx, pd.MultiIndex) and len(idx.names) == 3:
        key = (store, product, fold_idx)
        if key in idx:
            return _row_to_tuple(df.loc[key])
        return 0, 0, 0

    if isinstance(idx, pd.MultiIndex) and len(idx.names) == 2:
        for k in ((store, product),):
            try:
                if k in idx:
                    return _row_to_tuple(df.loc[k])
            except (KeyError, TypeError):
                pass
        return 0, 0, 0

    if {"store", "product", "fold_idx"}.issubset(df.columns):
        m = (
            (df["store"] == store)
            & (df["product"] == product)
            & (df["fold_idx"] == fold_idx)
        )
        sub = df.loc[m]
        if len(sub) >= 1:
            return _row_to_tuple(sub.iloc[0])
        return 0, 0, 0

    return 0, 0, 0


def _row_to_tuple(row: Union[pd.Series, pd.DataFrame]) -> Tuple[int, int, int]:
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    oh = int(round(float(row["on_hand"])))
    q1 = int(round(float(row["intransit_1"])))
    q2 = int(round(float(row["intransit_2"])))
    return oh, q1, q2


def load_state_parquet(path: Optional[Path]) -> Optional[pd.DataFrame]:
    """Load parquet; normalize MultiIndex names to store, product[, fold_idx]."""
    if path is None or not Path(path).exists():
        return None
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.MultiIndex) and {"store", "product"}.issubset(
        df.columns
    ):
        if "fold_idx" in df.columns:
            df = df.set_index(["store", "product", "fold_idx"]).sort_index()
        else:
            df = df.set_index(["store", "product"]).sort_index()
        return df
    if isinstance(df.index, pd.MultiIndex):
        nn = []
        for n in df.index.names:
            ln = str(n).lower() if n is not None else ""
            if ln == "store":
                nn.append("store")
            elif ln == "product":
                nn.append("product")
            elif ln in ("fold_idx", "week", "fold"):
                nn.append("fold_idx")
            else:
                nn.append(n)
        df = df.copy()
        df.index = df.index.set_names(nn)
    return df
