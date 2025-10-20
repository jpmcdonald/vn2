#!/usr/bin/env python3
import os
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
res_dir = ROOT / 'models' / 'results'

suffix = '_v4_comp'  # keep consistent with running job

folds_final = res_dir / f'eval_folds{suffix}.parquet'
leader_path = res_dir / f'eval_leaderboard{suffix}.parquet'


def combine_parts() -> Path:
    parts = sorted(res_dir.glob(f'eval_folds__v4_comp_part-*.parquet'))
    if not parts:
        print('No part files found.', file=sys.stderr)
        return folds_final
    print(f'Combining {len(parts)} part files -> {folds_final}')
    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(folds_final)
    return folds_final


def aggregate(folds_path: Path):
    from vn2.analyze.model_eval import aggregate_results
    out_dir = res_dir
    aggregate_results(folds_path, out_dir, out_suffix='_v4_comp')


def print_top():
    if not leader_path.exists():
        print('Leaderboard not found.', file=sys.stderr)
        return
    df = pd.read_parquet(leader_path)
    cols = [c for c in ['model','sip_realized_cost_w2','shortage_cost','holding_cost','service_level'] if c in df.columns]
    print('=== Top 15 Leaderboard (sip_realized_cost_w2) ===')
    print(df.sort_values('sip_realized_cost_w2').head(15)[cols].to_string(index=False))


def main():
    # If final folds file missing, combine parts
    if not folds_final.exists():
        combine_parts()
    # Aggregate
    aggregate(folds_final)
    # Print
    print_top()


if __name__ == '__main__':
    main()


