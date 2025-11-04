#!/usr/bin/env python3
"""
Generate next order with selector and guardrail overrides.

Usage:
  python scripts/generate_order_with_overrides.py \
    --state-file data/states/state3.csv \
    --selector-map models/results/selector_map_seq12_v1.parquet \
    --selector-overrides reports/selector_overrides_w6.csv \
    --guardrail-overrides reports/guardrail_overrides_w6.csv \
    --checkpoints-dir models/checkpoints \
    --output data/submissions/order4_jpatrickmcdonald.csv
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from vn2.analyze.sequential_planner import Costs, choose_order_L2
from vn2.analyze.sequential_backtest import quantiles_to_pmf


def load_state_csv(state_path: Path) -> pd.DataFrame:
    df = pd.read_csv(state_path)
    # Normalize
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if 'end inventory' in cl:
            col_map[col] = 'end_inventory'
        elif 'in transit w+1' in cl:
            col_map[col] = 'intransit_1'
        elif 'in transit w+2' in cl:
            col_map[col] = 'intransit_2'
    if col_map:
        df = df.rename(columns=col_map)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state-file', type=Path, required=True)
    p.add_argument('--selector-map', type=Path, required=True)
    p.add_argument('--selector-overrides', type=Path, default=None)
    p.add_argument('--guardrail-overrides', type=Path, default=None)
    p.add_argument('--checkpoints-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sip-grain', type=int, default=500)
    args = p.parse_args()

    state = load_state_csv(args.state_file)
    selector = pd.read_parquet(args.selector_map)

    # Apply selector overrides (only where recommend_switch == True)
    if args.selector_overrides and Path(args.selector_overrides).exists():
        sel_ov = pd.read_csv(args.selector_overrides)
        sel_ov = sel_ov[sel_ov.get('recommend_switch', False) == True]
        if not sel_ov.empty:
            sel_ov = sel_ov[['Store','Product','best_challenger']].rename(columns={'Store':'store','Product':'product','best_challenger':'override_model'})
            selector = selector.merge(sel_ov, on=['store','product'], how='left')
            selector['model_name'] = selector['override_model'].fillna(selector['model_name'])
            selector = selector.drop(columns=['override_model'])

    # Guardrail overrides
    guard = None
    if args.guardrail_overrides and Path(args.guardrail_overrides).exists():
        guard = pd.read_csv(args.guardrail_overrides)
        if not guard.empty:
            guard = guard[['Store','Product','service_level_override']].rename(columns={'Store':'store','Product':'product'})

    # Build a lookup for model per SKU
    model_for = {(int(r.store), int(r.product)): r.model_name for r in selector.itertuples(index=False)}

    # Orders result
    out_rows = []

    # Fixed cost baseline
    base_co = 0.2
    base_cu = 1.0

    # Quantile levels
    q_levels = np.array([0.01,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99])

    for _, row in state.iterrows():
        store = int(row['Store']); product = int(row['Product'])
        I0 = int(row.get('end_inventory', row.get('End Inventory', 0)))
        # Use intransit_2 as the order arriving next week per prior script's semantics
        Q1 = 0
        Q2 = int(row.get('intransit_2', row.get('In Transit W+2', 0)))

        model = model_for.get((store, product))
        if model is None:
            out_rows.append({'Store': store, 'Product': product, '0': 0})
            continue
        ckpt = args.checkpoints_dir / model / f'{store}_{product}' / 'fold_0.pkl'
        if not ckpt.exists():
            out_rows.append({'Store': store, 'Product': product, '0': 0})
            continue
        try:
            with open(ckpt, 'rb') as f:
                data = pickle.load(f)
            qdf = data.get('quantiles')
            if qdf is None or qdf.empty or (1 not in qdf.index) or (2 not in qdf.index):
                out_rows.append({'Store': store, 'Product': product, '0': 0})
                continue
            h1 = quantiles_to_pmf(qdf.loc[1].values, q_levels, grain=args.sip_grain)
            h2 = quantiles_to_pmf(qdf.loc[2].values, q_levels, grain=args.sip_grain)
        except Exception:
            out_rows.append({'Store': store, 'Product': product, '0': 0})
            continue

        # Guardrail: adjust service level by modifying cu to achieve s = cu/(cu+co)
        s_override = None
        if guard is not None:
            gr = guard[(guard['store']==store)&(guard['product']==product)]
            if not gr.empty:
                s_override = float(gr.iloc[0]['service_level_override'])
        if s_override is not None and 0.5 < s_override < 0.99:
            co = base_co
            cu = co * s_override / (1.0 - s_override)
            costs = Costs(holding=co, shortage=cu)
        else:
            costs = Costs(holding=base_co, shortage=base_cu)

        try:
            q_opt, _ = choose_order_L2(h1, h2, I0, Q1, Q2, costs)
            out_rows.append({'Store': store, 'Product': product, '0': int(q_opt)})
        except Exception:
            out_rows.append({'Store': store, 'Product': product, '0': 0})

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.output, index=False)
    print(f"✓ Wrote order: {args.output}  total_units={out_df['0'].sum()}  skus={(out_df['0']>0).sum()}")


if __name__ == '__main__':
    main()
