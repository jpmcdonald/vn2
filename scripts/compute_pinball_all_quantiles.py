#!/usr/bin/env python3
"""
Compute pinball loss at ALL quantile levels per model.

Model quantiles: [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
Service-level fractiles: [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.833]

For SL fractiles not matching model quantiles (e.g. 0.833), we linearly
interpolate between adjacent model quantiles.

For each (model, SKU, week) triplet and each quantile q:
  - Unweighted pinball: q * max(y - pred_q, 0) + (1-q) * max(pred_q - y, 0)
  - Cost-weighted pinball: cu * max(y - pred_q, 0) + co * max(pred_q - y, 0)
  - Volume-weighted: unweighted pinball * (SKU total 8-week demand / mean demand)

Outputs:
  - reports/pinball/pinball_summary.csv        (model x quantile table)
  - reports/pinball/pinball_cost_weighted.csv   (cost-weighted version)
  - reports/pinball/pinball_volume_weighted.csv (volume-weighted version)
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console

from vn2.forecast.evaluation import crps_score

console = Console()

MODELS = ['seasonal_naive', 'lightgbm_quantile', 'slurp_bootstrap', 'slurp_stockout_aware', 'deepar']
MODEL_QUANTILES = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
SL_FRACTILES = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.833]
ALL_FRACTILES = sorted(set(MODEL_QUANTILES + SL_FRACTILES))

CU = 1.0
CO = 0.2

CHECKPOINTS_DIR = Path('models/checkpoints_h3')
SALES_DIR = Path('data/raw')
OUTPUT_DIR = Path('reports/pinball')

WEEK_FILES = {
    1: ("Week 1 - 2024-04-15 - Sales.csv", "2024-04-15"),
    2: ("Week 2 - 2024-04-22 - Sales.csv", "2024-04-22"),
    3: ("Week 3 - 2024-04-29 - Sales.csv", "2024-04-29"),
    4: ("Week 4 - 2024-05-06 - Sales.csv", "2024-05-06"),
    5: ("Week 5 - 2024-05-13 - Sales.csv", "2024-05-13"),
    6: ("Week 6 - 2024-05-20 - Sales.csv", "2024-05-20"),
    7: ("Week 7 - 2024-05-27 - Sales.csv", "2024-05-27"),
    8: ("Week 8 - 2024-06-03 - Sales.csv", "2024-06-03"),
}


def load_actuals(week: int) -> dict:
    fname, date_col = WEEK_FILES[week]
    path = SALES_DIR / fname
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return {(int(r['Store']), int(r['Product'])): int(r[date_col]) for _, r in df.iterrows()}


def interpolate_quantile(qdf_row, target_q: float) -> float:
    """Linearly interpolate a quantile value from the model's quantile grid."""
    available_qs = sorted([q for q in qdf_row.index if isinstance(q, float)])
    vals = [qdf_row[q] for q in available_qs]

    if target_q in available_qs:
        return qdf_row[target_q]
    if target_q <= available_qs[0]:
        return vals[0]
    if target_q >= available_qs[-1]:
        return vals[-1]

    for i in range(len(available_qs) - 1):
        if available_qs[i] <= target_q <= available_qs[i + 1]:
            frac = (target_q - available_qs[i]) / (available_qs[i + 1] - available_qs[i])
            return vals[i] + frac * (vals[i + 1] - vals[i])
    return vals[-1]


def pinball(q: float, y: float, pred_q: float) -> float:
    """Standard pinball (quantile) loss."""
    if y >= pred_q:
        return q * (y - pred_q)
    else:
        return (1 - q) * (pred_q - y)


def cost_weighted_pinball(y: float, pred_q: float) -> float:
    """Asymmetric cost-weighted loss: cu * undershoot + co * overshoot."""
    if y >= pred_q:
        return CU * (y - pred_q)
    else:
        return CO * (pred_q - y)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    actuals_by_week = {w: load_actuals(w) for w in range(1, 9)}
    skus = sorted(actuals_by_week[1].keys())

    # Compute per-SKU total 8-week demand for volume weighting
    sku_total_demand = {}
    for sku in skus:
        sku_total_demand[sku] = sum(actuals_by_week[w].get(sku, 0) for w in range(1, 9))
    mean_total = np.mean(list(sku_total_demand.values()))

    # Collect pinball values: {model: {quantile: [list of losses]}}
    unweighted = {m: {q: [] for q in ALL_FRACTILES} for m in MODELS}
    cost_wt = {m: {q: [] for q in ALL_FRACTILES} for m in MODELS}
    vol_wt = {m: {q: [] for q in ALL_FRACTILES} for m in MODELS}
    crps_values = {m: [] for m in MODELS}

    q_levels_arr = np.array(MODEL_QUANTILES)

    for model in MODELS:
        console.print(f"[bold]{model}[/bold]")
        for week in range(1, 9):
            fold_idx = week - 1
            actuals = actuals_by_week[week]

            for sku in skus:
                store, product = sku
                actual = actuals.get(sku, 0)

                cp_path = CHECKPOINTS_DIR / model / f"{store}_{product}" / f"fold_{fold_idx}.pkl"
                if not cp_path.exists():
                    continue
                with open(cp_path, 'rb') as f:
                    data = pickle.load(f)
                qdf = data.get('quantiles')
                if qdf is None or 1 not in qdf.index:
                    continue

                row = qdf.loc[1]
                vol_weight = sku_total_demand[sku] / mean_total if mean_total > 0 else 1.0

                for q in ALL_FRACTILES:
                    pred = interpolate_quantile(row, q)
                    pb = pinball(q, actual, pred)
                    cpb = cost_weighted_pinball(actual, pred)

                    unweighted[model][q].append(pb)
                    cost_wt[model][q].append(cpb)
                    vol_wt[model][q].append(pb * vol_weight)

                # CRPS from the model's native quantile grid
                q_vals = np.array([row[q] for q in MODEL_QUANTILES if q in row.index])
                q_lvls = np.array([q for q in MODEL_QUANTILES if q in row.index])
                if len(q_vals) > 0:
                    try:
                        crps_values[model].append(float(crps_score(actual, q_vals, q_lvls)))
                    except Exception:
                        pass

        console.print(f"  Done ({len(unweighted[model][0.50])} obs, {len(crps_values[model])} CRPS)")

    # Build summary tables
    def build_table(data_dict: dict) -> pd.DataFrame:
        rows = []
        for model in MODELS:
            row = {'model': model}
            for q in ALL_FRACTILES:
                vals = data_dict[model][q]
                row[f'q_{q:.3f}'] = round(np.mean(vals), 4) if vals else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    uw_df = build_table(unweighted)
    cw_df = build_table(cost_wt)
    vw_df = build_table(vol_wt)

    uw_df.to_csv(OUTPUT_DIR / 'pinball_summary.csv', index=False)
    cw_df.to_csv(OUTPUT_DIR / 'pinball_cost_weighted.csv', index=False)
    vw_df.to_csv(OUTPUT_DIR / 'pinball_volume_weighted.csv', index=False)

    console.print("\n[bold green]Unweighted Pinball Loss (mean)[/bold green]")
    console.print(uw_df.to_string(index=False))

    console.print("\n[bold green]Cost-Weighted Pinball Loss (mean)[/bold green]")
    console.print(cw_df.to_string(index=False))

    console.print("\n[bold green]Volume-Weighted Pinball Loss (mean)[/bold green]")
    console.print(vw_df.to_string(index=False))

    # CRPS summary
    crps_rows = []
    for model in MODELS:
        vals = crps_values[model]
        if vals:
            crps_rows.append({
                'model': model,
                'crps_mean': round(np.mean(vals), 4),
                'crps_median': round(np.median(vals), 4),
                'crps_p90': round(np.percentile(vals, 90), 4),
                'crps_std': round(np.std(vals), 4),
                'n': len(vals),
            })
    crps_df = pd.DataFrame(crps_rows)
    crps_df.to_csv(OUTPUT_DIR / 'crps_summary.csv', index=False)
    console.print("\n[bold green]CRPS Summary[/bold green]")
    console.print(crps_df.to_string(index=False))

    console.print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
