#!/usr/bin/env python3
"""Generate guardrail and selector override files for a configurable week.

Analyzes Weeks 1..N actuals to flag under-forecasting SKUs and write:
- reports/guardrail_overrides_w{N+1}.csv (by default)
- reports/selector_overrides_w{N+1}.csv (by default)
"""

import argparse
from pathlib import Path
from typing import Dict
import pickle
import numpy as np
import pandas as pd
from scipy.stats import kstest

# Local imports
from vn2.analyze.sequential_backtest import quantiles_to_pmf
from vn2.analyze.sequential_planner import Costs
from vn2.analyze.sequential_planner import (
    _safe_pmf, _shift_right, _conv_fft,
    leftover_from_stock_and_demand, expected_pos_neg_from_Z, leftover_from_Z,
)

WEEK_SALES_FILES: Dict[int, Path] = {
    1: Path('data/raw/Week 1 - 2024-04-15 - Sales.csv'),
    2: Path('data/raw/Week 2 - 2024-04-22 - Sales.csv'),
    3: Path('data/raw/Week 3 - 2024-04-29 - Sales.csv'),
    4: Path('data/raw/Week 4 - 2024-05-06 - Sales.csv'),
    5: Path('data/raw/Week 5 - 2024-05-13 - Sales.csv'),
}


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def extract_actuals_up_to_week(target_week: int) -> Dict[int, pd.DataFrame]:
    """Extract actual demand from Week 1..target_week sales files."""
    actuals: Dict[int, pd.DataFrame] = {}
    for week in range(1, target_week + 1):
        sales_path = WEEK_SALES_FILES.get(week)
        if sales_path is None:
            print(f"Warning: No configured sales file for week {week}")
            continue
        if not sales_path.exists():
            print(f"Warning: Sales file missing for week {week}: {sales_path}")
            continue
        df = pd.read_csv(sales_path)
        week_col = df.columns[-1]
        actuals[week] = df[['Store', 'Product', week_col]].rename(
            columns={week_col: f'actual_w{week}'}
        )
    return actuals


def get_quantiles(store: int, product: int, model: str):
    """Load quantile forecasts for a SKU and model."""
    ckpt = Path('models/checkpoints') / model / f'{store}_{product}' / 'fold_0.pkl'
    if not ckpt.exists():
        return None
    try:
        with open(ckpt, 'rb') as f:
            data = pickle.load(f)
        qdf = data.get('quantiles')
        if qdf is None or len(qdf) < 2:
            return None
        return qdf
    except Exception:
        return None


def interp_pit(quantiles: np.ndarray, levels: np.ndarray, actual: float) -> float:
    """Compute PIT (Probability Integral Transform) for actual value."""
    idx = np.searchsorted(quantiles, actual)
    if idx == 0:
        return float(levels[0] * (actual/quantiles[0] if quantiles[0] > 0 else 0.0))
    if idx >= len(quantiles):
        return 1.0
    x0, x1 = quantiles[idx-1], quantiles[idx]
    p0, p1 = levels[idx-1], levels[idx]
    if x1 == x0:
        return float(p1)
    frac = (actual - x0) / (x1 - x0)
    return float(p0 + frac * (p1 - p0))


def compute_calibration(selector_map: pd.DataFrame,
                        quantile_levels: np.ndarray,
                        actuals: dict) -> pd.DataFrame:
    """Compute calibration metrics (coverage, PIT, pinball loss) for each SKU."""
    rows = []
    # Gaussian weighting centered at 0.8333 within [0.73, 0.93]
    mu = 0.8333
    sigma = 0.05
    taus = quantile_levels
    gauss_w = np.exp(-0.5 * ((taus - mu) / sigma) ** 2)
    band_mask = (taus >= 0.73) & (taus <= 0.93)
    gauss_w = gauss_w * band_mask
    gauss_w = gauss_w / gauss_w.sum() if gauss_w.sum() > 0 else gauss_w
    
    for _, row in selector_map.iterrows():
        s = int(row['store']); p = int(row['product']); model = row['model_name']
        qdf = get_quantiles(s, p, model)
        if qdf is None:
            continue
        for h in [1, 2, 3]:
            if h not in qdf.index:
                continue
            q = qdf.loc[h].values
            # Compare h=1 forecasts to same-week actuals (weeks 1..target)
            # Additional horizons (h=2, h=3) may be extended similarly if needed.
            pit_vals = []
            pinball_losses = []
            for week in actuals.keys():
                # h=1 forecasts for week, h=2 forecasts for week+1
                if h == 1 and week in actuals:
                    a_df = actuals[week]
                    a_row = a_df[(a_df['Store'] == s) & (a_df['Product'] == p)]
                    if not a_row.empty:
                        actual = float(a_row.iloc[0, 2])
                        pit = interp_pit(q, quantile_levels, actual)
                        pit_vals.append(pit)
                        # Coverage metrics
                        cov50 = 1.0 if (actual >= np.interp(0.25, quantile_levels, q) and 
                                       actual <= np.interp(0.75, quantile_levels, q)) else 0.0
                        cov90 = 1.0 if (actual >= np.interp(0.05, quantile_levels, q) and 
                                       actual <= np.interp(0.95, quantile_levels, q)) else 0.0
                        # Pinball loss
                        diffs = actual - q
                        pinball_losses_w = np.maximum(taus * diffs, (taus - 1.0) * diffs)
                        pinball_losses.append(pinball_losses_w)
            if pit_vals:
                rows.append({
                    'store': s, 'product': p, 'h': h,
                    'cov50': np.mean([1.0 if 0.25 <= p <= 0.75 else 0.0 for p in pit_vals]),
                    'cov90': np.mean([1.0 if 0.05 <= p <= 0.95 else 0.0 for p in pit_vals]),
                    'pit': np.mean(pit_vals),
                    'pinball_mean': np.mean([np.mean(pl) for pl in pinball_losses]) if pinball_losses else np.nan,
                    'pinball_w': np.mean([np.sum(gauss_w * pl) for pl in pinball_losses]) if pinball_losses else np.nan
                })
    return pd.DataFrame(rows)


def score_underforecast(calib_df: pd.DataFrame,
                        quantile_levels: np.ndarray,
                        actuals: dict) -> pd.DataFrame:
    """Score SKUs for under-forecasting based on PIT exceedances."""
    records = []
    for (s,p), grp in calib_df.groupby(['store','product']):
        score = 0.0
        count_q90 = 0
        count_q95 = 0
        tail_sum = 0.0
        for h in [1,2,3]:
            sub = grp[grp['h']==h]
            if sub.empty:
                continue
            pit = sub.iloc[0]['pit']
            # PIT > 0.9 implies actual above Q90
            if pit > 0.9:
                count_q90 += 1
                score += 1
            if pit > 0.95:
                count_q95 += 1
                score += 1  # additional weight
            # Tail distance
            if pit > 0.9:
                tail_sum += (pit - 0.9) / 0.1
        score += count_q95  # 2× weight total for >Q95
        score += tail_sum
        flag = (score >= 2.0) or (count_q90 >= 2) or (count_q95 >= 1 and tail_sum >= 0.5)
        records.append({
            'store': s, 'product': p, 'score': score, 
            'count_q90': count_q90, 'count_q95': count_q95, 
            'tail_sum': tail_sum, 'flag': flag
        })
    return pd.DataFrame(records)


def write_overrides(out_csv: Path, flags_df: pd.DataFrame, week_range_label: str):
    """Write guardrail overrides CSV."""
    ensure_parent(out_csv)
    if flags_df.empty or not flags_df['flag'].any():
        pd.DataFrame(columns=['Store','Product','service_level_override','sigma_multiplier','reason']).to_csv(out_csv, index=False)
        return
    recs = flags_df[flags_df['flag']].copy()
    recs['Store'] = recs['store'].astype(int)
    recs['Product'] = recs['product'].astype(int)
    recs['service_level_override'] = 0.88
    recs['sigma_multiplier'] = 1.15
    recs['reason'] = f'Under-forecast score threshold met ({week_range_label})'
    recs[['Store','Product','service_level_override','sigma_multiplier','reason']].to_csv(out_csv, index=False)


def write_selector_overrides(out_csv: Path):
    """Write empty selector overrides (can be populated later if needed)."""
    ensure_parent(out_csv)
    pd.DataFrame(columns=['Store','Product','current_model','best_challenger','cost_current','cost_best','cost_improv','pinball_w_current','pinball_w_best','pinball_improv','recommend_switch']).to_csv(out_csv, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selector-map', type=Path, default=Path('models/results/selector_map_bias_adjusted.parquet'))
    ap.add_argument('--target-week', type=int, default=5, help='Latest completed week to include (>=1)')
    ap.add_argument('--out-guardrails', type=Path, default=None, help='Output CSV for guardrail overrides (defaults to reports/guardrail_overrides_w{target+1}.csv)')
    ap.add_argument('--out-selector', type=Path, default=None, help='Output CSV for selector overrides (defaults to reports/selector_overrides_w{target+1}.csv)')
    args = ap.parse_args()

    if args.target_week < 1:
        print("Error: target-week must be >= 1")
        return 1

    guardrail_out = args.out_guardrails or Path(f'reports/guardrail_overrides_w{args.target_week + 1}.csv')
    selector_out = args.out_selector or Path(f'reports/selector_overrides_w{args.target_week + 1}.csv')

    print(f"Generating override files using Weeks 1–{args.target_week} actuals...")

    # Load selector map
    if not args.selector_map.exists():
        print(f"Error: Selector map not found: {args.selector_map}")
        return 1
    selector = pd.read_parquet(args.selector_map)
    print(f"Loaded selector map with {len(selector)} SKUs")

    # Load actuals
    actuals = extract_actuals_up_to_week(args.target_week)
    print(f"Loaded actuals for weeks: {list(actuals.keys())}")
    if not actuals:
        print("Error: No actuals data found")
        return 1

    # Quantile levels
    q_levels = np.array([0.01,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99])

    # Compute calibration
    print("Computing calibration metrics...")
    calib_df = compute_calibration(selector, q_levels, actuals)
    print(f"Computed calibration for {len(calib_df)} SKU-horizon pairs")

    # Score under-forecasting
    print("Scoring under-forecast SKUs...")
    flags_df = score_underforecast(calib_df, q_levels, actuals)
    flagged_count = flags_df['flag'].sum()
    print(f"Flagged {flagged_count} SKUs for guardrails")

    # Write outputs
    week_range_label = f"W1–W{args.target_week}"
    write_overrides(guardrail_out, flags_df, week_range_label)
    write_selector_overrides(selector_out)

    print(f"✓ Guardrails written: {guardrail_out}")
    print(f"✓ Selector overrides written: {selector_out}")

    return 0


if __name__ == '__main__':
    exit(main())



