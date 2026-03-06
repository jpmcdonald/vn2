#!/usr/bin/env python3
"""
Compute per-model bias, calibration, demand-segment analysis,
Wasserstein distance, and CRPS against competition-week actuals (Weeks 1-8).

For each model and each competition week:
  - Load fold_{week-1} checkpoint (h=1 forecast vs that week's actual)
  - Compare quantile forecasts against realized demand
  - Compute distributional quality (Wasserstein, CRPS)

Outputs:
  - reports/bias/bias_summary.csv           (per-model aggregate bias + Wasserstein/CRPS)
  - reports/bias/calibration_table.csv      (empirical coverage at each quantile)
  - reports/bias/segment_breakdown.csv      (bias by demand segment + Wasserstein/CRPS)
  - reports/bias/per_sku_week_detail.parquet (full detail for downstream use)
  - reports/bias/worst_skus.csv             (top 50 worst SKUs per model by composite score)
"""

import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from vn2.analyze.sip_opt import quantiles_to_pmf
from vn2.forecast.evaluation import crps_score, pinball_loss

console = Console()

CRITICAL_FRACTILE = 0.833

MODELS = ['seasonal_naive', 'lightgbm_quantile', 'slurp_bootstrap', 'slurp_stockout_aware', 'deepar']
QUANTILE_LEVELS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
CHECKPOINTS_DIR = Path('models/checkpoints_h3')
SALES_DIR = Path('data/raw')
OUTPUT_DIR = Path('reports/bias')

WEEK_FILES = {
    1: "Week 1 - 2024-04-15 - Sales.csv",
    2: "Week 2 - 2024-04-22 - Sales.csv",
    3: "Week 3 - 2024-04-29 - Sales.csv",
    4: "Week 4 - 2024-05-06 - Sales.csv",
    5: "Week 5 - 2024-05-13 - Sales.csv",
    6: "Week 6 - 2024-05-20 - Sales.csv",
    7: "Week 7 - 2024-05-27 - Sales.csv",
    8: "Week 8 - 2024-06-03 - Sales.csv",
}

WEEK_DATE_COLS = {
    1: "2024-04-15", 2: "2024-04-22", 3: "2024-04-29", 4: "2024-05-06",
    5: "2024-05-13", 6: "2024-05-20", 7: "2024-05-27", 8: "2024-06-03",
}


def demand_segment(d: int) -> str:
    if d == 0:
        return 'zero'
    elif d <= 3:
        return 'low (1-3)'
    elif d <= 10:
        return 'medium (4-10)'
    else:
        return 'high (10+)'


def load_actuals(week: int) -> dict:
    path = SALES_DIR / WEEK_FILES[week]
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    date_col = WEEK_DATE_COLS[week]
    return {(int(r['Store']), int(r['Product'])): int(r[date_col]) for _, r in df.iterrows()}


def load_checkpoint_quantiles(model: str, store: int, product: int, fold: int):
    path = CHECKPOINTS_DIR / model / f"{store}_{product}" / f"fold_{fold}.pkl"
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data.get('quantiles')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all actuals
    actuals_by_week = {w: load_actuals(w) for w in range(1, 9)}

    # Get SKU list from week 1
    skus = sorted(actuals_by_week[1].keys())
    console.print(f"SKUs: {len(skus)}")

    records = []

    for model in MODELS:
        console.print(f"\n[bold]{model}[/bold]")
        for week in range(1, 9):
            fold_idx = week - 1  # Order N uses fold N-1; week W's actuals correspond to order W
            horizon_step = 1     # h=1 from that fold's perspective

            actuals = actuals_by_week[week]
            n_loaded = 0
            for sku in skus:
                store, product = sku
                actual = actuals.get(sku, 0)
                qdf = load_checkpoint_quantiles(model, store, product, fold_idx)
                if qdf is None:
                    continue
                n_loaded += 1

                if horizon_step not in qdf.index:
                    continue

                row = qdf.loc[horizon_step]
                median_pred = row[0.50] if 0.50 in row.index else np.nan
                mean_pred = row.mean()

                q_levels_arr = np.array(QUANTILE_LEVELS)
                q_values_arr = np.array([row[q] for q in QUANTILE_LEVELS if q in row.index])
                q_levels_used = np.array([q for q in QUANTILE_LEVELS if q in row.index])

                # Wasserstein W1: distance between forecast PMF and point mass at actual
                try:
                    pmf = quantiles_to_pmf(q_values_arr, q_levels_used, grain=500)
                    support = np.arange(len(pmf))
                    wass = float(np.sum(pmf * np.abs(support - actual)))
                except Exception:
                    wass = np.nan

                # CRPS
                try:
                    crps_val = float(crps_score(actual, q_values_arr, q_levels_used))
                except Exception:
                    crps_val = np.nan

                # Pinball at the critical fractile (0.833) via interpolation
                try:
                    pred_at_cf = float(np.interp(CRITICAL_FRACTILE, q_levels_used, q_values_arr))
                    pinball_cf = float(pinball_loss(actual, pred_at_cf, CRITICAL_FRACTILE))
                except Exception:
                    pinball_cf = np.nan

                rec = {
                    'model': model,
                    'week': week,
                    'Store': store,
                    'Product': product,
                    'actual': actual,
                    'median_pred': median_pred,
                    'mean_pred': mean_pred,
                    'bias_median': median_pred - actual,
                    'bias_mean': mean_pred - actual,
                    'abs_error': abs(median_pred - actual),
                    'segment': demand_segment(actual),
                    'wasserstein': wass,
                    'crps': crps_val,
                    'pinball_cf': pinball_cf,
                    'composite': pinball_cf * wass if not (np.isnan(pinball_cf) or np.isnan(wass)) else np.nan,
                }

                # Coverage at each quantile
                for q in QUANTILE_LEVELS:
                    if q in row.index:
                        rec[f'q_{q:.2f}'] = row[q]
                        rec[f'covered_{q:.2f}'] = int(actual <= row[q])

                records.append(rec)

            console.print(f"  Week {week} (fold {fold_idx}): {n_loaded} SKUs loaded")

    df = pd.DataFrame(records)

    # --- Bias Summary ---
    bias_summary = df.groupby('model').agg(
        n=('actual', 'count'),
        mean_actual=('actual', 'mean'),
        mean_median_pred=('median_pred', 'mean'),
        bias_median=('bias_median', 'mean'),
        bias_mean=('bias_mean', 'mean'),
        mae=('abs_error', 'mean'),
        rmse=('abs_error', lambda x: np.sqrt((x**2).mean())),
        wasserstein_mean=('wasserstein', 'mean'),
        wasserstein_median=('wasserstein', 'median'),
        crps_mean=('crps', 'mean'),
        crps_median=('crps', 'median'),
        pinball_cf_mean=('pinball_cf', 'mean'),
        composite_mean=('composite', 'mean'),
    ).round(4)
    bias_summary.to_csv(OUTPUT_DIR / 'bias_summary.csv')
    console.print("\n[bold green]Bias Summary[/bold green]")
    console.print(bias_summary.to_string())

    # --- Calibration Table ---
    coverage_cols = [c for c in df.columns if c.startswith('covered_')]
    cal_rows = []
    for model in MODELS:
        mdf = df[df['model'] == model]
        for col in coverage_cols:
            q = float(col.replace('covered_', ''))
            empirical = mdf[col].mean()
            cal_rows.append({
                'model': model,
                'quantile': q,
                'expected_coverage': q,
                'empirical_coverage': round(empirical, 4),
                'miscalibration': round(empirical - q, 4),
            })
    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(OUTPUT_DIR / 'calibration_table.csv', index=False)

    # Print pivot
    cal_pivot = cal_df.pivot_table(
        index='model', columns='quantile', values='empirical_coverage')
    console.print("\n[bold green]Calibration (empirical coverage)[/bold green]")
    console.print(cal_pivot.to_string(float_format='%.3f'))

    # --- Segment Breakdown ---
    seg = df.groupby(['model', 'segment']).agg(
        n=('actual', 'count'),
        mean_actual=('actual', 'mean'),
        bias_median=('bias_median', 'mean'),
        mae=('abs_error', 'mean'),
        wasserstein_mean=('wasserstein', 'mean'),
        crps_mean=('crps', 'mean'),
    ).round(4)
    seg.to_csv(OUTPUT_DIR / 'segment_breakdown.csv')
    console.print("\n[bold green]Segment Breakdown[/bold green]")
    console.print(seg.to_string())

    # --- Per-SKU detail ---
    df.to_parquet(OUTPUT_DIR / 'per_sku_week_detail.parquet', index=False)
    console.print(f"\nDetail saved to {OUTPUT_DIR / 'per_sku_week_detail.parquet'}")

    # --- Weekly bias trend ---
    weekly = df.groupby(['model', 'week']).agg(
        bias_median=('bias_median', 'mean'),
        mae=('abs_error', 'mean'),
        wasserstein_mean=('wasserstein', 'mean'),
        crps_mean=('crps', 'mean'),
    ).round(4)
    console.print("\n[bold green]Weekly Bias Trend[/bold green]")
    console.print(weekly.to_string())

    # --- Worst SKUs by composite score (pinball_cf * wasserstein) ---
    worst_rows = []
    for model in MODELS:
        mdf = df[df['model'] == model].dropna(subset=['composite'])
        if mdf.empty:
            continue
        sku_composite = mdf.groupby(['Store', 'Product']).agg(
            composite_mean=('composite', 'mean'),
            wasserstein_mean=('wasserstein', 'mean'),
            crps_mean=('crps', 'mean'),
            pinball_cf_mean=('pinball_cf', 'mean'),
            mae=('abs_error', 'mean'),
            mean_actual=('actual', 'mean'),
            n_weeks=('actual', 'count'),
        ).sort_values('composite_mean', ascending=False).head(50)
        sku_composite = sku_composite.reset_index()
        sku_composite.insert(0, 'model', model)
        worst_rows.append(sku_composite)

    if worst_rows:
        worst_df = pd.concat(worst_rows, ignore_index=True).round(4)
        worst_df.to_csv(OUTPUT_DIR / 'worst_skus.csv', index=False)
        console.print(f"\n[bold green]Worst SKUs[/bold green] saved to {OUTPUT_DIR / 'worst_skus.csv'}")
        for model in MODELS:
            m_worst = worst_df[worst_df['model'] == model].head(5)
            if not m_worst.empty:
                console.print(f"\n  [bold]{model}[/bold] top-5 worst:")
                console.print(m_worst[['Store', 'Product', 'composite_mean', 'wasserstein_mean', 'crps_mean']].to_string(index=False))

    # --- Wasserstein/CRPS summary (separate file for downstream scripts) ---
    wc_summary = df.groupby('model').agg(
        wasserstein_mean=('wasserstein', 'mean'),
        wasserstein_p50=('wasserstein', 'median'),
        wasserstein_p90=('wasserstein', lambda x: x.quantile(0.9)),
        crps_mean=('crps', 'mean'),
        crps_p50=('crps', 'median'),
        crps_p90=('crps', lambda x: x.quantile(0.9)),
        composite_mean=('composite', 'mean'),
    ).round(4)
    wc_summary.to_csv(OUTPUT_DIR / 'wasserstein_crps_summary.csv')
    console.print(f"\n[bold green]Wasserstein/CRPS Summary[/bold green]")
    console.print(wc_summary.to_string())


if __name__ == '__main__':
    main()
