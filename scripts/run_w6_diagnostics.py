#!/usr/bin/env python3
"""
W6 Diagnostics & Report Generator

Outputs:
- reports/diagnostics_w6.md
- reports/guardrail_overrides_w6.csv

CLI:
    python scripts/run_w6_diagnostics.py \
      --out-md reports/diagnostics_w6.md \
      --out-overrides reports/guardrail_overrides_w6.csv
"""

import argparse
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.stats import kstest

# Local imports
from vn2.analyze.sequential_backtest import quantiles_to_pmf
from vn2.analyze.order_analysis import compute_expected_cost_with_ci
from vn2.analyze.sequential_planner import Costs
from vn2.analyze.sequential_planner import (
	_safe_pmf, _shift_right, _conv_fft,
	leftover_from_stock_and_demand, expected_pos_neg_from_Z, leftover_from_Z,
)


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def load_states() -> dict:
    base = Path('data/states')
    states = {}
    for w, fname in [(1, 'state1.csv'), (2, 'state2.csv'), (3, 'state3.csv')]:
        p = base / fname
        if p.exists():
            states[w] = pd.read_csv(p)
    return states


def realized_portfolio_metrics(state_df: pd.DataFrame) -> dict:
    hold = float(state_df['Holding Cost'].sum())
    short = float(state_df['Shortage Cost'].sum())
    sales = float(state_df['Sales'].sum())
    missed = float(state_df['Missed Sales'].sum())
    demand = sales + missed
    fill = (sales / demand) if demand > 0 else 1.0
    stockout_skus = int((state_df['Missed Sales'] > 0).sum())
    return {
        'holding': hold,
        'shortage': short,
        'total': hold + short,
        'fill_rate': fill,
        'stockout_skus': stockout_skus,
        'demand': demand,
    }


def pipeline_sum(state_df: pd.DataFrame) -> int:
    col = 'In Transit W+1'
    return int(state_df[col].sum()) if col in state_df.columns else 0


def load_selector_map() -> pd.DataFrame:
    p = Path('models/results/selector_map_seq12_v1.parquet')
    return pd.read_parquet(p)


def load_initial_state_parquet() -> pd.DataFrame:
    p = Path('data/interim/state.parquet')
    df = pd.read_parquet(p)
    if not isinstance(df.index, pd.MultiIndex) and {'Store','Product'}.issubset(df.columns):
        df = df.set_index(['Store','Product'])
    return df


def load_sales_week(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def extract_actuals_w1_w3() -> dict:
    week_paths = {
        1: Path('data/raw/Week 1 - 2024-04-15 - Sales.csv'),
        2: Path('data/raw/Week 2 - 2024-04-22 - Sales.csv'),
        3: Path('data/raw/Week 3 - 2024-04-29 - Sales.csv'),
    }
    actuals = {}
    for w, p in week_paths.items():
        df = load_sales_week(p)
        # Last column is the week demand
        week_col = df.columns[-1]
        actuals[w] = df[['Store','Product', week_col]].rename(columns={week_col: f'actual_w{w}'})
    return actuals


def get_quantiles(store: int, product: int, model: str):
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


def compute_expected_w3_portfolio(submission_path: Path,
                                  selector_map: pd.DataFrame,
                                  initial_state: pd.DataFrame,
                                  quantile_levels: np.ndarray,
                                  costs: Costs,
                                  pmf_grain: int = 500) -> dict:
	orders = pd.read_csv(submission_path)
	week1_col = '2024-04-15'
	results = []
	for _, r in orders.iterrows():
		s = int(r['Store']); p = int(r['Product']); q = int(r[week1_col])
		if (s,p) not in initial_state.index:
			continue
		I0 = int(initial_state.loc[(s,p)]['on_hand'])
		Q1 = int(initial_state.loc[(s,p)]['intransit_1'])
		Q2 = int(initial_state.loc[(s,p)]['intransit_2'])
		row = selector_map[(selector_map['store']==s) & (selector_map['product']==p)]
		if row.empty:
			continue
		model = row.iloc[0]['model_name']
		qdf = get_quantiles(s, p, model)
		if qdf is None:
			continue
		try:
			q1 = qdf.loc[1].values if 1 in qdf.index else qdf.iloc[0].values
			q2 = qdf.loc[2].values if 2 in qdf.index else qdf.iloc[1].values
			if 3 in qdf.index:
				q3 = qdf.loc[3].values
			else:
				q3 = q2
		except Exception:
			continue
		# Convert quantiles to PMFs
		h1_pmf = quantiles_to_pmf(q1, quantile_levels, pmf_grain)
		h2_pmf = quantiles_to_pmf(q2, quantile_levels, pmf_grain)
		h3_pmf = quantiles_to_pmf(q3, quantile_levels, pmf_grain)
		# Step 1: leftover after week 1
		S0 = I0 + Q1
		L1_pmf = leftover_from_stock_and_demand(S0, h1_pmf)
		# Step 2: inventory entering week 2 (shift by Q2)
		Lpre_pmf = _shift_right(L1_pmf, Q2)
		# Step 3: leftover after week 2 via convolution Z = Lpre - D2
		h2_rev = _safe_pmf(h2_pmf)[::-1]
		Z_pmf = _conv_fft(Lpre_pmf, h2_rev)
		z_min = -(len(h2_rev) - 1)
		L2_pmf = leftover_from_Z(Z_pmf, z_min)
		# Step 4: inventory when q arrives at start of week 3
		inv_w3_pmf = _shift_right(L2_pmf, int(q))
		# Step 5: cost vs week 3 demand
		h3_rev = _safe_pmf(h3_pmf)[::-1]
		final_Z = _conv_fft(inv_w3_pmf, h3_rev)
		final_z_min = -(len(h3_rev) - 1)
		E_over, E_under = expected_pos_neg_from_Z(final_Z, final_z_min)
		expected_cost = costs.holding * E_over + costs.shortage * E_under
		# Build cost distribution for CI
		idx = np.arange(len(final_Z))
		z_vals = final_z_min + idx
		cost_values = np.where(z_vals > 0, costs.holding * z_vals, costs.shortage * (-z_vals))
		sorted_idx = np.argsort(cost_values)
		sorted_costs = cost_values[sorted_idx]
		sorted_pmf = final_Z[sorted_idx]
		cdf = np.cumsum(sorted_pmf)
		c5 = sorted_costs[np.searchsorted(cdf, 0.05)]
		c95 = sorted_costs[np.searchsorted(cdf, 0.95)]
		var = np.sum(final_Z * (cost_values - expected_cost)**2)
		std = np.sqrt(var)
		results.append((expected_cost, c5, c95, std))
	if not results:
		return {'expected': 0.0, 'c5': 0.0, 'c95': 0.0, 'std': 0.0}
	arr = np.array(results)
	expected = float(arr[:,0].sum())
	c5 = float(arr[:,1].sum())
	c95 = float(arr[:,2].sum())
	std = float(np.sqrt((arr[:,3]**2).sum()))
	return {'expected': expected, 'c5': c5, 'c95': c95, 'std': std}


def interp_pit(quantiles: np.ndarray, levels: np.ndarray, actual: float) -> float:
    # Monotonic quantiles assumed
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
            a_df = actuals[h]
            a_row = a_df[(a_df['Store'] == s) & (a_df['Product'] == p)]
            if a_row.empty:
                continue
            actual = float(a_row.iloc[0, 2])
            def covered(q_lo, q_hi):
                return 1.0 if (actual >= q_lo and actual <= q_hi) else 0.0
            cov50 = covered(np.interp(0.25, quantile_levels, q), np.interp(0.75, quantile_levels, q))
            cov90 = covered(np.interp(0.05, quantile_levels, q), np.interp(0.95, quantile_levels, q))
            pit = interp_pit(q, quantile_levels, actual)
            diffs = actual - q
            # Per-quantile pinball losses (non-negative)
            pinball_losses = np.maximum(taus * diffs, (taus - 1.0) * diffs)
            pinball_mean = float(np.mean(pinball_losses))
            if gauss_w.sum() > 0:
                pinball_weighted = float(np.sum(gauss_w * pinball_losses))
            else:
                pinball_weighted = pinball_mean
            rows.append({
                'store': s, 'product': p, 'h': h,
                'cov50': cov50, 'cov90': cov90, 'pit': pit,
                'pinball_mean': pinball_mean, 'pinball_w': pinball_weighted
            })
    return pd.DataFrame(rows)


def derive_cohorts(demand_long_path: Path) -> pd.DataFrame:
    if not demand_long_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(demand_long_path)
    grp = df.groupby(['Store','Product'])['sales']
    summary = grp.agg(['mean','std','count', lambda x: (x==0).mean()]).rename(columns={'<lambda_0>':'zero_share'})
    summary['cv'] = summary['std'] / summary['mean'].replace(0, np.nan)
    summary = summary.reset_index().rename(columns={'Store':'store','Product':'product','mean':'rate'})
    # Bin features
    summary['rate_bin'] = pd.qcut(summary['rate'].fillna(0), q=4, duplicates='drop').astype(str)
    summary['cv_bin'] = pd.qcut(summary['cv'].fillna(0), q=4, duplicates='drop').astype(str)
    summary['zero_bin'] = pd.qcut(summary['zero_share'].fillna(0), q=4, duplicates='drop').astype(str)
    return summary[['store','product','rate','cv','zero_share','rate_bin','cv_bin','zero_bin']]


def score_underforecast(calib_df: pd.DataFrame,
                        quantile_levels: np.ndarray,
                        actuals: dict) -> pd.DataFrame:
    # Compute exceedances >Q90 and >Q95 per SKU across h=1..3
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
                score += 1  # additional weight captured below as 2x
            # Tail distance approx: (pit-0.9)/0.1 if pit>0.9
            if pit > 0.9:
                tail_sum += (pit - 0.9) / 0.1
        score += count_q95  # 2× weight total for >Q95
        score += tail_sum
        flag = (score >= 2.0) or (count_q90 >= 2) or (count_q95 >= 1 and tail_sum >= 0.5)
        records.append({'store': s, 'product': p, 'score': score, 'count_q90': count_q90, 'count_q95': count_q95, 'tail_sum': tail_sum, 'flag': flag})
    return pd.DataFrame(records)


def list_candidate_models() -> list:
    ckpt_dir = Path('models/checkpoints')
    if not ckpt_dir.exists():
        return []
    return sorted([p.name for p in ckpt_dir.iterdir() if p.is_dir()])


def load_quantiles_for_model(store: int, product: int, model: str):
    ckpt = Path('models/checkpoints') / model / f'{store}_{product}' / 'fold_0.pkl'
    if not ckpt.exists():
        return None
    try:
        with open(ckpt, 'rb') as f:
            d = pickle.load(f)
        qdf = d.get('quantiles')
        return qdf
    except Exception:
        return None


def per_model_weighted_pinball(store: int, product: int, model: str,
                               actuals: dict,
                               quantile_levels: np.ndarray) -> float:
    qdf = load_quantiles_for_model(store, product, model)
    if qdf is None:
        return np.inf
    # Gaussian weights centered at 0.8333 within [0.73, 0.93]
    mu = 0.8333; sigma = 0.05
    taus = quantile_levels
    gauss_w = np.exp(-0.5 * ((taus - mu) / sigma) ** 2)
    band_mask = (taus >= 0.73) & (taus <= 0.93)
    gauss_w = gauss_w * band_mask
    if gauss_w.sum() == 0:
        return np.inf
    gauss_w = gauss_w / gauss_w.sum()
    vals = []
    for h in [1,2,3]:
        if h not in qdf.index:
            continue
        q = qdf.loc[h].values if h in qdf.index else None
        if q is None:
            continue
        a_df = actuals[h]
        a_row = a_df[(a_df['Store']==store)&(a_df['Product']==product)]
        if a_row.empty:
            continue
        actual = float(a_row.iloc[0,2])
        diffs = actual - q
        pinball_losses = np.maximum(taus*diffs, (taus-1.0)*diffs)
        vals.append(float(np.sum(gauss_w * pinball_losses)))
    if not vals:
        return np.inf
    return float(np.mean(vals))


def expected_w3_cost_for_model(store: int, product: int, model: str,
                               week1_order_qty: int,
                               initial_state: pd.DataFrame,
                               quantile_levels: np.ndarray,
                               costs: Costs,
                               pmf_grain: int = 500) -> float:
    if (store, product) not in initial_state.index:
        return np.inf
    I0 = int(initial_state.loc[(store, product)]['on_hand'])
    Q1 = int(initial_state.loc[(store, product)]['intransit_1'])
    Q2 = int(initial_state.loc[(store, product)]['intransit_2'])
    qdf = load_quantiles_for_model(store, product, model)
    if qdf is None:
        return np.inf
    try:
        q1 = qdf.loc[1].values if 1 in qdf.index else qdf.iloc[0].values
        q2 = qdf.loc[2].values if 2 in qdf.index else qdf.iloc[1].values
        q3 = qdf.loc[3].values if 3 in qdf.index else q2
    except Exception:
        return np.inf
    h1 = quantiles_to_pmf(q1, quantile_levels, pmf_grain)
    h2 = quantiles_to_pmf(q2, quantile_levels, pmf_grain)
    h3 = quantiles_to_pmf(q3, quantile_levels, pmf_grain)
    L1 = leftover_from_stock_and_demand(I0 + Q1, h1)
    Lpre = _shift_right(L1, Q2)
    Z = _conv_fft(Lpre, _safe_pmf(h2)[::-1])
    L2 = leftover_from_Z(Z, -(len(h2)-1))
    inv = _shift_right(L2, int(week1_order_qty))
    Z3 = _conv_fft(inv, _safe_pmf(h3)[::-1])
    Eover, Eunder = expected_pos_neg_from_Z(Z3, -(len(h3)-1))
    return float(costs.holding * Eover + costs.shortage * Eunder)


def challenger_comparisons(flags_df: pd.DataFrame,
                           selector_map: pd.DataFrame,
                           orders: pd.DataFrame,
                           initial_state: pd.DataFrame,
                           actuals: dict,
                           quantile_levels: np.ndarray,
                           costs: Costs) -> pd.DataFrame:
    cands = list_candidate_models()
    if not cands:
        return pd.DataFrame()
    week1_col = '2024-04-15'
    results = []
    flagged = flags_df[flags_df['flag']]
    for _, fr in flagged.iterrows():
        s = int(fr['store']); p = int(fr['product'])
        # current model
        row = selector_map[(selector_map['store']==s)&(selector_map['product']==p)]
        if row.empty:
            continue
        current_model = row.iloc[0]['model_name']
        # order qty
        ord_row = orders[(orders['Store']==s)&(orders['Product']==p)]
        if ord_row.empty:
            continue
        q = int(ord_row.iloc[0][week1_col])
        # baseline metrics
        base_cost = expected_w3_cost_for_model(s,p,current_model,q,initial_state,quantile_levels,costs)
        base_pinw = per_model_weighted_pinball(s,p,current_model,actuals,quantile_levels)
        best_model = current_model
        best_cost = base_cost
        best_pinw = base_pinw
        for m in cands:
            if m == current_model:
                continue
            cost_m = expected_w3_cost_for_model(s,p,m,q,initial_state,quantile_levels,costs)
            pinw_m = per_model_weighted_pinball(s,p,m,actuals,quantile_levels)
            # Prefer lower cost; tie-breaker lower pinball
            better = (cost_m < best_cost - 1e-9) or (abs(cost_m - best_cost) <= 1e-9 and pinw_m < best_pinw)
            if better:
                best_model, best_cost, best_pinw = m, cost_m, pinw_m
        # Improvements
        cost_improv = 0.0 if base_cost in [0.0, np.inf] else max(0.0, (base_cost - best_cost) / base_cost)
        pin_improv = 0.0 if base_pinw in [0.0, np.inf] else max(0.0, (base_pinw - best_pinw) / base_pinw)
        recommend = (best_model != current_model) and (cost_improv >= 0.10) and (pin_improv >= 0.20)
        results.append({
            'Store': s, 'Product': p,
            'current_model': current_model,
            'best_challenger': best_model,
            'cost_current': base_cost,
            'cost_best': best_cost,
            'cost_improv': cost_improv,
            'pinball_w_current': base_pinw,
            'pinball_w_best': best_pinw,
            'pinball_improv': pin_improv,
            'recommend_switch': recommend
        })
    return pd.DataFrame(results)


def write_selector_overrides(out_csv: Path, comp_df: pd.DataFrame):
    ensure_parent(out_csv)
    if comp_df.empty:
        pd.DataFrame(columns=['Store','Product','current_model','best_challenger','recommend_switch']).to_csv(out_csv, index=False)
        return
    cols = ['Store','Product','current_model','best_challenger','cost_current','cost_best','cost_improv','pinball_w_current','pinball_w_best','pinball_improv','recommend_switch']
    comp_df[cols].to_csv(out_csv, index=False)


def write_report(out_md: Path,
                 w_metrics: dict,
                 expected_w3: dict,
                 realized_w3_total: float,
                 calib_df: pd.DataFrame,
                 cohorts_df: pd.DataFrame,
                 flags_df: pd.DataFrame,
                 chall_df: pd.DataFrame = None):
    ensure_parent(out_md)
    lines = []
    lines.append('# W6 Diagnostics Report')
    lines.append('')
    lines.append('## Executive Summary')
    lines.append(f"- Week 3 realized total cost: €{realized_w3_total:.2f}")
    lines.append(f"- Expected Week 3 cost (decision-time): €{expected_w3['expected']:.2f} (90% CI: €{expected_w3['c5']:.2f}–€{expected_w3['c95']:.2f})")
    z = 0.0
    if expected_w3['std'] > 0:
        z = (realized_w3_total - expected_w3['expected']) / expected_w3['std']
    lines.append(f"- Deviation vs expected: {z:+.2f}σ")
    lines.append('')
    lines.append('## Portfolio Metrics (W1–W3)')
    lines.append('Week | Holding | Shortage | Total | Fill rate | Stockout SKUs | Demand')
    lines.append('---- | -------:| --------:| -----:| ---------:| -------------:| ------:')
    for w in [1,2,3]:
        m = w_metrics.get(w, {})
        if not m:
            continue
        lines.append(f"{w} | €{m['holding']:.1f} | €{m['shortage']:.1f} | €{m['total']:.1f} | {m['fill_rate']:.3f} | {m['stockout_skus']} | {int(m['demand'])}")
    lines.append('')
    lines.append('## Calibration Summary (h=1,2,3)')
    if not calib_df.empty:
        cal_sum = calib_df.groupby('h').agg(
            cov50=('cov50','mean'), cov90=('cov90','mean'),
            pit_ks=('pit', lambda x: kstest(x, 'uniform').statistic),
            pinball_w=('pinball_w','mean'), pinball_mean=('pinball_mean','mean')
        ).reset_index()
        lines.append('h | cov@50 | cov@90 | PIT_KS | weighted pinball (0.73–0.93) | mean pinball')
        lines.append('--|--------:|-------:|-------:|-----------------------------:|------------:')
        for _, r in cal_sum.iterrows():
            lines.append(f"{int(r['h'])} | {r['cov50']*100:.1f}% | {r['cov90']*100:.1f}% | {r['pit_ks']:.3f} | {r['pinball_w']:.3f} | {r['pinball_mean']:.3f}")
    else:
        lines.append('_No calibration data available_')
    lines.append('')
    lines.append('## Cohort Diagnostics')
    if not cohorts_df.empty and not calib_df.empty:
        merged = calib_df.merge(cohorts_df, on=['store','product'], how='left')
        coh = merged.groupby(['h','rate_bin']).agg(cov90=('cov90','mean')).reset_index()
        lines.append('By rate_bin (cov@90):')
        lines.append('h | rate_bin | cov@90')
        lines.append('--|----------|-------:')
        for _, r in coh.iterrows():
            lines.append(f"{int(r['h'])} | {r['rate_bin']} | {r['cov90']*100:.1f}%")
    else:
        lines.append('_No cohort data available_')
    lines.append('')
    lines.append('## Top Under-forecast SKUs (for W6 Guardrails)')
    if not flags_df.empty:
        top = flags_df[flags_df['flag']].nlargest(10, 'score')
        lines.append('Store | Product | Score | >Q90 | >Q95 | TailSum')
        lines.append('-----:| -------:| -----:| ----:| ----:| -------:')
        for _, r in top.iterrows():
            lines.append(f"{int(r['store'])} | {int(r['product'])} | {r['score']:.2f} | {int(r['count_q90'])} | {int(r['count_q95'])} | {r['tail_sum']:.2f}")
    else:
        lines.append('_No flagged SKUs_')
    lines.append('')
    if chall_df is not None and not chall_df.empty:
        lines.append('## Selector Challenger Comparisons (W6)')
        lines.append('Store | Product | Current | Best | ΔCost | ΔPinball_w | Switch?')
        lines.append('-----:| -------:|:------- |:---- | ----:| ---------:|:-------:')
        summary_rows = chall_df.copy()
        # Show up to 10 with recommended_switch True first
        summary_rows = pd.concat([
            summary_rows[summary_rows['recommend_switch']].nlargest(10, 'cost_improv'),
            summary_rows[~summary_rows['recommend_switch']].nlargest(10, 'cost_improv')
        ]).drop_duplicates(subset=['Store','Product'])
        for _, r in summary_rows.iterrows():
            lines.append(
                f"{int(r['Store'])} | {int(r['Product'])} | {r['current_model']} | {r['best_challenger']} | "
                f"{r['cost_improv']*100:.1f}% | {r['pinball_improv']*100:.1f}% | {'YES' if r['recommend_switch'] else 'NO'}"
            )
        lines.append('')
    lines.append('## Appendix')
    lines.append('- Methods: coverage@50/90, PIT uniformity (KS), (weighted) pinball, cohort splits')
    lines.append('- Guardrail policy: service_level_override=0.88 or sigma_multiplier=1.15 for flagged SKUs')
    out_md.write_text("\n".join(lines))


def write_overrides(out_csv: Path, flags_df: pd.DataFrame):
    ensure_parent(out_csv)
    if flags_df.empty:
        pd.DataFrame(columns=['Store','Product','service_level_override','sigma_multiplier','reason']).to_csv(out_csv, index=False)
        return
    recs = flags_df[flags_df['flag']].copy()
    recs['Store'] = recs['store'].astype(int)
    recs['Product'] = recs['product'].astype(int)
    recs['service_level_override'] = 0.88
    recs['sigma_multiplier'] = 1.15
    recs['reason'] = 'Under-forecast score threshold met (W1–W3)'
    recs[['Store','Product','service_level_override','sigma_multiplier','reason']].to_csv(out_csv, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-md', type=Path, default=Path('reports/diagnostics_w6.md'))
    ap.add_argument('--out-overrides', type=Path, default=Path('reports/guardrail_overrides_w6.csv'))
    ap.add_argument('--selector-overrides', type=Path, default=Path('reports/selector_overrides_w6.csv'))
    ap.add_argument('--submission', type=Path, default=Path('data/submissions/jpatrickmcdonald_actual.csv'))
    args = ap.parse_args()

    costs = Costs(holding=0.2, shortage=1.0)
    q_levels = np.array([0.01,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99])

    # Portfolio metrics W1–W3
    states = load_states()
    w_metrics = {w: realized_portfolio_metrics(df) for w, df in states.items()}

    # Expected vs realized for W3
    selector = load_selector_map()
    initial_state = load_initial_state_parquet()
    exp_w3 = compute_expected_w3_portfolio(args.submission, selector, initial_state, q_levels, costs)
    realized_w3_total = w_metrics.get(3, {}).get('total', 0.0)

    # Calibration
    actuals = extract_actuals_w1_w3()
    calib_df = compute_calibration(selector, q_levels, actuals)

    # Cohorts
    cohorts_df = derive_cohorts(Path('data/processed/demand_long.parquet'))

    # Flags & guardrails
    flags_df = score_underforecast(calib_df, q_levels, actuals)

    # Challenger comparisons on flagged SKUs
    orders = pd.read_csv(args.submission)
    chall_df = challenger_comparisons(flags_df, selector, orders, initial_state, actuals, q_levels, costs)
    write_selector_overrides(args.selector_overrides, chall_df)

    # Outputs
    write_report(args.out_md, w_metrics, exp_w3, realized_w3_total, calib_df, cohorts_df, flags_df, chall_df)
    write_overrides(args.out_overrides, flags_df)

    print(f"✓ Diagnostics written: {args.out_md}")
    print(f"✓ Guardrails written: {args.out_overrides}")
    print(f"✓ Selector comparisons: {args.selector_overrides}")


if __name__ == '__main__':
    main()
