#!/usr/bin/env python3
"""
Backtest the ordinal progression of safety stock formulas against
our density-based SIP optimization using the 8-week competition data.

Policies compared:
  1. z * sigma * sqrt(L)         -- CSL-based, ignores review period
  2. z * sigma * sqrt(L+R)       -- accounts for review period
  3. z * RMSE * sqrt(L+R)        -- forecast error instead of demand variability
  4. k * RMSE                    -- optimize k via simulation (similar to VN2 winner)
  5. k * MAE                    -- more stable error indicator, optimize k
  6. Density-based SIP           -- full probabilistic quantile forecasts + newsvendor

All policies use the SAME simulation engine (initial state, actuals, L=3,
eval_costs cu=1.0, co=0.2) so results are directly comparable.

Usage:
    uv run python scripts/backtest_safety_stock_policies.py
"""

from __future__ import annotations

import argparse
import math
import pickle
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.stats import norm

console = Console()

EVAL_COSTS_HOLDING = 0.2
EVAL_COSTS_SHORTAGE = 1.0
L = 3
R = 1

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

QUANTILE_LEVELS = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40,
                            0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])


# ---------------------------------------------------------------------------
# Data loading (shared with full_L3_simulation.py)
# ---------------------------------------------------------------------------

@dataclass
class SKUState:
    store: int
    product: int
    on_hand: int
    in_transit: List[int] = field(default_factory=lambda: [0, 0, 0])


def load_initial_state(path: Path) -> Dict[Tuple[int, int], SKUState]:
    df = pd.read_csv(path)
    states = {}
    for _, row in df.iterrows():
        store, product = int(row['Store']), int(row['Product'])
        on_hand = int(row.get('End Inventory', row.get('Start Inventory', 0)))
        it1 = int(row.get('In Transit W+1', 0))
        it2 = int(row.get('In Transit W+2', 0))
        states[(store, product)] = SKUState(store=store, product=product,
                                            on_hand=on_hand,
                                            in_transit=[it1, it2, 0])
    return states


def load_sales(sales_dir: Path, week: int) -> Dict[Tuple[int, int], int]:
    if week not in WEEK_FILES:
        return {}
    fname, date_col = WEEK_FILES[week]
    path = sales_dir / fname
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return {(int(r['Store']), int(r['Product'])): int(r[date_col]) for _, r in df.iterrows()}


# ---------------------------------------------------------------------------
# Per-SKU statistics (pre-computed before simulation)
# ---------------------------------------------------------------------------

def compute_sku_demand_stats(demand_path: Path) -> pd.DataFrame:
    """Compute per-SKU mean and std from pre-competition demand."""
    df = pd.read_parquet(demand_path)
    comp_start = pd.Timestamp('2024-04-15')
    pre = df[df['week'] < comp_start]
    stats = pre.groupby(['Store', 'Product'])['demand'].agg(
        mean_demand='mean', sigma='std'
    ).fillna(0).reset_index()
    return stats


def compute_sku_forecast_errors(bias_detail_path: Path, model: str = 'slurp_bootstrap') -> pd.DataFrame:
    """Compute per-SKU RMSE and MAE from bias analysis detail."""
    df = pd.read_parquet(bias_detail_path)
    mdf = df[df['model'] == model]
    stats = mdf.groupby(['Store', 'Product']).agg(
        mae=('abs_error', 'mean'),
        rmse=('abs_error', lambda x: np.sqrt((x**2).mean())),
    ).reset_index()
    return stats


# ---------------------------------------------------------------------------
# Simulation engine (same as full_L3_simulation.py)
# ---------------------------------------------------------------------------

def simulate_week(
    states: Dict[Tuple[int, int], SKUState],
    sales: Dict[Tuple[int, int], int],
) -> Tuple[float, float, int, int, int]:
    """Simulate one week. Returns (holding, shortage, stockouts, total_demand, total_sold)."""
    total_holding = 0.0
    total_shortage = 0.0
    stockouts = 0
    total_demand = 0
    total_sold = 0

    for key, state in states.items():
        arriving = state.in_transit[0]
        available = state.on_hand + arriving
        demand = sales.get(key, 0)
        sold = min(available, demand)
        shortage = max(0, demand - available)
        leftover = max(0, available - demand)

        total_holding += EVAL_COSTS_HOLDING * leftover
        total_shortage += EVAL_COSTS_SHORTAGE * shortage
        if shortage > 0:
            stockouts += 1
        total_demand += demand
        total_sold += sold

        state.on_hand = leftover
        state.in_transit = [state.in_transit[1], state.in_transit[2], 0]

    return total_holding, total_shortage, stockouts, total_demand, total_sold


def place_orders(
    states: Dict[Tuple[int, int], SKUState],
    orders: Dict[Tuple[int, int], int],
):
    """Place orders into in_transit[2] (arriving in 3 weeks)."""
    for key, qty in orders.items():
        if key in states:
            states[key].in_transit[2] = qty


def run_simulation(
    initial_state_path: Path,
    sales_dir: Path,
    order_fn,
    max_weeks: int = 8,
) -> Dict:
    """
    Run full 8-week simulation with a given ordering function.

    order_fn(states, order_num) -> Dict[(store,product), int]
        Called at each decision point (orders 1-6).
    """
    states = load_initial_state(initial_state_path)

    # Order 1 at end of week 0
    orders = order_fn(states, 1)
    place_orders(states, orders)
    total_ordered = sum(orders.values())

    weekly_results = []
    for week in range(1, max_weeks + 1):
        sales = load_sales(sales_dir, week)
        holding, shortage, stockouts, demand, sold = simulate_week(states, sales)
        weekly_results.append({
            'week': week, 'holding': holding, 'shortage': shortage,
            'total': holding + shortage, 'stockouts': stockouts,
            'demand': demand, 'sold': sold,
        })

        next_order_num = week + 1
        if next_order_num <= 6:
            orders = order_fn(states, next_order_num)
            place_orders(states, orders)
            total_ordered += sum(orders.values())

    rdf = pd.DataFrame(weekly_results)
    return {
        'holding': rdf['holding'].sum(),
        'shortage': rdf['shortage'].sum(),
        'total': rdf['total'].sum(),
        'stockouts': rdf['stockouts'].sum(),
        'total_demand': rdf['demand'].sum(),
        'total_sold': rdf['sold'].sum(),
        'fill_rate': rdf['sold'].sum() / max(rdf['demand'].sum(), 1),
        'stockout_rate': rdf['stockouts'].sum() / (len(states) * max_weeks),
        'total_ordered': total_ordered,
        'weekly': rdf,
    }


# ---------------------------------------------------------------------------
# Policy implementations
# ---------------------------------------------------------------------------

def make_base_stock_policy(sku_stats: Dict[Tuple[int, int], dict], lead_time: int, safety_stock_fn):
    """
    Generic base-stock policy factory.
    Order = max(0, base_stock_level - net_inventory_position)
    base_stock = mean_demand * cover_period + safety_stock
    """
    def order_fn(states, order_num):
        orders = {}
        for key, state in states.items():
            s = sku_stats.get(key, {})
            mu = s.get('mean_demand', 0)
            ss = safety_stock_fn(s)
            cover = lead_time + R
            base_stock = mu * cover + ss
            net_inv = state.on_hand + sum(state.in_transit)
            order = max(0, int(round(base_stock - net_inv)))
            orders[key] = order
        return orders
    return order_fn


def policy_z_sigma_sqrtL(sku_stats, z: float):
    """Policy 1: z * sigma * sqrt(L)"""
    def ss_fn(s):
        return z * s.get('sigma', 0) * math.sqrt(L)
    return make_base_stock_policy(sku_stats, L, ss_fn)


def policy_z_sigma_sqrtLR(sku_stats, z: float):
    """Policy 2: z * sigma * sqrt(L+R)"""
    def ss_fn(s):
        return z * s.get('sigma', 0) * math.sqrt(L + R)
    return make_base_stock_policy(sku_stats, L, ss_fn)


def policy_z_rmse_sqrtLR(sku_stats, z: float):
    """Policy 3: z * RMSE * sqrt(L+R)"""
    def ss_fn(s):
        return z * s.get('rmse', 0) * math.sqrt(L + R)
    return make_base_stock_policy(sku_stats, L, ss_fn)


def policy_k_rmse(sku_stats, k: float):
    """Policy 4: k * RMSE"""
    def ss_fn(s):
        return k * s.get('rmse', 0)
    return make_base_stock_policy(sku_stats, L, ss_fn)


def policy_k_mae(sku_stats, k: float):
    """Policy 5: k * MAE"""
    def ss_fn(s):
        return k * s.get('mae', 0)
    return make_base_stock_policy(sku_stats, L, ss_fn)


def make_sip_policy(
    checkpoints_dir: Path,
    model: str,
    sl: float,
):
    """Policy 6: density-based SIP (newsvendor on quantile PMFs)."""
    from vn2.analyze.sequential_planner import Costs, choose_order_L3
    from vn2.analyze.sip_opt import quantiles_to_pmf

    co = EVAL_COSTS_HOLDING
    cu = co * sl / (1.0 - sl)
    costs = Costs(holding=co, shortage=cu)

    def load_q(store, product, fold_idx):
        path = checkpoints_dir / model / f"{store}_{product}" / f"fold_{fold_idx}.pkl"
        if not path.exists() and fold_idx != 0:
            path = checkpoints_dir / model / f"{store}_{product}" / "fold_0.pkl"
        if not path.exists():
            return None
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data.get('quantiles')

    def order_fn(states, order_num):
        fold_idx = order_num - 1
        orders = {}
        for key, state in states.items():
            qdf = load_q(state.store, state.product, fold_idx)
            if qdf is None or qdf.empty:
                orders[key] = 0
                continue
            if 3 not in qdf.index:
                if 2 in qdf.index:
                    qdf = qdf.copy()
                    qdf.loc[3] = qdf.loc[2].values
                else:
                    orders[key] = 0
                    continue
            try:
                h1 = quantiles_to_pmf(qdf.loc[1].values, QUANTILE_LEVELS, grain=500)
                h2 = quantiles_to_pmf(qdf.loc[2].values, QUANTILE_LEVELS, grain=500)
                h3 = quantiles_to_pmf(qdf.loc[3].values, QUANTILE_LEVELS, grain=500)
                q_opt, _ = choose_order_L3(
                    h1, h2, h3,
                    state.on_hand, state.in_transit[0],
                    state.in_transit[1], state.in_transit[2],
                    costs
                )
                orders[key] = int(q_opt)
            except Exception:
                orders[key] = 0
        return orders
    return order_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Safety stock policy comparison')
    parser.add_argument('--initial-state', type=Path,
                        default=Path('data/raw/Week 0 - 2024-04-08 - Initial State.csv'))
    parser.add_argument('--sales-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--demand-path', type=Path,
                        default=Path('data/processed/demand_long.parquet'))
    parser.add_argument('--bias-detail', type=Path,
                        default=Path('reports/bias/per_sku_week_detail.parquet'))
    parser.add_argument('--checkpoints-dir', type=Path,
                        default=Path('models/checkpoints_h3'))
    parser.add_argument('--sip-model', type=str, default='slurp_bootstrap')
    parser.add_argument('--sip-sl', type=float, default=0.833)
    parser.add_argument('--error-model', type=str, default='slurp_bootstrap',
                        help='Model to use for RMSE/MAE in policies 3-5')
    parser.add_argument('--output-dir', type=Path, default=Path('reports/safety_stock'))
    parser.add_argument('--max-weeks', type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Compute per-SKU statistics ---
    console.print("[bold]Computing per-SKU demand statistics...[/bold]")
    demand_stats = compute_sku_demand_stats(args.demand_path)
    console.print(f"  {len(demand_stats)} SKUs from demand_long")

    forecast_errors = compute_sku_forecast_errors(args.bias_detail, args.error_model)
    console.print(f"  {len(forecast_errors)} SKUs from bias detail ({args.error_model})")

    merged = demand_stats.merge(forecast_errors, on=['Store', 'Product'], how='left').fillna(0)
    sku_stats = {
        (int(r['Store']), int(r['Product'])): {
            'mean_demand': r['mean_demand'],
            'sigma': r['sigma'],
            'rmse': r['rmse'],
            'mae': r['mae'],
        }
        for _, r in merged.iterrows()
    }

    # --- z values for CSL targeting (90% CSL ≈ z=1.28) ---
    z_values = [0.5, 0.84, 1.0, 1.28, 1.64, 2.0]

    # --- k sweep values ---
    k_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    results = []

    # --- Policy 1: z * sigma * sqrt(L) ---
    console.print("\n[bold cyan]Policy 1: z * sigma * sqrt(L)[/bold cyan]")
    for z in z_values:
        fn = policy_z_sigma_sqrtL(sku_stats, z)
        r = run_simulation(args.initial_state, args.sales_dir, fn, args.max_weeks)
        console.print(f"  z={z:.2f}: €{r['total']:,.2f} (H={r['holding']:,.0f} S={r['shortage']:,.0f} FR={r['fill_rate']:.3f})")
        results.append({'policy': 'z_sigma_sqrtL', 'param': z, **{k: v for k, v in r.items() if k != 'weekly'}})

    # --- Policy 2: z * sigma * sqrt(L+R) ---
    console.print("\n[bold cyan]Policy 2: z * sigma * sqrt(L+R)[/bold cyan]")
    for z in z_values:
        fn = policy_z_sigma_sqrtLR(sku_stats, z)
        r = run_simulation(args.initial_state, args.sales_dir, fn, args.max_weeks)
        console.print(f"  z={z:.2f}: €{r['total']:,.2f} (H={r['holding']:,.0f} S={r['shortage']:,.0f} FR={r['fill_rate']:.3f})")
        results.append({'policy': 'z_sigma_sqrtLR', 'param': z, **{k: v for k, v in r.items() if k != 'weekly'}})

    # --- Policy 3: z * RMSE * sqrt(L+R) ---
    console.print("\n[bold cyan]Policy 3: z * RMSE * sqrt(L+R)[/bold cyan]")
    for z in z_values:
        fn = policy_z_rmse_sqrtLR(sku_stats, z)
        r = run_simulation(args.initial_state, args.sales_dir, fn, args.max_weeks)
        console.print(f"  z={z:.2f}: €{r['total']:,.2f} (H={r['holding']:,.0f} S={r['shortage']:,.0f} FR={r['fill_rate']:.3f})")
        results.append({'policy': 'z_RMSE_sqrtLR', 'param': z, **{k: v for k, v in r.items() if k != 'weekly'}})

    # --- Policy 4: k * RMSE ---
    console.print("\n[bold cyan]Policy 4: k * RMSE[/bold cyan]")
    for k in k_values:
        fn = policy_k_rmse(sku_stats, k)
        r = run_simulation(args.initial_state, args.sales_dir, fn, args.max_weeks)
        console.print(f"  k={k:.1f}: €{r['total']:,.2f} (H={r['holding']:,.0f} S={r['shortage']:,.0f} FR={r['fill_rate']:.3f})")
        results.append({'policy': 'k_RMSE', 'param': k, **{k_: v for k_, v in r.items() if k_ != 'weekly'}})

    # --- Policy 5: k * MAE ---
    console.print("\n[bold cyan]Policy 5: k * MAE[/bold cyan]")
    for k in k_values:
        fn = policy_k_mae(sku_stats, k)
        r = run_simulation(args.initial_state, args.sales_dir, fn, args.max_weeks)
        console.print(f"  k={k:.1f}: €{r['total']:,.2f} (H={r['holding']:,.0f} S={r['shortage']:,.0f} FR={r['fill_rate']:.3f})")
        results.append({'policy': 'k_MAE', 'param': k, **{k_: v for k_, v in r.items() if k_ != 'weekly'}})

    # --- Policy 6: Density-based SIP ---
    console.print(f"\n[bold cyan]Policy 6: Density SIP ({args.sip_model} @ SL={args.sip_sl})[/bold cyan]")
    sip_fn = make_sip_policy(args.checkpoints_dir, args.sip_model, args.sip_sl)
    r = run_simulation(args.initial_state, args.sales_dir, sip_fn, args.max_weeks)
    console.print(f"  SIP: €{r['total']:,.2f} (H={r['holding']:,.0f} S={r['shortage']:,.0f} FR={r['fill_rate']:.3f})")
    results.append({'policy': f'density_SIP_{args.sip_model}', 'param': args.sip_sl,
                    **{k: v for k, v in r.items() if k != 'weekly'}})

    # --- Build results DataFrame ---
    rdf = pd.DataFrame(results)
    rdf.to_csv(args.output_dir / 'policy_comparison.csv', index=False)

    # --- Find best per policy family ---
    console.print("\n" + "=" * 90)
    console.print("[bold green]Best result per policy family[/bold green]")
    console.print("=" * 90)

    table = Table(title="Safety Stock Policy Comparison")
    table.add_column("Policy", style="bold")
    table.add_column("Best Param", justify="right")
    table.add_column("Holding", justify="right")
    table.add_column("Shortage", justify="right")
    table.add_column("Total", justify="right", style="bold")
    table.add_column("Fill Rate", justify="right")
    table.add_column("Stockouts", justify="right")

    for policy in rdf['policy'].unique():
        pdf = rdf[rdf['policy'] == policy]
        best = pdf.loc[pdf['total'].idxmin()]
        table.add_row(
            policy,
            f"{best['param']:.2f}",
            f"€{best['holding']:,.2f}",
            f"€{best['shortage']:,.2f}",
            f"€{best['total']:,.2f}",
            f"{best['fill_rate']:.3f}",
            f"{int(best['stockouts'])}",
        )

    console.print(table)

    # --- Reference points ---
    console.print(f"\n{'Benchmark (ref)':<30} €5,247.80")
    console.print(f"{'Winner (ref)':<30} €4,677.00")
    console.print(f"{'Our best SIP (ref)':<30} €5,086.60 (slurp_bootstrap static @ 0.833)")

    best_overall = rdf.loc[rdf['total'].idxmin()]
    console.print(f"\n[bold]Overall best:[/bold] {best_overall['policy']} param={best_overall['param']:.2f} → €{best_overall['total']:,.2f}")

    # --- Save markdown summary ---
    md_lines = ["# Safety Stock Policy Comparison\n"]
    md_lines.append(f"Simulation: 8 weeks, L={L}, R={R}, cu={EVAL_COSTS_SHORTAGE}, co={EVAL_COSTS_HOLDING}\n")
    md_lines.append("## Best per policy family\n")
    md_lines.append("| Policy | Best Param | Holding | Shortage | Total | Fill Rate | Stockouts |")
    md_lines.append("|--------|----------:|--------:|---------:|------:|----------:|----------:|")
    for policy in rdf['policy'].unique():
        pdf = rdf[rdf['policy'] == policy]
        b = pdf.loc[pdf['total'].idxmin()]
        md_lines.append(
            f"| {policy} | {b['param']:.2f} | €{b['holding']:,.2f} | €{b['shortage']:,.2f} "
            f"| €{b['total']:,.2f} | {b['fill_rate']:.3f} | {int(b['stockouts'])} |"
        )
    md_lines.append(f"\n## Reference points\n")
    md_lines.append(f"- Benchmark: €5,247.80")
    md_lines.append(f"- Winner: €4,677.00")
    md_lines.append(f"- Our best SIP: €5,086.60 (slurp_bootstrap static @ 0.833)")
    md_lines.append(f"\n## Full sweep results\n")
    md_lines.append(rdf.to_markdown(index=False))

    with open(args.output_dir / 'policy_comparison.md', 'w') as f:
        f.write('\n'.join(md_lines))

    console.print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
