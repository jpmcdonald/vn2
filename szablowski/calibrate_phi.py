"""
Calibrate the φ parameter by grid-searching over the validation window.

Runs the analytical newsvendor policy through the L=3 simulation for a range
of φ values and selects the one that minimizes total cost (holding + shortage).
Also produces the cost-vs-φ sensitivity data that feeds H5.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from szablowski.policy import PolicyParams, compute_order


# ---------------------------------------------------------------------------
# Lightweight L=3 simulation for phi calibration
# ---------------------------------------------------------------------------

def simulate_with_phi(
    forecasts_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    initial_states: Dict[Tuple[int, int], dict],
    phi: float,
    cs: float = 1.0,
    ch: float = 0.2,
    n_weeks: int = 8,
) -> Tuple[float, float, float, pd.DataFrame]:
    """Run a simplified L=3 simulation using the analytical policy at a given φ.

    Parameters
    ----------
    forecasts_df : DataFrame
        Columns: Store, Product, week (int, 1-indexed decision week), h1, h2, h3.
    actuals_df : DataFrame
        Columns: Store, Product, week (int, 1-indexed), actual_demand.
    initial_states : {(store, product): {"on_hand": int, "in_transit": [q1, q2, q3]}}
    phi : float
    cs, ch : cost parameters
    n_weeks : number of simulation weeks

    Returns
    -------
    total_holding, total_shortage, total_cost, sku_detail_df
    """
    params = PolicyParams(cs=cs, ch=ch, phi=phi)

    # Copy state so we don't mutate
    states = {
        k: {"on_hand": v["on_hand"], "in_transit": list(v["in_transit"])}
        for k, v in initial_states.items()
    }

    all_skus = list(states.keys())
    records = []

    # Build lookup: (store, product, week) -> (h1, h2, h3)
    fc_lookup = {}
    for _, row in forecasts_df.iterrows():
        key = (int(row["Store"]), int(row["Product"]), int(row["week"]))
        fc_lookup[key] = (float(row["h1"]), float(row["h2"]), float(row["h3"]))

    act_lookup = {}
    for _, row in actuals_df.iterrows():
        key = (int(row["Store"]), int(row["Product"]), int(row["week"]))
        act_lookup[key] = float(row["actual_demand"])

    # Place initial order (order 1) at end of week 0
    for sku in all_skus:
        fc = fc_lookup.get((sku[0], sku[1], 1))
        if fc is None:
            fc = (0, 0, 0)
        order = compute_order(fc, states[sku]["on_hand"], tuple(states[sku]["in_transit"]), params)
        states[sku]["in_transit"][2] = order

    for week in range(1, n_weeks + 1):
        for sku in all_skus:
            st = states[sku]
            arriving = st["in_transit"][0]
            available = st["on_hand"] + arriving
            demand = act_lookup.get((sku[0], sku[1], week), 0)

            sold = min(available, demand)
            shortage = max(0, demand - available)
            leftover = max(0, available - demand)

            h_cost = ch * leftover
            s_cost = cs * shortage

            records.append({
                "Store": sku[0],
                "Product": sku[1],
                "week": week,
                "demand": demand,
                "available": available,
                "sold": sold,
                "shortage": shortage,
                "leftover": leftover,
                "holding_cost": h_cost,
                "shortage_cost": s_cost,
            })

            st["on_hand"] = leftover
            st["in_transit"] = [st["in_transit"][1], st["in_transit"][2], 0]

            # Place next order (if within order window)
            next_order_week = week + 1
            if next_order_week <= 6:
                fc = fc_lookup.get((sku[0], sku[1], next_order_week))
                if fc is None:
                    fc = (0, 0, 0)
                order = compute_order(fc, st["on_hand"], tuple(st["in_transit"]), params)
                st["in_transit"][2] = order

    detail = pd.DataFrame(records)
    total_holding = detail["holding_cost"].sum()
    total_shortage = detail["shortage_cost"].sum()
    total_cost = total_holding + total_shortage

    return total_holding, total_shortage, total_cost, detail


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def calibrate_phi(
    forecasts_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    initial_states: Dict[Tuple[int, int], dict],
    phi_range: Optional[np.ndarray] = None,
    cs: float = 1.0,
    ch: float = 0.2,
    n_weeks: int = 8,
) -> Tuple[float, pd.DataFrame]:
    """Grid-search φ to minimize total cost.

    Returns (best_phi, results_df) where results_df has columns:
    phi, holding, shortage, total_cost.
    """
    if phi_range is None:
        phi_range = np.arange(0.0, 3.01, 0.05)

    results = []
    for phi in phi_range:
        h, s, total, _ = simulate_with_phi(
            forecasts_df, actuals_df, initial_states, phi, cs, ch, n_weeks
        )
        results.append({"phi": float(phi), "holding": h, "shortage": s, "total_cost": total})
        print(f"  φ={phi:.2f}  →  H={h:.1f}  S={s:.1f}  Total={total:.1f}")

    results_df = pd.DataFrame(results)
    best_idx = results_df["total_cost"].idxmin()
    best_phi = results_df.loc[best_idx, "phi"]

    print(f"\nBest φ = {best_phi:.2f}  (Total cost = {results_df.loc[best_idx, 'total_cost']:.1f})")
    return float(best_phi), results_df


def main():
    parser = argparse.ArgumentParser(description="Calibrate φ for Szabłowski analytical policy")
    parser.add_argument("--forecasts", type=Path, required=True,
                        help="Parquet with Store, Product, week, h1, h2, h3")
    parser.add_argument("--actuals", type=Path, required=True,
                        help="Parquet with Store, Product, week, actual_demand")
    parser.add_argument("--initial-state", type=Path, required=True,
                        help="CSV with initial inventory state (Week 0)")
    parser.add_argument("--output-dir", type=Path, default=Path("models/szablowski"))
    parser.add_argument("--phi-min", type=float, default=0.0)
    parser.add_argument("--phi-max", type=float, default=3.0)
    parser.add_argument("--phi-step", type=float, default=0.05)
    parser.add_argument("--cu", type=float, default=1.0)
    parser.add_argument("--co", type=float, default=0.2)
    parser.add_argument("--n-weeks", type=int, default=8)
    args = parser.parse_args()

    forecasts_df = pd.read_parquet(args.forecasts)
    actuals_df = pd.read_parquet(args.actuals)

    # Load initial state
    state_csv = pd.read_csv(args.initial_state)
    initial_states = {}
    for _, row in state_csv.iterrows():
        store = int(row["Store"])
        product = int(row["Product"])
        on_hand = int(row.get("End Inventory", row.get("Start Inventory", 0)))
        it1 = int(row.get("In Transit W+1", 0))
        it2 = int(row.get("In Transit W+2", 0))
        initial_states[(store, product)] = {
            "on_hand": on_hand,
            "in_transit": [it1, it2, 0],
        }

    phi_range = np.arange(args.phi_min, args.phi_max + args.phi_step / 2, args.phi_step)

    best_phi, results_df = calibrate_phi(
        forecasts_df, actuals_df, initial_states,
        phi_range=phi_range,
        cs=args.cu, ch=args.co, n_weeks=args.n_weeks,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(args.output_dir / "phi_calibration.parquet", index=False)
    with open(args.output_dir / "best_phi.json", "w") as f:
        json.dump({"best_phi": best_phi, "total_cost": float(results_df.loc[results_df["total_cost"].idxmin(), "total_cost"])}, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/phi_calibration.parquet")


if __name__ == "__main__":
    main()
