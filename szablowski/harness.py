"""
Unified L=3 comparison harness with PolicyAdapter abstraction.

Runs both SIP density and Szabłowski analytical policies through the SAME
simulation loop with identical inventory dynamics, initial state, and
realized demand.  Produces per-SKU-week cost output joined to EDA artifacts
for hypothesis-test stratification.

Lead time semantics (CRITICAL — do not change):
  Order placed END of week t → arrives START of week t+3.
  State has 3 transit slots: in_transit[0] arriving next week,
  in_transit[1] in 2 weeks, in_transit[2] in 3 weeks.
"""

import abc
import argparse
import json
import pickle
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# State and cost dataclasses (mirroring full_L3_simulation.py)
# ---------------------------------------------------------------------------

@dataclass
class SKUState:
    store: int
    product: int
    on_hand: int
    in_transit: List[int] = field(default_factory=lambda: [0, 0, 0])

    def copy(self) -> "SKUState":
        return SKUState(self.store, self.product, self.on_hand, list(self.in_transit))


@dataclass
class CostParams:
    holding: float = 0.2
    shortage: float = 1.0


# ---------------------------------------------------------------------------
# Policy adapter ABC
# ---------------------------------------------------------------------------

class PolicyAdapter(abc.ABC):
    """Abstract base for order-generation policies."""

    @abc.abstractmethod
    def generate_order(
        self,
        state: SKUState,
        order_number: int,
        fold_idx: int,
        costs: CostParams,
    ) -> int:
        """Return non-negative integer order quantity for a single SKU."""
        ...

    @abc.abstractmethod
    def name(self) -> str:
        ...


# ---------------------------------------------------------------------------
# SIP density policy (existing pipeline)
# ---------------------------------------------------------------------------

class SIPPolicy(PolicyAdapter):
    """Loads quantile checkpoints, builds PMFs, calls choose_order_L3."""

    def __init__(
        self,
        checkpoints_dir: Path,
        selector_map: Dict[Tuple[int, int], str],
        default_model: str = "seasonal_naive",
        sip_grain: int = 500,
    ):
        self.checkpoints_dir = checkpoints_dir
        self.selector_map = selector_map
        self.default_model = default_model
        self.sip_grain = sip_grain
        self.quantile_levels = np.array([
            0.01, 0.05, 0.10, 0.20, 0.30, 0.40,
            0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99,
        ])

        from vn2.analyze.sequential_planner import Costs as PlannerCosts, choose_order_L3
        from vn2.analyze.sip_opt import quantiles_to_pmf
        self._choose_order_L3 = choose_order_L3
        self._quantiles_to_pmf = quantiles_to_pmf
        self._PlannerCosts = PlannerCosts

    def name(self) -> str:
        return "sip"

    def generate_order(
        self, state: SKUState, order_number: int, fold_idx: int, costs: CostParams,
    ) -> int:
        model = self.selector_map.get((state.store, state.product), self.default_model)
        qdf = self._load_quantiles(state.store, state.product, model, fold_idx)
        if qdf is None or qdf.empty:
            return 0

        if 3 not in qdf.index:
            if 2 in qdf.index:
                qdf = qdf.copy()
                qdf.loc[3] = qdf.loc[2].values
            else:
                return 0

        h1_pmf = self._quantiles_to_pmf(qdf.loc[1].values, self.quantile_levels, grain=self.sip_grain)
        h2_pmf = self._quantiles_to_pmf(qdf.loc[2].values, self.quantile_levels, grain=self.sip_grain)
        h3_pmf = self._quantiles_to_pmf(qdf.loc[3].values, self.quantile_levels, grain=self.sip_grain)

        planner_costs = self._PlannerCosts(holding=costs.holding, shortage=costs.shortage)
        try:
            q_opt, _ = self._choose_order_L3(
                h1_pmf, h2_pmf, h3_pmf,
                state.on_hand,
                state.in_transit[0], state.in_transit[1], state.in_transit[2],
                planner_costs,
            )
            return int(q_opt)
        except Exception:
            return 0

    def _load_quantiles(self, store, product, model, fold_idx):
        path = self.checkpoints_dir / model / f"{store}_{product}" / f"fold_{fold_idx}.pkl"
        if not path.exists() and fold_idx != 0:
            path = self.checkpoints_dir / model / f"{store}_{product}" / "fold_0.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data.get("quantiles")


# ---------------------------------------------------------------------------
# Analytical policy (Szabłowski point + φ√D buffer)
# ---------------------------------------------------------------------------

class AnalyticalPolicy(PolicyAdapter):
    """Szabłowski's newsvendor with point forecast and φ√D uncertainty proxy."""

    def __init__(
        self,
        forecasts: Dict[Tuple[int, int, int], Tuple[float, float, float]],
        phi: float = 1.0,
        cs: float = 1.0,
        ch: float = 0.2,
    ):
        """
        Parameters
        ----------
        forecasts : {(store, product, order_number): (d_h1, d_h2, d_h3)}
        phi : calibrated uncertainty parameter
        """
        self.forecasts = forecasts
        self.phi = phi
        from szablowski.policy import PolicyParams, compute_order
        self._params = PolicyParams(cs=cs, ch=ch, phi=phi)
        self._compute_order = compute_order

    def name(self) -> str:
        return "analytical"

    def generate_order(
        self, state: SKUState, order_number: int, fold_idx: int, costs: CostParams,
    ) -> int:
        fc = self.forecasts.get((state.store, state.product, order_number))
        if fc is None:
            return 0
        self._params.cs = costs.shortage
        self._params.ch = costs.holding
        return self._compute_order(
            fc, state.on_hand, tuple(state.in_transit), self._params,
        )


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def load_initial_states(path: Path) -> Dict[Tuple[int, int], SKUState]:
    df = pd.read_csv(path)
    states = {}
    for _, row in df.iterrows():
        store = int(row["Store"])
        product = int(row["Product"])
        on_hand = int(row.get("End Inventory", row.get("Start Inventory", 0)))
        it1 = int(row.get("In Transit W+1", 0))
        it2 = int(row.get("In Transit W+2", 0))
        states[(store, product)] = SKUState(store, product, on_hand, [it1, it2, 0])
    return states


def load_weekly_sales(sales_dir: Path, week: int) -> Dict[Tuple[int, int], int]:
    """Load actual sales for a specific competition week."""
    week_info = {
        1: ("Week 1 - 2024-04-15 - Sales.csv", "2024-04-15"),
        2: ("Week 2 - 2024-04-22 - Sales.csv", "2024-04-22"),
        3: ("Week 3 - 2024-04-29 - Sales.csv", "2024-04-29"),
        4: ("Week 4 - 2024-05-06 - Sales.csv", "2024-05-06"),
        5: ("Week 5 - 2024-05-13 - Sales.csv", "2024-05-13"),
        6: ("Week 6 - 2024-05-20 - Sales.csv", "2024-05-20"),
        7: ("Week 7 - 2024-05-27 - Sales.csv", "2024-05-27"),
        8: ("Week 8 - 2024-06-03 - Sales.csv", "2024-06-03"),
    }
    if week not in week_info:
        return {}
    filename, date_col = week_info[week]
    path = sales_dir / filename
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return {(int(r["Store"]), int(r["Product"])): int(r[date_col]) for _, r in df.iterrows()}


def simulate_week_l3(
    states: Dict[Tuple[int, int], SKUState],
    sales: Dict[Tuple[int, int], int],
    costs: CostParams,
    week_num: int,
    sku_records: Optional[List[dict]] = None,
) -> Tuple[float, float, int]:
    """One week of L=3 inventory dynamics.

    Timeline for week W:
      START of W: in_transit[0] arrives
      DURING W: demand served from on_hand + arrivals
      END of W: state updated, transit pipeline shifted
    """
    total_holding = 0.0
    total_shortage = 0.0
    stockouts = 0

    for key, state in states.items():
        arriving = state.in_transit[0]
        available = state.on_hand + arriving
        demand = sales.get(key, 0)
        sold = min(available, demand)
        shortage_qty = max(0, demand - available)
        leftover = max(0, available - demand)

        h_cost = costs.holding * leftover
        s_cost = costs.shortage * shortage_qty
        total_holding += h_cost
        total_shortage += s_cost
        if shortage_qty > 0:
            stockouts += 1

        if sku_records is not None:
            sku_records.append({
                "week": week_num,
                "Store": key[0],
                "Product": key[1],
                "on_hand_start": state.on_hand,
                "arriving": arriving,
                "available": available,
                "demand": demand,
                "sold": sold,
                "shortage": shortage_qty,
                "leftover": leftover,
                "holding_cost": h_cost,
                "shortage_cost": s_cost,
            })

        state.on_hand = leftover
        state.in_transit = [state.in_transit[1], state.in_transit[2], 0]

    return total_holding, total_shortage, stockouts


def run_comparison(
    policies: List[PolicyAdapter],
    initial_state_path: Path,
    sales_dir: Path,
    costs: CostParams = CostParams(),
    max_weeks: int = 8,
    static_folds: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Run the same simulation for each policy; return per-SKU-week detail.

    All policies see identical initial states and realized demand — the only
    difference is how orders are generated.
    """
    base_states = load_initial_states(initial_state_path)
    all_skus = list(base_states.keys())

    results = {}

    for policy in policies:
        pname = policy.name()
        console.print(f"\n[bold cyan]Running policy: {pname}[/bold cyan]")

        states = {k: v.copy() for k, v in base_states.items()}
        sku_records: List[dict] = []
        weekly_summary = []

        # Place Order 1 at end of Week 0
        for key in all_skus:
            order = policy.generate_order(states[key], order_number=1, fold_idx=0, costs=costs)
            states[key].in_transit[2] = order

        for week in range(1, max_weeks + 1):
            sales = load_weekly_sales(sales_dir, week)

            h, s, stockouts = simulate_week_l3(states, sales, costs, week, sku_records)
            total = h + s

            console.print(f"  Week {week}: H={h:.1f}  S={s:.1f}  Total={total:.1f}  Stockouts={stockouts}")

            next_order_num = week + 1
            if next_order_num <= 6:
                fold = 0 if static_folds else week
                for key in all_skus:
                    order = policy.generate_order(
                        states[key], order_number=next_order_num, fold_idx=fold, costs=costs,
                    )
                    states[key].in_transit[2] = order

            weekly_summary.append({
                "week": week, "holding": h, "shortage": s, "total": total, "stockouts": stockouts,
            })

        detail = pd.DataFrame(sku_records)
        detail["policy"] = pname

        summary = pd.DataFrame(weekly_summary)
        total_cost = summary["total"].sum()
        console.print(f"  [bold]Total cost ({pname}): {total_cost:.2f}[/bold]")

        results[pname] = detail

    return results


# ---------------------------------------------------------------------------
# EDA join for stratification
# ---------------------------------------------------------------------------

def join_eda_artifacts(
    detail: pd.DataFrame,
    processed_dir: Path,
) -> pd.DataFrame:
    """Merge per-SKU EDA columns into the per-SKU-week detail."""
    out = detail.copy()

    for fname in [
        "summary_statistics.parquet",
        "stationarity_tests.parquet",
        "summary_statistics_ext.parquet",
        "stationarity_tests_ext.parquet",
    ]:
        path = processed_dir / fname
        if not path.exists():
            continue
        eda = pd.read_parquet(path)
        merge_cols = [c for c in eda.columns if c not in ("Store", "Product")]
        # Avoid duplicate columns
        existing = set(out.columns)
        new_cols = [c for c in merge_cols if c not in existing]
        if new_cols:
            out = out.merge(eda[["Store", "Product"] + new_cols], on=["Store", "Product"], how="left")

    # Cohort features
    cohort_path = Path("models/results/cohort_features_temp.parquet")
    if cohort_path.exists():
        cohort = pd.read_parquet(cohort_path)
        if "store" in cohort.columns:
            cohort = cohort.rename(columns={"store": "Store", "product": "Product"})
        merge_cols = [c for c in cohort.columns if c not in ("Store", "Product") and c not in out.columns]
        if merge_cols:
            out = out.merge(cohort[["Store", "Product"] + merge_cols], on=["Store", "Product"], how="left")

    return out


# ---------------------------------------------------------------------------
# Summary and comparison
# ---------------------------------------------------------------------------

def compare_policies(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Produce a head-to-head cost comparison across policies."""
    summaries = []
    for pname, detail in results.items():
        total_h = detail["holding_cost"].sum()
        total_s = detail["shortage_cost"].sum()
        summaries.append({
            "policy": pname,
            "total_holding": total_h,
            "total_shortage": total_s,
            "total_cost": total_h + total_s,
            "n_stockout_events": int((detail["shortage"] > 0).sum()),
        })

    comp = pd.DataFrame(summaries)

    table = Table(title="Policy Comparison")
    table.add_column("Policy")
    table.add_column("Holding", justify="right")
    table.add_column("Shortage", justify="right")
    table.add_column("Total Cost", justify="right")
    table.add_column("Stockout Events", justify="right")

    for _, row in comp.iterrows():
        table.add_row(
            row["policy"],
            f"{row['total_holding']:.2f}",
            f"{row['total_shortage']:.2f}",
            f"{row['total_cost']:.2f}",
            str(row["n_stockout_events"]),
        )
    console.print(table)

    return comp


def per_sku_cost_delta(
    results: Dict[str, pd.DataFrame],
    baseline: str = "sip",
    comparison: str = "analytical",
) -> pd.DataFrame:
    """Compute per-SKU cost delta: comparison - baseline."""
    if baseline not in results or comparison not in results:
        return pd.DataFrame()

    base = results[baseline].groupby(["Store", "Product"]).agg(
        base_holding=("holding_cost", "sum"),
        base_shortage=("shortage_cost", "sum"),
    ).reset_index()
    base["base_total"] = base["base_holding"] + base["base_shortage"]

    comp = results[comparison].groupby(["Store", "Product"]).agg(
        comp_holding=("holding_cost", "sum"),
        comp_shortage=("shortage_cost", "sum"),
    ).reset_index()
    comp["comp_total"] = comp["comp_holding"] + comp["comp_shortage"]

    merged = base.merge(comp, on=["Store", "Product"], how="outer")
    merged["cost_delta"] = merged["comp_total"] - merged["base_total"]
    merged["holding_delta"] = merged["comp_holding"] - merged["base_holding"]
    merged["shortage_delta"] = merged["comp_shortage"] - merged["base_shortage"]

    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified L=3 comparison harness")
    parser.add_argument("--initial-state", type=Path,
                        default=Path("data/raw/Week 0 - 2024-04-08 - Initial State.csv"))
    parser.add_argument("--sales-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("models/checkpoints_h3"))
    parser.add_argument("--selector-map", type=Path,
                        default=Path("reports/dynamic_selector/static_composite_selector.parquet"))
    parser.add_argument("--szablowski-forecasts", type=Path, default=None,
                        help="Parquet with CatBoost forecasts (Store, Product, order_number, h1, h2, h3)")
    parser.add_argument("--phi", type=float, default=None,
                        help="φ parameter for analytical policy (reads best_phi.json if omitted)")
    parser.add_argument("--szablowski-dir", type=Path, default=Path("models/szablowski"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/comparison"))
    parser.add_argument("--max-weeks", type=int, default=8)
    parser.add_argument("--cu", type=float, default=1.0)
    parser.add_argument("--co", type=float, default=0.2)
    parser.add_argument("--default-model", type=str, default="seasonal_naive")
    parser.add_argument("--static-folds", action="store_true")
    parser.add_argument("--policy", type=str, nargs="+", default=["sip", "analytical"],
                        choices=["sip", "analytical"],
                        help="Which policies to run")
    args = parser.parse_args()

    costs = CostParams(holding=args.co, shortage=args.cu)
    policies: List[PolicyAdapter] = []

    if "sip" in args.policy:
        selector_map = {}
        if args.selector_map.exists():
            sel_df = pd.read_parquet(args.selector_map)
            selector_map = {
                (int(r.store), int(r.product)): r.model_name
                for r in sel_df.itertuples(index=False)
            }
        policies.append(SIPPolicy(args.checkpoints_dir, selector_map, args.default_model))

    if "analytical" in args.policy:
        phi = args.phi
        if phi is None:
            phi_path = args.szablowski_dir / "best_phi.json"
            if phi_path.exists():
                with open(phi_path) as f:
                    phi = json.load(f)["best_phi"]
            else:
                phi = 1.0
                console.print(f"[yellow]Warning: no best_phi.json found, using φ={phi}[/yellow]")

        # Load forecasts
        forecasts: Dict[Tuple[int, int, int], Tuple[float, float, float]] = {}
        fc_path = args.szablowski_forecasts or args.szablowski_dir / "forecasts.parquet"
        if fc_path.exists():
            fc_df = pd.read_parquet(fc_path)
            order_col = "order_number" if "order_number" in fc_df.columns else "week"
            for _, row in fc_df.iterrows():
                key = (int(row["Store"]), int(row["Product"]), int(row[order_col]))
                forecasts[key] = (float(row["h1"]), float(row["h2"]), float(row["h3"]))
        else:
            console.print(f"[red]Error: Szabłowski forecasts not found at {fc_path}[/red]")
            return

        policies.append(AnalyticalPolicy(forecasts, phi=phi, cs=args.cu, ch=args.co))

    if not policies:
        console.print("[red]No policies selected[/red]")
        return

    results = run_comparison(
        policies, args.initial_state, args.sales_dir, costs,
        max_weeks=args.max_weeks, static_folds=args.static_folds,
    )

    comp = compare_policies(results)

    # Join EDA artifacts and save
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for pname, detail in results.items():
        enriched = join_eda_artifacts(detail, Path("data/processed"))
        enriched.to_parquet(args.output_dir / f"sku_detail_{pname}.parquet", index=False)

    comp.to_csv(args.output_dir / "policy_comparison.csv", index=False)

    if len(results) >= 2:
        policy_names = list(results.keys())
        delta = per_sku_cost_delta(results, baseline=policy_names[0], comparison=policy_names[1])
        delta = join_eda_artifacts(delta, Path("data/processed"))
        delta.to_parquet(args.output_dir / "per_sku_cost_delta.parquet", index=False)
        console.print(f"\nPer-SKU cost delta saved to {args.output_dir / 'per_sku_cost_delta.parquet'}")

    console.print(f"\n[green]All results saved to {args.output_dir}[/green]")


if __name__ == "__main__":
    main()
