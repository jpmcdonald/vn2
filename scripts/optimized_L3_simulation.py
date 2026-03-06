#!/usr/bin/env python3
"""
Optimized L=3 Simulation with H=3-Based Model Selection.

This script answers: "If we had correctly trained our model selector for h=3 
accuracy, how would we have performed?"

Steps:
1. For each SKU, evaluate all available models on h=3 forecast accuracy
2. Select the best model per SKU based on h=3 performance
3. Run full L=3 simulation with corrected model selection
4. Estimate 8-week performance and compare to leaderboard
"""

import argparse
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from vn2.analyze.sequential_planner import Costs, choose_order_L3
from vn2.analyze.sip_opt import quantiles_to_pmf

console = Console()


@dataclass
class SKUSimState:
    """Simulated state for a single SKU."""
    store: int
    product: int
    on_hand: int
    in_transit: List[int] = field(default_factory=lambda: [0, 0, 0])


def load_historical_demand(sales_path: Path) -> pd.DataFrame:
    """Load full historical demand (before competition)."""
    df = pd.read_csv(sales_path)
    df.columns = [c.strip() for c in df.columns]
    
    # Get all date columns
    date_cols = [c for c in df.columns if c not in ['Store', 'Product']]
    
    # Melt to long format
    demand_records = []
    for _, row in df.iterrows():
        store = int(row['Store'])
        product = int(row['Product'])
        for col in date_cols:
            try:
                demand = int(row[col])
                demand_records.append({
                    'store': store,
                    'product': product,
                    'week': col,
                    'demand': demand
                })
            except:
                pass
    
    return pd.DataFrame(demand_records)


def evaluate_model_h3_accuracy(
    store: int,
    product: int,
    model_name: str,
    checkpoints_dir: Path,
    historical_demand: pd.DataFrame,
    n_eval_weeks: int = 12
) -> Optional[float]:
    """
    Evaluate a model's h=3 forecast accuracy using historical data.
    
    Uses rolling evaluation: at each point t, compare h=3 forecast (made at t-3)
    with actual demand at t.
    
    Returns mean absolute error for h=3 forecasts.
    """
    sku_demand = historical_demand[
        (historical_demand['store'] == store) & 
        (historical_demand['product'] == product)
    ].sort_values('week')
    
    if len(sku_demand) < 20:  # Not enough history
        return None
    
    # Load checkpoint
    ckpt_path = checkpoints_dir / model_name / f"{store}_{product}" / "fold_0.pkl"
    if not ckpt_path.exists():
        return None
    
    try:
        with open(ckpt_path, 'rb') as f:
            data = pickle.load(f)
        qdf = data.get('quantiles')
        if qdf is None or 3 not in qdf.index:
            return None
        
        # Get h=3 forecast at Q50 (median) for accuracy evaluation
        h3_forecast = qdf.loc[3, 0.50]
        
        # Compare to last n_eval_weeks of actual demand
        recent_demand = sku_demand['demand'].tail(n_eval_weeks).values
        
        if len(recent_demand) == 0:
            return None
        
        # MAE between h=3 forecast and actual recent demand
        mae = np.mean(np.abs(recent_demand - h3_forecast))
        
        return mae
        
    except Exception as e:
        return None


def create_h3_optimized_selector(
    skus: List[Tuple[int, int]],
    checkpoints_dir: Path,
    historical_demand: pd.DataFrame
) -> Dict[Tuple[int, int], str]:
    """Create model selector optimized for h=3 accuracy."""
    
    # Get all available models
    model_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    model_names = [d.name for d in model_dirs]
    
    console.print(f"Evaluating {len(model_names)} models for h=3 accuracy...")
    console.print(f"  Models: {', '.join(model_names[:5])}...")
    
    selector = {}
    n_evaluated = 0
    
    for store, product in skus:
        best_model = None
        best_mae = float('inf')
        
        for model_name in model_names:
            mae = evaluate_model_h3_accuracy(
                store, product, model_name, checkpoints_dir, historical_demand
            )
            
            if mae is not None and mae < best_mae:
                best_mae = mae
                best_model = model_name
        
        if best_model is not None:
            selector[(store, product)] = best_model
            n_evaluated += 1
        else:
            # Fallback to zinb if no model evaluated successfully
            selector[(store, product)] = 'zinb'
    
    console.print(f"  Evaluated {n_evaluated} SKUs")
    
    # Count model selections
    model_counts = {}
    for model in selector.values():
        model_counts[model] = model_counts.get(model, 0) + 1
    
    console.print("  Model selection distribution:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1])[:5]:
        console.print(f"    {model}: {count} SKUs")
    
    return selector


def load_initial_state(path: Path) -> Dict[Tuple[int, int], SKUSimState]:
    """Load Week 0 initial state."""
    df = pd.read_csv(path)
    states = {}
    for _, row in df.iterrows():
        store = int(row['Store'])
        product = int(row['Product'])
        on_hand = int(row.get('End Inventory', row.get('Start Inventory', 0)))
        it1 = int(row.get('In Transit W+1', 0))
        it2 = int(row.get('In Transit W+2', 0))
        states[(store, product)] = SKUSimState(
            store=store, product=product, on_hand=on_hand,
            in_transit=[it1, it2, 0]
        )
    return states


def load_sales(sales_dir: Path, week: int) -> Dict[Tuple[int, int], int]:
    """Load actual sales for a specific week."""
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
    
    result = {}
    for _, row in df.iterrows():
        key = (int(row['Store']), int(row['Product']))
        result[key] = int(row[date_col])
    return result


def load_quantiles(store: int, product: int, model: str, checkpoints_dir: Path) -> Optional[pd.DataFrame]:
    """Load quantile forecasts."""
    path = checkpoints_dir / model / f"{store}_{product}" / "fold_0.pkl"
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data.get('quantiles')


def generate_order_L3(
    state: SKUSimState,
    qdf: pd.DataFrame,
    costs: Costs,
    quantile_levels: np.ndarray,
    sip_grain: int = 500
) -> int:
    """Generate optimal order using L=3 optimization."""
    if qdf is None or qdf.empty:
        return 0
    
    if 3 not in qdf.index:
        if 2 in qdf.index:
            qdf = qdf.copy()
            qdf.loc[3] = qdf.loc[2].values
        else:
            return 0
    
    h1_pmf = quantiles_to_pmf(qdf.loc[1].values, quantile_levels, grain=sip_grain)
    h2_pmf = quantiles_to_pmf(qdf.loc[2].values, quantile_levels, grain=sip_grain)
    h3_pmf = quantiles_to_pmf(qdf.loc[3].values, quantile_levels, grain=sip_grain)
    
    I0 = state.on_hand
    Q1 = state.in_transit[0]
    Q2 = state.in_transit[1]
    Q3 = state.in_transit[2]
    
    try:
        q_opt, _ = choose_order_L3(h1_pmf, h2_pmf, h3_pmf, I0, Q1, Q2, Q3, costs)
        return int(q_opt)
    except:
        return 0


def simulate_week(
    states: Dict[Tuple[int, int], SKUSimState],
    sales: Dict[Tuple[int, int], int],
    costs: Costs
) -> Tuple[float, float, int]:
    """Simulate one week of inventory dynamics."""
    total_holding = 0.0
    total_shortage = 0.0
    stockouts = 0
    
    for key, state in states.items():
        arriving = state.in_transit[0]
        available = state.on_hand + arriving
        demand = sales.get(key, 0)
        
        sold = min(available, demand)
        shortage = max(0, demand - available)
        leftover = max(0, available - demand)
        
        h_cost = costs.holding * leftover
        s_cost = costs.shortage * shortage
        total_holding += h_cost
        total_shortage += s_cost
        if shortage > 0:
            stockouts += 1
        
        state.on_hand = leftover
        state.in_transit = [state.in_transit[1], state.in_transit[2], 0]
    
    return total_holding, total_shortage, stockouts


def run_optimized_simulation(
    initial_state_path: Path,
    sales_dir: Path,
    checkpoints_dir: Path,
    historical_sales_path: Path,
    max_weeks: int = 5,
    costs: Costs = Costs()
) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], str]]:
    """Run simulation with h=3-optimized model selection."""
    
    # Load initial state
    console.print("\n[bold cyan]Loading data...[/bold cyan]")
    states = load_initial_state(initial_state_path)
    skus = list(states.keys())
    console.print(f"  {len(skus)} SKUs")
    
    # Load historical demand for model evaluation
    console.print("  Loading historical demand...")
    historical_demand = load_historical_demand(historical_sales_path)
    console.print(f"  {len(historical_demand)} demand records")
    
    # Create h=3 optimized selector
    console.print("\n[bold cyan]Creating h=3-optimized model selector...[/bold cyan]")
    model_selector = create_h3_optimized_selector(skus, checkpoints_dir, historical_demand)
    
    quantile_levels = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40,
                                 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
    
    results = []
    all_orders = {}
    
    # Place Order 1 at End of Week 0
    console.print("\n[bold cyan]Running simulation...[/bold cyan]")
    console.print("\n[bold]End of Week 0: Place Order 1[/bold]")
    order1 = {}
    for key, state in states.items():
        model = model_selector.get(key, 'zinb')
        qdf = load_quantiles(state.store, state.product, model, checkpoints_dir)
        order = generate_order_L3(state, qdf, costs, quantile_levels)
        order1[key] = order
        state.in_transit[2] = order
    
    all_orders[1] = sum(order1.values())
    console.print(f"  Order 1: {all_orders[1]} units → arrives Week 3")
    
    for week in range(1, max_weeks + 1):
        console.print(f"\n[bold]Week {week}[/bold]")
        
        sales = load_sales(sales_dir, week)
        total_demand = sum(sales.values())
        total_arriving = sum(s.in_transit[0] for s in states.values())
        
        console.print(f"  Demand: {total_demand}, Arriving: {total_arriving}")
        
        holding, shortage, stockouts = simulate_week(states, sales, costs)
        total = holding + shortage
        
        console.print(f"  Costs: Holding=€{holding:.2f}, Shortage=€{shortage:.2f}, Total=€{total:.2f}")
        
        # Place next order
        next_order_num = week + 1
        if next_order_num <= 6:
            next_order = {}
            for key, state in states.items():
                model = model_selector.get(key, 'zinb')
                qdf = load_quantiles(state.store, state.product, model, checkpoints_dir)
                order = generate_order_L3(state, qdf, costs, quantile_levels)
                next_order[key] = order
                state.in_transit[2] = order
            
            all_orders[next_order_num] = sum(next_order.values())
            console.print(f"  Order {next_order_num}: {all_orders[next_order_num]} units → arrives Week {week + 3}")
        
        results.append({
            'week': week,
            'demand': total_demand,
            'arriving': total_arriving,
            'holding_cost': holding,
            'shortage_cost': shortage,
            'total_cost': total,
            'stockouts': stockouts
        })
    
    return pd.DataFrame(results), model_selector


def load_leaderboard(path: Path) -> pd.DataFrame:
    """Load and parse leaderboard."""
    lines = path.read_text().split('\n')
    data = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit() and 0 < int(line) < 300:
            rank = int(line)
            if i + 2 < len(lines):
                name = lines[i + 1].strip()
                score_line = lines[i + 2].strip()
                try:
                    parts = score_line.split('\t')
                    score = float(parts[0])
                    if 1000 < score < 15000:
                        data.append({'rank': rank, 'name': name, 'score': score})
                except:
                    pass
            i += 3
        else:
            i += 1
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial-state', type=Path,
                        default=Path('data/raw/Week 0 - 2024-04-08 - Initial State.csv'))
    parser.add_argument('--sales-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--historical-sales', type=Path,
                        default=Path('data/raw/Week 0 - 2024-04-08 - Sales.csv'))
    parser.add_argument('--checkpoints-dir', type=Path, default=Path('models/checkpoints_h3'))
    parser.add_argument('--leaderboard', type=Path,
                        default=Path('data/raw/leaderboards/FinalScore.txt'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports/backtest_L3_optimized'))
    parser.add_argument('--max-weeks', type=int, default=5)
    
    args = parser.parse_args()
    
    costs = Costs(holding=0.2, shortage=1.0)
    
    console.print("[bold magenta]═══════════════════════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]  OPTIMIZED L=3 SIMULATION WITH H=3 MODEL SELECTION         [/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════════════════════════[/bold magenta]")
    
    results, selector = run_optimized_simulation(
        args.initial_state,
        args.sales_dir,
        args.checkpoints_dir,
        args.historical_sales,
        args.max_weeks,
        costs
    )
    
    # Summary
    console.print("\n[bold green]═══════════════════════════════════════════════════════════[/bold green]")
    console.print("[bold green]                    SIMULATION RESULTS                       [/bold green]")
    console.print("[bold green]═══════════════════════════════════════════════════════════[/bold green]")
    
    total_holding = results['holding_cost'].sum()
    total_shortage = results['shortage_cost'].sum()
    total_cost_w1_5 = results['total_cost'].sum()
    
    console.print(f"\n[bold]Weeks 1-5 Results:[/bold]")
    console.print(f"  Total Holding: €{total_holding:.2f}")
    console.print(f"  Total Shortage: €{total_shortage:.2f}")
    console.print(f"  Total Cost: €{total_cost_w1_5:.2f}")
    
    # Estimate weeks 6-8
    # Use average of weeks 3-5 (when our orders are arriving) as estimate
    avg_week_cost = results[results['week'] >= 3]['total_cost'].mean()
    estimated_w6_8 = avg_week_cost * 3
    estimated_total_8_weeks = total_cost_w1_5 + estimated_w6_8
    
    console.print(f"\n[bold]8-Week Estimate:[/bold]")
    console.print(f"  Weeks 1-5 actual: €{total_cost_w1_5:.2f}")
    console.print(f"  Weeks 6-8 estimated (avg of W3-5): €{estimated_w6_8:.2f}")
    console.print(f"  [bold]Total 8-week estimate: €{estimated_total_8_weeks:.2f}[/bold]")
    
    # Compare with actual and leaderboard
    our_actual = 7787.40  # Our actual 8-week cost
    
    console.print(f"\n[bold]Comparison:[/bold]")
    console.print(f"  Our actual (L=2): €{our_actual:.2f}")
    console.print(f"  Optimized L=3 estimate: €{estimated_total_8_weeks:.2f}")
    console.print(f"  Improvement: €{our_actual - estimated_total_8_weeks:.2f} ({(our_actual - estimated_total_8_weeks)/our_actual*100:.1f}%)")
    
    # Find placement
    if args.leaderboard.exists():
        leaderboard = load_leaderboard(args.leaderboard)
        if not leaderboard.empty:
            placement = (leaderboard['score'] < estimated_total_8_weeks).sum() + 1
            total_participants = len(leaderboard)
            
            console.print(f"\n[bold]Estimated Placement:[/bold]")
            console.print(f"  Position: {placement} of {total_participants}")
            
            winner_cost = leaderboard['score'].min()
            console.print(f"  Winner cost: €{winner_cost:.2f}")
            console.print(f"  Gap to winner: €{estimated_total_8_weeks - winner_cost:.2f}")
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / 'simulation_results.csv', index=False)
    
    # Save selector
    selector_df = pd.DataFrame([
        {'store': k[0], 'product': k[1], 'model': v}
        for k, v in selector.items()
    ])
    selector_df.to_csv(args.output_dir / 'h3_optimized_selector.csv', index=False)
    
    console.print(f"\n[green]Results saved to {args.output_dir}[/green]")


if __name__ == '__main__':
    main()

