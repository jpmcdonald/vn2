#!/usr/bin/env python3
"""
Simulate full inventory dynamics with L=3 counterfactual orders.

This script:
1. Loads initial state (Week 0)
2. Loads counterfactual orders from the L=3 backtest
3. Loads actual sales for all weeks
4. Simulates inventory week-by-week with proper L=3 lead time
5. Computes exact realized costs
6. Compares against leaderboard to determine placement

Lead Time Semantics:
- Order placed END of week t arrives START of week t+3
- Order 1 (placed end W0) → arrives start W3
- Order 2 (placed end W1) → arrives start W4
- etc.
"""

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class SKUState:
    on_hand: int
    intransit_w1: int  # Arriving start of next week
    intransit_w2: int  # Arriving in 2 weeks
    intransit_w3: int  # Arriving in 3 weeks


def load_initial_state(path: Path) -> pd.DataFrame:
    """Load Week 0 initial state."""
    df = pd.read_csv(path)
    return df


def load_sales(sales_dir: Path, week: int) -> pd.DataFrame:
    """Load actual sales for a specific week."""
    # Map week number to file and date column
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
        return None
    
    filename, date_col = week_info[week]
    path = sales_dir / filename
    if not path.exists():
        return None
    
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    
    # Extract the specific week's sales
    result = df[['Store', 'Product', date_col]].copy()
    result = result.rename(columns={date_col: 'demand'})
    result['demand'] = result['demand'].fillna(0).astype(int)
    
    return result


def load_orders(orders_dir: Path, decision_week: int) -> pd.DataFrame:
    """Load counterfactual orders for a decision week."""
    path = orders_dir / f"counterfactual_orders_week{decision_week}.csv"
    if not path.exists():
        return None
    
    df = pd.read_csv(path)
    df = df.rename(columns={'0': 'order_qty'})
    df['order_qty'] = df['order_qty'].fillna(0).astype(int)
    return df


def simulate_inventory(
    initial_state: pd.DataFrame,
    orders: Dict[int, pd.DataFrame],  # decision_week -> orders_df
    sales: Dict[int, pd.DataFrame],   # sales_week -> sales_df
    max_weeks: int = 8,
    holding_cost: float = 0.2,
    shortage_cost: float = 1.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate inventory dynamics week by week.
    
    Timeline:
    - Initial state has End Inventory + In Transit W+1 (Week 1) + In Transit W+2 (Week 2)
    - Order placed at decision_week=1 (end of W0) arrives at week 3 (W0+3)
    - Order placed at decision_week=2 (end of W1) arrives at week 4 (W1+3)
    - etc.
    
    Returns:
        (sku_results_df, weekly_summary_df)
    """
    
    # Initialize state for each SKU
    skus = initial_state[['Store', 'Product']].drop_duplicates()
    
    # Build lookup for initial inventory
    # Initial state has: End Inventory, In Transit W+1 (arrives W1), In Transit W+2 (arrives W2)
    init_lookup = {}
    for _, row in initial_state.iterrows():
        key = (int(row['Store']), int(row['Product']))
        # Get initial inventory and in-transit
        on_hand = int(row.get('End Inventory', row.get('Start Inventory', 0)))
        it1 = int(row.get('In Transit W+1', 0))  # Arrives start of W1
        it2 = int(row.get('In Transit W+2', 0))  # Arrives start of W2
        init_lookup[key] = SKUState(on_hand=on_hand, intransit_w1=it1, intransit_w2=it2, intransit_w3=0)
    
    # Build order lookup: order at decision_week arrives at week decision_week+2
    # decision_week=1 means placed end of W0, arrives start of W3 (0+3=3, but decision_week=1 so +2)
    # decision_week=2 means placed end of W1, arrives start of W4
    order_lookup = {}  # (store, product, arrival_week) -> qty
    for decision_week, orders_df in orders.items():
        arrival_week = decision_week + 2  # e.g., decision_week=1 → arrives week 3
        for _, row in orders_df.iterrows():
            key = (int(row['Store']), int(row['Product']), arrival_week)
            order_lookup[key] = int(row['order_qty'])
    
    # Build sales lookup
    sales_lookup = {}  # (store, product, week) -> demand
    for week, sales_df in sales.items():
        if sales_df is not None:
            for _, row in sales_df.iterrows():
                key = (int(row['Store']), int(row['Product']), week)
                sales_lookup[key] = int(row['demand'])
    
    # Simulate week by week
    weekly_results = []
    sku_details = []
    
    # Current state by SKU
    state = {key: SKUState(s.on_hand, s.intransit_w1, s.intransit_w2, s.intransit_w3) 
             for key, s in init_lookup.items()}
    
    for week in range(1, max_weeks + 1):
        week_holding = 0.0
        week_shortage = 0.0
        week_stockouts = 0
        
        for _, sku_row in skus.iterrows():
            store = int(sku_row['Store'])
            product = int(sku_row['Product'])
            key = (store, product)
            
            if key not in state:
                continue
            
            s = state[key]
            
            # 1. Arrivals at start of week
            arriving = s.intransit_w1
            available = s.on_hand + arriving
            
            # 2. Get demand for this week
            demand = sales_lookup.get((store, product, week), 0)
            
            # 3. Compute sales and shortages
            sold = min(available, demand)
            shortage = max(0, demand - available)
            leftover = max(0, available - demand)
            
            # 4. Compute costs
            h_cost = holding_cost * leftover
            s_cost = shortage_cost * shortage
            
            week_holding += h_cost
            week_shortage += s_cost
            if shortage > 0:
                week_stockouts += 1
            
            # 5. Record details
            sku_details.append({
                'week': week,
                'store': store,
                'product': product,
                'start_inventory': s.on_hand,
                'arriving': arriving,
                'available': available,
                'demand': demand,
                'sold': sold,
                'shortage': shortage,
                'leftover': leftover,
                'holding_cost': h_cost,
                'shortage_cost': s_cost
            })
            
            # 6. Get new order arriving at this week (from decision at week-3)
            new_arrival_next = s.intransit_w2
            new_arrival_w2 = s.intransit_w3
            new_arrival_w3 = order_lookup.get((store, product, week + 3), 0)
            
            # 7. Update state for next week
            state[key] = SKUState(
                on_hand=leftover,
                intransit_w1=new_arrival_next,
                intransit_w2=new_arrival_w2,
                intransit_w3=new_arrival_w3
            )
        
        weekly_results.append({
            'week': week,
            'holding_cost': week_holding,
            'shortage_cost': week_shortage,
            'total_cost': week_holding + week_shortage,
            'stockouts': week_stockouts
        })
        
        console.print(f"  Week {week}: Holding=€{week_holding:.2f}, Shortage=€{week_shortage:.2f}, Total=€{week_holding + week_shortage:.2f}")
    
    return pd.DataFrame(sku_details), pd.DataFrame(weekly_results)


def load_leaderboard(path: Path) -> pd.DataFrame:
    """Load and parse leaderboard data."""
    lines = path.read_text().split('\n')
    
    data = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for rank numbers (standalone line with just a number)
        if line.isdigit() and int(line) > 0 and int(line) < 300:
            rank = int(line)
            if i + 2 < len(lines):
                name = lines[i + 1].strip()
                score_line = lines[i + 2].strip()
                try:
                    # Score line format: "4677.0\t8\t" or similar
                    parts = score_line.split('\t')
                    if parts[0]:
                        score = float(parts[0])
                        # Validate score is reasonable (competition costs were 4000-10000+)
                        if 1000 < score < 15000:
                            data.append({'rank': rank, 'name': name, 'score': score})
                except:
                    pass
            i += 3
        else:
            i += 1
    
    return pd.DataFrame(data)


def find_placement(leaderboard: pd.DataFrame, cost: float) -> int:
    """Find where a given cost would rank."""
    better_count = (leaderboard['score'] < cost).sum()
    return better_count + 1


def main():
    parser = argparse.ArgumentParser(description='Simulate L=3 inventory costs')
    parser.add_argument('--initial-state', type=Path, 
                        default=Path('data/raw/Week 0 - 2024-04-08 - Initial State.csv'))
    parser.add_argument('--orders-dir', type=Path,
                        default=Path('reports/backtest_L3'))
    parser.add_argument('--sales-dir', type=Path,
                        default=Path('data/raw'))
    parser.add_argument('--leaderboard', type=Path,
                        default=Path('data/raw/leaderboards/FinalScore.txt'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('reports/backtest_L3'))
    parser.add_argument('--max-weeks', type=int, default=5,
                        help='Max weeks to simulate (we have sales through week 5)')
    
    args = parser.parse_args()
    
    console.print("[bold cyan]L=3 Counterfactual Cost Simulation[/bold cyan]\n")
    
    # Load initial state
    console.print("Loading initial state...")
    initial_state = load_initial_state(args.initial_state)
    console.print(f"  {len(initial_state)} SKUs")
    
    # Load counterfactual orders
    console.print("Loading counterfactual orders...")
    orders = {}
    for decision_week in range(1, 7):
        df = load_orders(args.orders_dir, decision_week)
        if df is not None:
            orders[decision_week] = df
            console.print(f"  Week {decision_week}: {df['order_qty'].sum()} units")
    
    # Load actual sales
    console.print("Loading actual sales...")
    sales = {}
    for week in range(1, args.max_weeks + 1):
        df = load_sales(args.sales_dir, week)
        if df is not None:
            sales[week] = df
            console.print(f"  Week {week}: {df['demand'].sum()} units demand")
    
    # Run simulation
    console.print("\n[bold]Simulating Inventory Dynamics[/bold]")
    sku_df, weekly_df = simulate_inventory(
        initial_state, orders, sales, 
        max_weeks=args.max_weeks
    )
    
    # Compute totals
    total_holding = weekly_df['holding_cost'].sum()
    total_shortage = weekly_df['shortage_cost'].sum()
    total_cost = weekly_df['total_cost'].sum()
    
    console.print(f"\n[bold green]Simulation Results (Weeks 1-{args.max_weeks})[/bold green]")
    console.print(f"  Total Holding Cost: €{total_holding:.2f}")
    console.print(f"  Total Shortage Cost: €{total_shortage:.2f}")
    console.print(f"  [bold]Total Cost: €{total_cost:.2f}[/bold]")
    
    # Load leaderboard and find placement
    if args.leaderboard.exists():
        console.print("\n[bold]Leaderboard Comparison[/bold]")
        leaderboard = load_leaderboard(args.leaderboard)
        
        # Note: The leaderboard shows cumulative costs through week 8
        # We only have sales through week 5, so we need to extrapolate
        # or compare against weekly leaderboards
        
        our_actual = leaderboard[leaderboard['name'].str.contains('Patrick McDonald', case=False)]
        if not our_actual.empty:
            actual_cost = our_actual.iloc[0]['score']
            actual_rank = our_actual.iloc[0]['rank']
            console.print(f"  Our Actual (L=2): Rank {int(actual_rank)}, €{actual_cost:.2f}")
        
        # Estimate full 8-week cost by scaling
        # Average weekly cost in simulation
        avg_weekly = total_cost / args.max_weeks
        estimated_8week = total_cost + (avg_weekly * (8 - args.max_weeks))
        
        console.print(f"  L=3 Simulated (Weeks 1-{args.max_weeks}): €{total_cost:.2f}")
        console.print(f"  L=3 Estimated (8 weeks): €{estimated_8week:.2f}")
        
        # Find estimated placement
        placement = find_placement(leaderboard, estimated_8week)
        console.print(f"  [bold]Estimated Placement: {placement} of {len(leaderboard)}[/bold]")
        
        # Compare with winner
        winner_cost = leaderboard['score'].min()
        console.print(f"\n  Winner Cost: €{winner_cost:.2f}")
        console.print(f"  Gap to Winner: €{estimated_8week - winner_cost:.2f}")
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weekly_df.to_csv(args.output_dir / 'simulated_weekly_costs.csv', index=False)
    sku_df.to_csv(args.output_dir / 'simulated_sku_details.csv', index=False)
    
    # Generate summary report
    summary_lines = [
        "# L=3 Counterfactual Simulation Results",
        "",
        "## Simulation Summary",
        "",
        f"**Weeks Simulated**: 1-{args.max_weeks}",
        "",
        "### Realized Costs",
        "",
        "| Week | Holding | Shortage | Total |",
        "|------|---------|----------|-------|",
    ]
    
    for _, row in weekly_df.iterrows():
        summary_lines.append(
            f"| {int(row['week'])} | €{row['holding_cost']:.2f} | €{row['shortage_cost']:.2f} | €{row['total_cost']:.2f} |"
        )
    
    summary_lines.extend([
        f"| **Total** | **€{total_holding:.2f}** | **€{total_shortage:.2f}** | **€{total_cost:.2f}** |",
        "",
        "### Comparison with Actual Results",
        "",
    ])
    
    if args.leaderboard.exists() and not our_actual.empty:
        summary_lines.extend([
            f"| Metric | Actual (L=2) | Counterfactual (L=3) |",
            f"|--------|--------------|---------------------|",
            f"| Cost (Weeks 1-{args.max_weeks}) | - | €{total_cost:.2f} |",
            f"| Est. 8-Week Cost | €{actual_cost:.2f} | €{estimated_8week:.2f} |",
            f"| Rank | {int(actual_rank)} | ~{placement} |",
            f"| Improvement | - | €{actual_cost - estimated_8week:.2f} ({(actual_cost - estimated_8week)/actual_cost*100:.1f}%) |",
        ])
    
    (args.output_dir / 'simulation_summary.md').write_text('\n'.join(summary_lines))
    console.print(f"\n[green]Results saved to {args.output_dir}[/green]")


if __name__ == '__main__':
    main()

