#!/usr/bin/env python3
"""
Generate next order based on state and sales data.

Usage:
    python scripts/generate_next_order.py \
        --state-file data/states/state1.csv \
        --output data/submissions/order2_jpatrickmcdonald.csv
    
    # For testing (output to _test file)
    python scripts/generate_next_order.py \
        --state-file data/states/state1.csv \
        --sales-file data/raw/Week\ 1\ -\ 2024-04-15\ -\ Sales.csv \
        --output data/submissions/order2_test.csv \
        --test
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from vn2.analyze.sequential_planner import choose_order_L2, Costs
from vn2.analyze.sip_opt import quantiles_to_pmf
from rich.console import Console
from rich.table import Table

console = Console()


def load_state_csv(state_path: Path) -> pd.DataFrame:
    """Load state CSV with proper column handling."""
    df = pd.read_csv(state_path)
    
    # Normalize column names (handle both "In Transit W+1" and "In Transit W+2" style)
    col_map = {}
    for col in df.columns:
        if col.lower() in ['store', 'product']:
            continue
        elif 'end inventory' in col.lower():
            col_map[col] = 'end_inventory'
        elif 'in transit w+1' in col.lower() or 'in transit w+2' in col.lower():
            # Keep original for reference but also map to standard names
            if 'w+1' in col.lower():
                col_map[col] = 'intransit_1'
            elif 'w+2' in col.lower():
                col_map[col] = 'intransit_2'
    
    if col_map:
        df = df.rename(columns=col_map)
    
    return df


def generate_order(
    state_path: Path,
    output_path: Path,
    selector_map_path: Path = None,
    checkpoints_dir: Path = None,
    cu: float = 1.0,
    co: float = 0.2,
    sip_grain: int = 500,
    fold_idx: int = 0,
    test_mode: bool = False
) -> None:
    """Generate next order from state data."""
    
    # Default paths
    if selector_map_path is None:
        selector_map_path = Path('models/results/selector_map_seq12_v1.parquet')
    if checkpoints_dir is None:
        checkpoints_dir = Path('models/checkpoints')
    
    if not selector_map_path.exists():
        raise FileNotFoundError(f"Selector map not found: {selector_map_path}")
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    console.print(f"[bold blue]📦 Generating Order from {state_path.name}[/bold blue]")
    console.print(f"  Output: {output_path}")
    console.print(f"  Costs: cu={cu}, co={co}")
    console.print(f"  PMF grain: {sip_grain}")
    console.print(f"  Test mode: {test_mode}")
    
    # Load state
    console.print("[cyan]Loading state data...[/cyan]")
    state = load_state_csv(state_path)
    console.print(f"  Loaded {len(state)} SKUs")
    
    # Load selector
    console.print("[cyan]Loading selector map...[/cyan]")
    selector = pd.read_parquet(selector_map_path)
    model_for_sku = {}
    for _, row in selector.iterrows():
        key = (int(row['store']), int(row['product']))
        model_for_sku[key] = row['model_name']
    console.print(f"  Loaded {len(model_for_sku)} SKU-model mappings")
    
    # Configuration
    costs = Costs(holding=co, shortage=cu)
    quantile_levels = np.array([0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99])
    
    # Generate orders
    console.print("[cyan]Generating orders...[/cyan]")
    orders = []
    expected_costs = []
    successful = 0
    
    for idx, row in state.iterrows():
        store = int(row['Store'])
        product = int(row['Product'])
        
        # Get state (with column flexibility)
        if 'end_inventory' in row.index:
            I0 = int(row['end_inventory'])
        elif 'End Inventory' in row.index:
            I0 = int(row['End Inventory'])
        else:
            # Try to find any inventory column
            inv_cols = [c for c in row.index if 'inventory' in c.lower()]
            if inv_cols:
                I0 = int(row[inv_cols[0]])
            else:
                I0 = 0
        
        # Q1: arriving next week (default to 0 if not in state)
        # NOTE: "In Transit W+1" in state files may represent past deliveries already counted in End Inventory
        # We treat it as 0 for future planning unless we have independent transit tracking
        Q1 = 0
        
        # Q2: arriving week after next (our previous order)
        # "In Transit W+2" is our Order N-1 that arrives week N+1
        if 'intransit_2' in row.index:
            Q2 = int(row['intransit_2'])
        elif 'In Transit W+2' in row.index:
            Q2 = int(row['In Transit W+2'])
        else:
            Q2 = 0
        
        # Get model
        model = model_for_sku.get((store, product), 'zinb')
        ckpt = checkpoints_dir / model / f'{store}_{product}' / f'fold_{fold_idx}.pkl'
        
        if not ckpt.exists():
            orders.append({'Store': store, 'Product': product, '0': 0})
            continue
        
        try:
            with open(ckpt, 'rb') as f:
                data = pickle.load(f)
            qdf = data.get('quantiles')
            
            if qdf is None or qdf.empty or (1 not in qdf.index) or (2 not in qdf.index):
                orders.append({'Store': store, 'Product': product, '0': 0})
                continue
            
            h1_pmf = quantiles_to_pmf(qdf.loc[1].values, quantile_levels, grain=sip_grain)
            h2_pmf = quantiles_to_pmf(qdf.loc[2].values, quantile_levels, grain=sip_grain)
            
            # Optimize order
            q_opt, exp_cost = choose_order_L2(h1_pmf, h2_pmf, I0, Q1, Q2, costs)
            
            orders.append({'Store': store, 'Product': product, '0': int(q_opt)})
            expected_costs.append(exp_cost)
            successful += 1
            
        except Exception as e:
            if test_mode:
                console.print(f"[yellow]Warning: {store}_{product} failed: {e}[/yellow]")
            orders.append({'Store': store, 'Product': product, '0': 0})
    
    # Save output
    orders_df = pd.DataFrame(orders)
    orders_df.to_csv(output_path, index=False)
    
    # Summary
    portfolio_cost = sum(expected_costs)
    total_units = orders_df['0'].sum()
    nonzero_count = (orders_df['0'] > 0).sum()
    
    console.print(f"\n[bold green]✅ Order Generated[/bold green]")
    console.print(f"  Total units: {total_units}")
    console.print(f"  SKUs with orders: {nonzero_count}")
    console.print(f"  Mean order per SKU: {orders_df['0'].mean():.2f}")
    console.print(f"  Expected portfolio cost: {portfolio_cost:.2f}")
    console.print(f"  SKUs with forecasts: {successful}")
    
    # Show non-zero orders in test mode
    if test_mode and nonzero_count > 0:
        nonzero = orders_df[orders_df['0'] > 0]
        console.print(f"\n[yellow]Sample non-zero orders (first 10):[/yellow]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Store")
        table.add_column("Product")
        table.add_column("Quantity")
        for _, r in nonzero.head(10).iterrows():
            table.add_row(str(r['Store']), str(r['Product']), str(r['0']))
        console.print(table)
    
    return orders_df, portfolio_cost


def main():
    parser = argparse.ArgumentParser(
        description="Generate next order from state data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate Order 2
  python scripts/generate_next_order.py \\
      --state-file data/states/state1.csv \\
      --output data/submissions/order2_jpatrickmcdonald.csv
  
  # Test mode (output to _test file)
  python scripts/generate_next_order.py \\
      --state-file data/states/state1.csv \\
      --output data/submissions/order2_test.csv \\
      --test
"""
    )
    
    parser.add_argument(
        '--state-file',
        type=Path,
        required=True,
        help='Path to state CSV file (with Store, Product, End Inventory, In Transit columns)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output CSV path for submission'
    )
    parser.add_argument(
        '--selector-map',
        type=Path,
        default=None,
        help='Path to selector map parquet (default: models/results/selector_map_seq12_v1.parquet)'
    )
    parser.add_argument(
        '--checkpoints-dir',
        type=Path,
        default=None,
        help='Path to checkpoints directory (default: models/checkpoints)'
    )
    parser.add_argument(
        '--cu',
        type=float,
        default=1.0,
        help='Shortage cost (default: 1.0)'
    )
    parser.add_argument(
        '--co',
        type=float,
        default=0.2,
        help='Holding cost (default: 0.2)'
    )
    parser.add_argument(
        '--sip-grain',
        type=int,
        default=500,
        help='PMF grain size (default: 500)'
    )
    parser.add_argument(
        '--fold-idx',
        type=int,
        default=0,
        help='Fold index for forecasts (default: 0)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: show more details and warnings'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.state_file.exists():
        console.print(f"[bold red]Error: State file not found: {args.state_file}[/bold red]")
        return 1
    
    # Generate order
    try:
        generate_order(
            state_path=args.state_file,
            output_path=args.output,
            selector_map_path=args.selector_map,
            checkpoints_dir=args.checkpoints_dir,
            cu=args.cu,
            co=args.co,
            sip_grain=args.sip_grain,
            fold_idx=args.fold_idx,
            test_mode=args.test
        )
        return 0
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
