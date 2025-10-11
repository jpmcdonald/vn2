"""Command-line interface for VN2"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from rich import print as rprint

from vn2.data import submission_index, load_initial_state, load_sales, load_master
from vn2.sim import Simulator, Costs, LeadTime
from vn2.policy import base_stock_orders
from vn2.policy.bruteforce_mc import optimize_bruteforce_mc
from vn2.submit import build_submission
from vn2.analyze import describe_sku, segment_abc, segment_xyz, print_segments
from vn2.uncertainty import (
    quantiles_to_sip_samples,
    load_quantiles,
    SLURP,
    make_sip_from_uniform_threshold,
    sip_meta_df,
)


def load_config(path: str) -> dict:
    """Load YAML configuration"""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---- Command implementations ----

def cmd_ingest(args):
    """Ingest raw data and create clean interim artifacts"""
    rprint(f"[cyan]Ingesting data from {args.raw}...[/cyan]")
    
    idx = submission_index(args.raw)
    state = load_initial_state(args.raw, idx)
    master = load_master(args.raw)
    
    # Save to interim
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    state.to_parquet(out_dir / "state.parquet")
    master.to_parquet(out_dir / "master.parquet")
    
    rprint(f"[green]✓ Ingested {len(idx)} SKUs -> {args.out}[/green]")
    rprint(f"  State shape: {state.shape}")
    rprint(f"  Master shape: {master.shape}")


def cmd_simulate(args):
    """Run simulation with base-stock policy"""
    cfg = load_config(args.config)
    
    rprint("[cyan]Running simulation...[/cyan]")
    
    idx = submission_index(cfg["paths"]["raw"])
    state = load_initial_state(cfg["paths"]["raw"], idx)
    
    # Placeholder forecasts (replace with real model)
    mu = pd.Series(1.0, index=idx)
    sigma = pd.Series(0.5, index=idx)
    
    orders = base_stock_orders(
        mu, sigma, state,
        lead_weeks=cfg["simulation"]["lead_weeks"],
        review_weeks=cfg["simulation"]["review_weeks"],
        service=cfg["policy_defaults"]["service_level"]
    )
    
    # Simulate one step
    sim = Simulator(
        Costs(**cfg["simulation"]["costs"]),
        LeadTime(cfg["simulation"]["lead_weeks"], cfg["simulation"]["review_weeks"])
    )
    
    demand = mu  # Placeholder
    next_state, costs = sim.step(state, demand, orders)
    
    rprint(f"[green]Simulation complete:[/green]")
    rprint(f"  Holding cost: ${costs['holding']:.2f}")
    rprint(f"  Shortage cost: ${costs['shortage']:.2f}")
    rprint(f"  Total cost: ${costs['total']:.2f}")
    
    if args.out:
        build_submission(idx, orders, args.out)
        rprint(f"[green]✓ Wrote submission -> {args.out}[/green]")


def cmd_optimize_mc(args):
    """Optimize using Monte Carlo over SIP samples"""
    cfg = load_config(args.config)
    
    rprint("[cyan]Running Monte Carlo optimization...[/cyan]")
    
    idx = submission_index(cfg["paths"]["raw"])
    state = load_initial_state(cfg["paths"]["raw"], idx)
    
    # Load or generate quantiles (placeholder: use uniform)
    quantiles = cfg["sip"]["quantiles"]
    horizon = cfg["sip"]["horizon_weeks"]
    
    # Placeholder: generate dummy quantiles
    rprint("[yellow]Using placeholder quantiles (replace with real forecasts)[/yellow]")
    q_tables = {}
    for t in range(1, horizon + 1):
        Q = pd.DataFrame(
            np.random.rand(len(idx), len(quantiles)) * 2,
            index=idx,
            columns=quantiles
        )
        Q = Q.apply(lambda row: sorted(row), axis=1, result_type="broadcast")
        q_tables[t] = Q
    
    # Generate SIP samples
    samples = quantiles_to_sip_samples(
        q_tables, idx, 
        n_sims=cfg["mc_opt"]["n_sims"],
        seed=cfg["mc_opt"]["seed"]
    )
    
    # Base-stock as initial guess
    mu = pd.Series(1.0, index=idx)
    sigma = pd.Series(0.5, index=idx)
    base_upto = mu * 3
    sigma_lt = sigma * np.sqrt(3)
    
    # Optimize
    orders = optimize_bruteforce_mc(
        state, samples, base_upto, sigma_lt,
        costs=Costs(**cfg["simulation"]["costs"]),
        lt=LeadTime(cfg["simulation"]["lead_weeks"], cfg["simulation"]["review_weeks"]),
        span=cfg["mc_opt"]["grid_span"],
        cap=cfg["mc_opt"]["max_grid"]
    )
    
    rprint(f"[green]✓ MC optimization complete[/green]")
    rprint(f"  Mean order: {orders.mean():.1f}")
    rprint(f"  Max order: {orders.max()}")
    
    if args.out:
        build_submission(idx, orders, args.out)
        rprint(f"[green]✓ Wrote submission -> {args.out}[/green]")


def cmd_submit(args):
    """Build submission file"""
    cfg = load_config(args.config)
    
    idx = submission_index(cfg["paths"]["raw"])
    
    # Placeholder: zeros
    orders = pd.Series(0, index=idx, dtype=int)
    
    build_submission(idx, orders, args.out)
    rprint(f"[green]✓ Wrote submission -> {args.out}[/green]")


def cmd_analyze(args):
    """Ad-hoc SKU analysis"""
    sales = load_sales(args.raw)
    describe_sku(sales, args.store, args.product)


def cmd_segment(args):
    """Segment SKUs using ABC/XYZ"""
    rprint("[cyan]Segmenting SKUs...[/cyan]")
    
    master = load_master(args.raw)
    sales = load_sales(args.raw)
    
    # Placeholder: use simple metrics
    rprint("[yellow]Using placeholder segmentation logic[/yellow]")
    
    rprint("[green]Segmentation complete (placeholder)[/green]")


def cmd_slurp_demo(args):
    """Demonstrate SIP/SLURP functionality"""
    rprint("[cyan]Running SIP/SLURP demo (from R example)...[/cyan]")
    
    # Replicate R example
    names = ["Moe", "Larry", "Curly", "Shep", "Sisko", "Adriana", "Lea", "Pat", "Perry", "Elizabeth"]
    rng = np.random.default_rng(1)
    
    closure_rate = np.clip(rng.normal(0.06, 0.015, len(names)), 0, 1)
    salary = rng.normal(60000, 15000, len(names)).round(2)
    age = np.floor(rng.normal(35, 10, len(names))).astype(int)
    
    df = pd.DataFrame({
        "names": names,
        "closure_rate": closure_rate,
        "salary": salary,
        "age": age
    })
    
    # Generate SIPs
    U = rng.random((2000, len(names)))
    sips = {}
    
    for i, nm in enumerate(names):
        sips[nm] = make_sip_from_uniform_threshold(
            U[:, i], 
            threshold=df.loc[i, "closure_rate"],
            value_if_success=50000.0
        )
    
    # Create SLURP
    meta = sip_meta_df(df, idvect=names, metanamesvect=["closure_rate", "salary", "age"])
    slurp = SLURP.from_dict(sips, meta=meta, provenance="VN2 cold-call example")
    
    # Sample with relationships preserved
    samples = slurp.sample_rows(5, seed=42)
    
    rprint("[green]✓ SLURP created with {slurp.n_scenarios} scenarios[/green]")
    rprint("\n[bold]Sample (first 5 rows):[/bold]")
    rprint(samples)
    
    if args.out:
        slurp.to_xml(args.out, csvr=4, average=True, median=True)
        rprint(f"\n[green]✓ Wrote SLURP XML -> {args.out}[/green]")


# ---- Main CLI ----

def main():
    """Main CLI entry point"""
    p = argparse.ArgumentParser(
        prog="go",
        description="VN2 Inventory Planning Challenge"
    )
    
    sp = p.add_subparsers(dest="cmd", required=True)
    
    # ingest
    g = sp.add_parser("ingest", help="Ingest raw data")
    g.add_argument("--raw", required=True, help="Raw data directory")
    g.add_argument("--out", required=True, help="Output directory")
    g.set_defaults(func=cmd_ingest)
    
    # simulate
    g = sp.add_parser("simulate", help="Run simulation with base-stock policy")
    g.add_argument("--config", default="configs/base.yaml", help="Config file")
    g.add_argument("--out", help="Output submission file")
    g.set_defaults(func=cmd_simulate)
    
    # optimize-mc
    g = sp.add_parser("optimize-mc", help="Optimize via Monte Carlo over SIP")
    g.add_argument("--config", default="configs/uncertainty.yaml", help="Config file")
    g.add_argument("--out", required=True, help="Output submission file")
    g.set_defaults(func=cmd_optimize_mc)
    
    # submit
    g = sp.add_parser("submit", help="Build submission file")
    g.add_argument("--config", default="configs/base.yaml", help="Config file")
    g.add_argument("--out", required=True, help="Output submission file")
    g.set_defaults(func=cmd_submit)
    
    # analyze
    g = sp.add_parser("analyze", help="Analyze specific SKU")
    g.add_argument("--raw", required=True, help="Raw data directory")
    g.add_argument("--store", type=int, required=True, help="Store ID")
    g.add_argument("--product", type=int, required=True, help="Product ID")
    g.set_defaults(func=cmd_analyze)
    
    # segment
    g = sp.add_parser("segment", help="Segment SKUs with ABC/XYZ")
    g.add_argument("--raw", required=True, help="Raw data directory")
    g.set_defaults(func=cmd_segment)
    
    # slurp-demo
    g = sp.add_parser("slurp-demo", help="Demonstrate SIP/SLURP")
    g.add_argument("--out", help="Output XML file")
    g.set_defaults(func=cmd_slurp_demo)
    
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

