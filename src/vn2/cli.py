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
from vn2.forecast.imputation_pipeline import (
    create_imputed_training_data,
    compute_imputation_summary,
    save_imputation_artifacts,
)
from vn2.uncertainty.stockout_imputation import impute_all_stockouts
from vn2.forecast.pipeline import ForecastPipeline
from vn2.forecast.models import (
    CrostonForecaster,
    SeasonalNaiveForecaster,
    ForecastConfig
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


def cmd_impute_stockouts(args):
    """Impute stockout-censored demand using profile-based SIP replacement"""
    cfg = load_config(args.config)
    
    # Determine paths
    processed_dir = args.processed if args.processed else cfg['paths']['processed']
    
    rprint("[bold blue]🔬 Imputing stockout-censored demand...[/bold blue]")
    
    # Load data
    demand_path = Path(processed_dir) / 'demand_long.parquet'
    surd_path = Path(processed_dir) / 'surd_transforms.parquet'
    
    if not demand_path.exists():
        rprint(f"[bold red]Error: {demand_path} not found. Run EDA notebook first.[/bold red]")
        return
    
    if not surd_path.exists():
        rprint(f"[bold yellow]Warning: {surd_path} not found. Using default 'log' transform.[/bold yellow]")
        surd_transforms = pd.DataFrame()
    else:
        surd_transforms = pd.read_parquet(surd_path)
        if 'Store' in surd_transforms.columns and 'Product' in surd_transforms.columns:
            surd_transforms = surd_transforms.set_index(['Store', 'Product'])
    
    df = pd.read_parquet(demand_path)
    
    # Get quantile levels from config
    q_levels = np.array(cfg.get('sip', {}).get('quantiles', 
                                                [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 
                                                 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]))
    
    rprint(f"📊 Data: {len(df):,} observations")
    rprint(f"   Stockouts: {(~df['in_stock']).sum():,} ({(~df['in_stock']).mean()*100:.1f}%)")
    rprint(f"   Quantile levels: {len(q_levels)}")
    rprint(f"   Neighbors per imputation: {args.n_neighbors}")
    
    # Create imputed training data
    rprint(f"🚀 Using {args.n_jobs if args.n_jobs > 0 else 'all'} CPU cores for parallel processing")
    df_imputed = create_imputed_training_data(
        df, surd_transforms, q_levels, 
        n_neighbors=args.n_neighbors, verbose=True,
        n_jobs=args.n_jobs
    )
    
    # Compute imputed SIPs separately for saving
    rprint("\n[cyan]Generating full SIP library for stockout weeks...[/cyan]")
    imputed_sips = impute_all_stockouts(
        df, surd_transforms, q_levels,
        n_neighbors=args.n_neighbors, verbose=False,
        n_jobs=args.n_jobs
    )
    
    # Summary
    rprint("\n[bold green]📈 Imputation Summary[/bold green]")
    summary = compute_imputation_summary(df, df_imputed)
    
    for _, row in summary.iterrows():
        metric = row['metric']
        value = row['value']
        if 'pct' in metric:
            rprint(f"   {metric}: {value:.1f}%")
        elif 'lift' in metric or 'mean' in metric or 'median' in metric:
            rprint(f"   {metric}: {value:.2f}")
        else:
            rprint(f"   {metric}: {value:,.0f}")
    
    # Save artifacts
    rprint("\n[cyan]💾 Saving artifacts...[/cyan]")
    save_imputation_artifacts(
        df_imputed, imputed_sips, summary, processed_dir
    )
    
    rprint("\n[bold green]✅ Stockout imputation complete![/bold green]")
    rprint(f"   Use [bold]{processed_dir}/demand_imputed.parquet[/bold] for model training")
    rprint(f"   Full SIPs saved to [bold]{processed_dir}/imputed_sips.parquet[/bold]")


def cmd_eval_models(args):
    """Evaluate trained models with cost-based ranking"""
    import os
    import subprocess
    
    # Set BLAS threads
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    # Compute n_jobs if not specified
    if args.n_jobs is None:
        result = subprocess.run(['sysctl', '-n', 'hw.logicalcpu'], capture_output=True, text=True)
        total_cores = int(result.stdout.strip())
        n_jobs = max(1, int(total_cores * args.cpu_fraction))
    else:
        n_jobs = args.n_jobs
    
    rprint(f"[bold blue]📊 Evaluating forecast models...[/bold blue]")
    rprint(f"   Holdout: {args.holdout} folds")
    rprint(f"   Workers: {n_jobs}")
    rprint(f"   Batch size: {args.batch_size}")
    rprint(f"   Simulations: {args.n_sims}")
    if args.use_sip_optimization:
        rprint(f"   [bold green]SIP Optimization: ENABLED[/bold green]")
        rprint(f"   SIP grain: {args.sip_grain}")
    
    # Import and run
    from vn2.analyze.model_eval import run_evaluation, aggregate_results
    
    checkpoint_dir = Path('models/checkpoints')
    # Use capped demand for SIP, original for baseline
    demand_path = Path('data/processed/demand_imputed_capped.parquet') if args.use_sip_optimization else Path('data/processed/demand_imputed.parquet')
    master_path = Path('data/processed/master.parquet')
    state_path = Path('data/interim/state.parquet')
    output_dir = Path('models/results')
    
    # Use versioned progress file if suffix specified
    suffix = args.out_suffix if hasattr(args, 'out_suffix') and args.out_suffix else ""
    progress_file = Path(f'models/results/eval_progress{suffix}.json') if suffix else Path('models/results/eval_progress.json')
    
    if args.aggregate:
        # Only aggregate
        suffix_str = f"_{suffix}" if suffix else ""
        input_path = output_dir / f"eval_folds{suffix_str}.parquet"
        if not input_path.exists():
            rprint(f"[bold red]❌ No results file found at {input_path}[/bold red]")
            return
        aggregate_results(input_path, output_dir, out_suffix=suffix)
    else:
        # Run evaluation
        costs_dict = {'holding': 0.2, 'shortage': 1.0}
        
        # Filter models if specified
        if hasattr(args, 'models') and args.models:
            # Filter checkpoint_dir to only include specified models
            rprint(f"   [bold yellow]Evaluating specific models: {', '.join(args.models)}[/bold yellow]")
        
        run_evaluation(
            checkpoint_dir=checkpoint_dir,
            demand_path=demand_path,
            master_path=master_path,
            output_dir=output_dir,
            progress_file=progress_file,
            holdout_weeks=args.holdout,
            n_jobs=n_jobs,
            batch_size=args.batch_size,
            n_sims=args.n_sims,
            resume=args.resume,
            costs_dict=costs_dict,
            lead_weeks=2,
            review_weeks=1,
            use_sip=args.use_sip_optimization,
            sip_grain=args.sip_grain,
            state_path=state_path if args.use_sip_optimization else None,
            out_suffix=suffix
        )
        
        # Auto-aggregate if completed
        suffix_str = f"_{suffix}" if suffix else ""
        final_path = output_dir / f"eval_folds{suffix_str}.parquet"
        if final_path.exists():
            aggregate_results(final_path, output_dir, out_suffix=suffix)


def cmd_forecast(args):
    """Train density forecast models with checkpoint/resume"""
    import os
    
    # Set environment variables for Metal optimization
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    cfg = load_config(args.config)
    
    rprint("[bold blue]📈 Starting forecast model training...[/bold blue]")
    
    # Load data - different sources for SLURP vs challengers
    # SLURP models: raw data with in_stock flag (no imputation, handles censoring)
    # Other models: winsorized imputed data (stable, no extreme outliers)
    demand_raw_path = Path(cfg['paths']['processed']) / 'demand_long.parquet'
    demand_winsor_path = Path(cfg['paths']['processed']) / 'demand_imputed_winsor.parquet'
    master_path = Path(cfg['paths']['processed']) / 'master.parquet'
    surd_path = Path(cfg['paths']['processed']) / 'surd_transforms.parquet'
    
    # Load both datasets
    if not demand_raw_path.exists():
        rprint(f"[bold red]Error: {demand_raw_path} not found.[/bold red]")
        return
    if not demand_winsor_path.exists():
        rprint(f"[bold red]Error: {demand_winsor_path} not found. Run winsorization script.[/bold red]")
        return
    
    df_raw = pd.read_parquet(demand_raw_path)
    df_winsor = pd.read_parquet(demand_winsor_path)
    master_df = pd.read_parquet(master_path) if master_path.exists() else None
    surd_df = pd.read_parquet(surd_path) if surd_path.exists() else None

    # Optional PID weights/k
    pid_w_path = Path('models/results/pid_feature_weights.parquet')
    pid_k_path = Path('models/results/pid_k.parquet')
    pid_w_df = pd.read_parquet(pid_w_path) if pid_w_path.exists() else None
    pid_k_df = pd.read_parquet(pid_k_path) if pid_k_path.exists() else None
    
    # Default to winsorized for non-SLURP models
    df = df_winsor
    
    rprint(f"📊 Data loaded:")
    rprint(f"   Observations: {len(df):,}")
    rprint(f"   SKUs: {len(df[['Store', 'Product']].drop_duplicates())}")
    rprint(f"   Imputed: {df['imputed'].sum():,} ({df['imputed'].mean()*100:.1f}%)")
    
    # Setup quantiles
    quantiles = np.array(cfg['quantiles'])
    forecast_config = ForecastConfig(
        quantiles=quantiles,
        horizon=cfg['horizon'],
        random_state=42
    )
    
    # Model factories
    def make_croston_classic():
        return CrostonForecaster(forecast_config, variant='classic', alpha=0.1)
    
    def make_croston_sba():
        return CrostonForecaster(forecast_config, variant='sba', alpha=0.1)
    
    def make_croston_tsb():
        return CrostonForecaster(forecast_config, variant='tsb', alpha=0.1)
    
    def make_seasonal_naive():
        return SeasonalNaiveForecaster(forecast_config, season_length=52)
    
    def make_zip():
        from vn2.forecast.models.zero_inflated import ZIPForecaster
        return ZIPForecaster(forecast_config)
    
    def make_zinb():
        from vn2.forecast.models.zero_inflated import ZINBForecaster
        return ZINBForecaster(forecast_config)
    
    def make_lightgbm_quantile():
        from vn2.forecast.models.lightgbm_quantile import LightGBMQuantileForecaster
        lgb_params = cfg['models']['lightgbm_quantile']
        return LightGBMQuantileForecaster(
            forecast_config,
            max_depth=lgb_params.get('max_depth', 6),
            num_leaves=lgb_params.get('num_leaves', 31),
            learning_rate=lgb_params.get('learning_rate', 0.05),
            n_estimators=lgb_params.get('n_estimators', 100),
            min_data_in_leaf=lgb_params.get('min_data_in_leaf', 20)
        )
    
    def make_ets():
        from vn2.forecast.models.ets import ETSForecaster
        ets_params = cfg['models']['ets']
        return ETSForecaster(
            forecast_config,
            error=ets_params.get('error', 'add'),
            trend=ets_params.get('trend'),
            seasonal=ets_params.get('seasonal'),
            seasonal_periods=ets_params.get('seasonal_periods', 52)
        )
    
    def make_slurp_bootstrap():
        from vn2.forecast.models.slurp_bootstrap import SLURPBootstrapForecaster
        params = cfg['models']['slurp_bootstrap']
        return SLURPBootstrapForecaster(
            forecast_config,
            n_neighbors=params.get('n_neighbors', 50),
            n_bootstrap=params.get('n_bootstrap', 1000),
            stockout_aware=False,
            use_pid_weights=bool(pid_w_df is not None),
            pid_weights_df=pid_w_df,
            k_from_pid=bool(pid_k_df is not None),
            pid_k_df=pid_k_df,
            synergy_interactions=True
        )
    
    def make_slurp_surd(surd_transforms_df=None):
        """SLURP with SURD transforms, NO stockout handling"""
        from vn2.forecast.models.slurp_bootstrap import SURDSLURPBootstrapForecaster
        params = cfg['models']['slurp_surd']
        return SURDSLURPBootstrapForecaster(
            forecast_config,
            n_neighbors=params.get('n_neighbors', 50),
            n_bootstrap=params.get('n_bootstrap', 1000),
            stockout_aware=False,  # KEY: No stockout handling
            use_surd=True,          # KEY: Yes SURD transforms
            surd_transforms_df=surd_transforms_df,
            use_pid_weights=bool(pid_w_df is not None),
            pid_weights_df=pid_w_df,
            k_from_pid=bool(pid_k_df is not None),
            pid_k_df=pid_k_df,
            synergy_interactions=True
        )
    
    def make_slurp_stockout_aware():
        from vn2.forecast.models.slurp_bootstrap import SLURPBootstrapForecaster
        params = cfg['models']['slurp_stockout_aware']
        return SLURPBootstrapForecaster(
            forecast_config,
            n_neighbors=params.get('n_neighbors', 50),
            n_bootstrap=params.get('n_bootstrap', 1000),
            stockout_aware=params.get('stockout_aware', True),
            use_pid_weights=bool(pid_w_df is not None),
            pid_weights_df=pid_w_df,
            k_from_pid=bool(pid_k_df is not None),
            pid_k_df=pid_k_df,
            synergy_interactions=True
        )
    
    def make_slurp_surd_stockout_aware(surd_transforms_df=None):
        from vn2.forecast.models.slurp_bootstrap import SURDSLURPBootstrapForecaster
        params = cfg['models']['slurp_surd_stockout_aware']
        return SURDSLURPBootstrapForecaster(
            forecast_config,
            n_neighbors=params.get('n_neighbors', 50),
            n_bootstrap=params.get('n_bootstrap', 1000),
            stockout_aware=params.get('stockout_aware', True),
            use_surd=params.get('use_surd', True),
            surd_transforms_df=surd_transforms_df,
            use_pid_weights=bool(pid_w_df is not None),
            pid_weights_df=pid_w_df,
            k_from_pid=bool(pid_k_df is not None),
            pid_k_df=pid_k_df,
            synergy_interactions=True
        )
    
    def make_linear_quantile():
        from vn2.forecast.models.linear_quantile import LinearQuantileForecaster
        params = cfg['models']['linear_quantile']
        return LinearQuantileForecaster(
            forecast_config,
            alpha=params.get('alpha', 1.0),
            solver=params.get('solver', 'highs')
        )
    
    def make_ngboost():
        from vn2.forecast.models.ngboost_dist import NGBoostForecaster
        params = cfg['models']['ngboost']
        return NGBoostForecaster(
            forecast_config,
            dist=params.get('dist', 'LogNormal'),
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.01)
        )
    
    def make_knn_profile():
        from vn2.forecast.models.knn_profile import KNNProfileForecaster
        params = cfg['models']['knn_profile']
        return KNNProfileForecaster(
            forecast_config,
            n_neighbors=params.get('n_neighbors', 20),
            lookback_weeks=params.get('lookback_weeks', 13)
        )
    
    def make_lightgbm_point():
        """LightGBM deterministic (MSE) for point forecast baseline"""
        from vn2.forecast.models.lightgbm_point import LightGBMPointForecaster
        params = cfg['models']['lightgbm_point']
        return LightGBMPointForecaster(
            forecast_config,
            max_depth=params.get('max_depth', 6),
            num_leaves=params.get('num_leaves', 31),
            learning_rate=params.get('learning_rate', 0.05),
            n_estimators=params.get('n_estimators', 100),
            min_data_in_leaf=params.get('min_data_in_leaf', 20)
        )
    
    def make_qrf():
        """Quantile Random Forest"""
        from vn2.forecast.models.qrf import QRFForecaster
        params = cfg['models']['qrf']
        return QRFForecaster(
            forecast_config,
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            min_samples_leaf=params.get('min_samples_leaf', 5)
        )
    
    def make_glm_poisson():
        """GLM with Poisson family"""
        from vn2.forecast.models.glm_count import GLMCountForecaster
        if 'glm_poisson' not in cfg['models']:
            cfg['models']['glm_poisson'] = {}
        return GLMCountForecaster(forecast_config, family='poisson')
    
    def make_glm_negbin():
        """GLM with Negative Binomial family"""
        from vn2.forecast.models.glm_count import GLMCountForecaster
        if 'glm_negbin' not in cfg['models']:
            cfg['models']['glm_negbin'] = {}
        return GLMCountForecaster(forecast_config, family='negbin')
    
    def make_naive4():
        """Naive 4-week rolling average with Poisson density"""
        from vn2.forecast.models.naive4 import Naive4WeekForecaster
        return Naive4WeekForecaster(
            quantiles=quantiles,
            horizon=cfg['horizon'],
            use_normal=False  # Use Poisson by default
        )
    
    # Select models
    if args.pilot:
        models = {
            'croston_classic': make_croston_classic,
            'seasonal_naive': make_seasonal_naive,
        }
        rprint("\n[yellow]🧪 PILOT MODE: Using 2 fast models[/yellow]")
    else:
        models = {}
        if cfg['models']['croston_classic']['enabled']:
            models['croston_classic'] = make_croston_classic
        if cfg['models']['croston_sba']['enabled']:
            models['croston_sba'] = make_croston_sba
        if cfg['models']['croston_tsb']['enabled']:
            models['croston_tsb'] = make_croston_tsb
        if cfg['models']['seasonal_naive']['enabled']:
            models['seasonal_naive'] = make_seasonal_naive
        if cfg['models']['zip']['enabled']:
            models['zip'] = make_zip
        if cfg['models']['zinb']['enabled']:
            models['zinb'] = make_zinb
        if cfg['models']['lightgbm_quantile']['enabled']:
            models['lightgbm_quantile'] = make_lightgbm_quantile
        if cfg['models']['ets']['enabled']:
            models['ets'] = make_ets
        if cfg['models']['slurp_bootstrap']['enabled']:
            models['slurp_bootstrap'] = make_slurp_bootstrap
        if cfg['models'].get('slurp_surd', {}).get('enabled', False):
            models['slurp_surd'] = lambda: make_slurp_surd(surd_df)
        if cfg['models'].get('slurp_stockout_aware', {}).get('enabled', False):
            models['slurp_stockout_aware'] = make_slurp_stockout_aware
        if cfg['models'].get('slurp_surd_stockout_aware', {}).get('enabled', False):
            models['slurp_surd_stockout_aware'] = lambda: make_slurp_surd_stockout_aware(surd_df)
        if cfg['models']['linear_quantile']['enabled']:
            models['linear_quantile'] = make_linear_quantile
        if cfg['models']['ngboost']['enabled']:
            models['ngboost'] = make_ngboost
        if cfg['models']['knn_profile']['enabled']:
            models['knn_profile'] = make_knn_profile
        if cfg['models'].get('lightgbm_point', {}).get('enabled', False):
            models['lightgbm_point'] = make_lightgbm_point
        if cfg['models'].get('qrf', {}).get('enabled', False):
            models['qrf'] = make_qrf
        if cfg['models'].get('glm_poisson', {}).get('enabled', False):
            models['glm_poisson'] = make_glm_poisson
        if cfg['models'].get('glm_negbin', {}).get('enabled', False):
            models['glm_negbin'] = make_glm_negbin
        if cfg['models'].get('naive4', {}).get('enabled', False):
            models['naive4'] = make_naive4
    
    rprint(f"\n🤖 Models to train: {list(models.keys())}")
    
    # Pilot or Test SKUs
    pilot_skus = None
    if args.test:
        # Test mode: just one SKU
        all_skus = df[['Store', 'Product']].drop_duplicates().values.tolist()
        pilot_skus = [tuple(all_skus[0])]
        rprint(f"🧪 TEST MODE: Training on 1 SKU: {pilot_skus[0]}")
    elif args.pilot:
        all_skus = df[['Store', 'Product']].drop_duplicates().values.tolist()
        pilot_skus = [tuple(sku) for sku in all_skus[:cfg['pilot']['n_skus']]]
        rprint(f"🧪 Pilot: Training on {len(pilot_skus)} SKUs")
    
    # Initialize pipeline
    pipeline = ForecastPipeline(cfg)
    
    # Train
    n_jobs = args.n_jobs if hasattr(args, 'n_jobs') else cfg['compute']['n_jobs']
    rprint(f"\n⚙️  Parallel workers: {n_jobs}")
    rprint(f"⏱️  Timeout per fit: {cfg['compute']['timeout_per_fit']}s")
    rprint("\n[cyan]Starting training...[/cyan]\n")
    
    # Train SLURP models on raw data, others on winsorized
    slurp_models = {k: v for k, v in models.items() if 'slurp' in k.lower()}
    other_models = {k: v for k, v in models.items() if 'slurp' not in k.lower()}
    
    results_dfs = []
    
    if slurp_models:
        rprint(f"[bold green]Training SLURP models on raw data (censoring-aware):[/bold green] {list(slurp_models.keys())}")
        results_slurp = pipeline.train_all(
            df_raw,
            slurp_models,
            master_df=master_df,
            n_jobs=n_jobs,
            pilot_skus=pilot_skus
        )
        results_dfs.append(results_slurp)
    
    if other_models:
        rprint(f"\n[bold blue]Training challenger models on winsorized data:[/bold blue] {list(other_models.keys())}")
        results_other = pipeline.train_all(
            df_winsor,
            other_models,
            master_df=master_df,
            n_jobs=n_jobs,
            pilot_skus=pilot_skus
        )
        results_dfs.append(results_other)
    
    # Combine results
    if len(results_dfs) > 1:
        results_df = pd.concat(results_dfs, ignore_index=True)
    elif len(results_dfs) == 1:
        results_df = results_dfs[0]
    else:
        results_df = pd.DataFrame()
    
    rprint(f"\n[bold green]✅ Training complete![/bold green]")
    rprint(f"   Results saved to: {cfg['paths']['results']}/training_results.parquet")
# ---- PID EDA ----
def cmd_eda_info(args):
    """Run Imin PID on a sample of SKUs and save results."""
    cfg = load_config(args.config)
    from vn2.analyze.info_decomp import sample_pid
    demand_path = Path(cfg['paths']['processed']) / 'demand_long.parquet'
    if not demand_path.exists():
        rprint(f"[bold red]Error: {demand_path} not found.[/bold red]")
        return
    df = pd.read_parquet(demand_path)
    rprint(f"📊 Running PID (Imin) for {args.n_skus} SKUs...")
    out = sample_pid(df, n=args.n_skus, seed=args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    rprint(f"✅ Saved PID sample to: {out_path}")
    return out


def cmd_w8_eval(args):
    """Compute exact 8-fold per-SKU costs with Q=0 fallback"""
    from vn2.analyze.w8_eval import W8Config, run_w8_eval
    
    config = W8Config(
        folds_path=Path(args.folds_path),
        demand_path=Path(args.demand_path),
        state_path=Path(args.state_path),
        output_dir=Path(args.out_dir),
        run_tag=args.run_tag,
        n_jobs=args.n_jobs,
        holding_cost=args.holding_cost,
        shortage_cost=args.shortage_cost,
        critical_fractile=args.shortage_cost / (args.holding_cost + args.shortage_cost)
    )
    
    run_w8_eval(config)


def cmd_sequential_eval(args):
    """Run sequential L=2 evaluation over H=12 epochs"""
    from vn2.analyze.sequential_eval import SequentialConfig, run_full_sequential_eval
    
    config = SequentialConfig(
        checkpoints_dir=Path(args.checkpoints),
        demand_path=Path(args.demand),
        state_path=Path(args.state),
        output_dir=Path(args.out_dir),
        run_tag=args.run_tag,
        n_jobs=args.n_jobs,
        holding_cost=args.co,
        shortage_cost=args.cu,
        sip_grain=args.sip_grain,
        holdout_weeks=args.holdout
    )
    
    run_full_sequential_eval(config)


def cmd_today_order(args):
    """Generate today's order using latest forecasts and current state"""
    from vn2.analyze.sequential_planner import choose_order_L2, Costs
    from vn2.analyze.sip_opt import quantiles_to_pmf
    import pickle
    
    rprint("[bold blue]📦 Generating Today's Orders[/bold blue]")
    
    # Load state
    state_df = pd.read_parquet(args.state)
    if not isinstance(state_df.index, pd.MultiIndex):
        if 'Store' in state_df.columns and 'Product' in state_df.columns:
            state_df = state_df.set_index(['Store', 'Product'])
    
    rprint(f"  State: {len(state_df)} SKUs")
    
    # Load latest forecasts (fold_idx=0, most recent)
    checkpoints_dir = Path(args.checkpoints)
    model_name = args.model
    fold_idx = 0  # Most recent fold
    
    rprint(f"  Using model: {model_name}")
    rprint(f"  Checkpoints: {checkpoints_dir}")
    
    # Quantile levels (from config or default)
    quantile_levels = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    
    costs = Costs(holding=args.co, shortage=args.cu)
    
    orders = []
    
    for idx, (store, product) in enumerate(state_df.index):
        if idx % 100 == 0:
            rprint(f"  Processing SKU {idx+1}/{len(state_df)}...")
        
        # Load checkpoint
        checkpoint_path = checkpoints_dir / model_name / f"{store}_{product}" / f"fold_{fold_idx}.pkl"
        
        if not checkpoint_path.exists():
            # No forecast: order 0
            orders.append({'Store': store, 'Product': product, 'q_now': 0})
            continue
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
        except Exception:
            orders.append({'Store': store, 'Product': product, 'q_now': 0})
            continue
        
        quantiles_df = checkpoint.get('quantiles')
        if quantiles_df is None or quantiles_df.empty:
            orders.append({'Store': store, 'Product': product, 'q_now': 0})
            continue
        
        # Extract h1 and h2
        if 1 not in quantiles_df.index or 2 not in quantiles_df.index:
            orders.append({'Store': store, 'Product': product, 'q_now': 0})
            continue
        
        h1_quantiles = quantiles_df.loc[1].values
        h2_quantiles = quantiles_df.loc[2].values
        
        # Convert to PMF
        try:
            h1_pmf = quantiles_to_pmf(h1_quantiles, quantile_levels, grain=args.sip_grain)
            h2_pmf = quantiles_to_pmf(h2_quantiles, quantile_levels, grain=args.sip_grain)
        except Exception:
            orders.append({'Store': store, 'Product': product, 'q_now': 0})
            continue
        
        # Get state
        try:
            I0 = int(state_df.loc[(store, product), 'on_hand'])
            Q1 = int(state_df.loc[(store, product), 'intransit_1'])
            Q2 = int(state_df.loc[(store, product), 'intransit_2'])
        except Exception:
            I0, Q1, Q2 = 0, 0, 0
        
        # Choose order
        try:
            q_now, _ = choose_order_L2(h1_pmf, h2_pmf, I0, Q1, Q2, costs)
        except Exception:
            q_now = 0
        
        orders.append({'Store': store, 'Product': product, 'q_now': q_now})
    
    # Save to CSV
    orders_df = pd.DataFrame(orders)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    orders_df.to_csv(output_path, index=False)
    
    rprint(f"[green]✅ Saved {len(orders_df)} orders to: {output_path}[/green]")
    rprint(f"  Total units ordered: {orders_df['q_now'].sum()}")
    rprint(f"  Mean order: {orders_df['q_now'].mean():.2f}")
    rprint(f"  Max order: {orders_df['q_now'].max()}")


def cmd_ensemble_eval(args):
    """Build and evaluate ensemble models (post-hoc from existing folds)"""
    from vn2.analyze.model_eval import build_ensemble_from_folds, aggregate_results
    from vn2.analyze.ensemble import cohort_selector_rules
    
    base = Path('models/results')
    eval_folds_path = Path(args.eval_folds)
    out_suffix = args.out_suffix
    
    if args.aggregate:
        # Aggregate existing ensemble fold parts
        rprint(f"📊 Aggregating ensemble results (suffix={out_suffix})...")
        folds_path = base / f'eval_folds{out_suffix}.parquet'
        parts = sorted(base.glob(f'eval_folds_{out_suffix}_part-*.parquet'))
        
        if not folds_path.exists() and parts:
            rprint(f"Merging {len(parts)} part files...")
            df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
            df.to_parquet(folds_path)
            rprint(f"✅ Wrote {folds_path}")
        
        if folds_path.exists():
            aggregate_results(folds_path, base, out_suffix=out_suffix)
            rprint(f"✅ Aggregated {out_suffix}")
        
        if args.append_to_combined:
            # Append to v4_all leaderboard
            combined_path = base / 'leaderboards__v4_all.parquet'
            new_lb_path = base / f'leaderboards_{out_suffix}.parquet'
            
            if new_lb_path.exists():
                new_lb = pd.read_parquet(new_lb_path)
                new_lb['run'] = out_suffix.replace('_v4_ens_', 'ens_')
                
                if combined_path.exists():
                    combined = pd.read_parquet(combined_path)
                    # Remove old ensemble rows if present
                    combined = combined[~combined['run'].str.startswith('ens_')]
                    combined = pd.concat([combined, new_lb], ignore_index=True)
                else:
                    combined = new_lb
                
                combined.to_parquet(combined_path)
                rprint(f"✅ Appended to {combined_path}")
        
        return
    
    # Build ensemble from folds
    stage = args.stage
    
    if stage == 'selector':
        selector_map_path = Path(args.selector_map)
        ensemble_folds = build_ensemble_from_folds(
            stage='selector',
            eval_folds_path=eval_folds_path,
            selector_map_path=selector_map_path,
            output_path=base / f'eval_folds{out_suffix}.parquet'
        )
    
    elif stage == 'cohort':
        # Load cohort features and fit rules
        if args.cohort_features is None:
            # Build from demand_long
            rprint("Building cohort features from demand_long.parquet...")
            demand_path = Path('data/processed/demand_long.parquet')
            df = pd.read_parquet(demand_path)
            
            # Compute cohort features per SKU (simplified)
            # Normalize column names
            df = df.rename(columns={'Store': 'store', 'Product': 'product'})
            grp = df.groupby(['store', 'product'])
            feat = grp['sales'].agg(['mean', 'std']).rename(columns={'mean': 'rate', 'std': 'std'})
            feat['zero_ratio'] = grp['sales'].apply(lambda s: (s <= 0).mean()).values
            feat['cv'] = feat['std'] / feat['rate'].replace({0: np.nan})
            
            # Simple binning
            feat['rate_bin'] = pd.qcut(feat['rate'], 3, labels=['low', 'mid', 'high'], duplicates='drop')
            feat['zero_bin'] = pd.qcut(feat['zero_ratio'], 3, labels=['low', 'mid', 'high'], duplicates='drop')
            feat['cv_bin'] = pd.qcut(feat['cv'].fillna(feat['cv'].median()), 3, labels=['low', 'mid', 'high'], duplicates='drop')
            feat['stockout_bin'] = 'unknown'  # Placeholder
            
            feat = feat.reset_index()
            cohort_features_path = base / 'cohort_features_temp.parquet'
            feat.to_parquet(cohort_features_path)
        else:
            cohort_features_path = Path(args.cohort_features)
        
        # Load selector map and fit rules
        selector_map_path = Path(args.selector_map)
        selector = pd.read_parquet(selector_map_path)
        cohort_feat = pd.read_parquet(cohort_features_path)
        
        rules = cohort_selector_rules(
            cohort_feat,
            selector,
            features=['rate_bin', 'zero_bin', 'cv_bin', 'stockout_bin']
        )
        
        rprint(f"Learned {len(rules)} cohort rules")
        
        ensemble_folds = build_ensemble_from_folds(
            stage='cohort',
            eval_folds_path=eval_folds_path,
            cohort_features_path=cohort_features_path,
            cohort_rules=rules,
            output_path=base / f'eval_folds{out_suffix}.parquet'
        )
    
    else:
        raise NotImplementedError(f"Stage {stage} not yet implemented")
    
    rprint(f"✅ Ensemble build complete. Run with --aggregate to produce leaderboard.")


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
    
    # impute
    g = sp.add_parser("impute", help="Impute stockout-censored demand")
    g.add_argument("--config", default="configs/uncertainty.yaml", help="Config file")
    g.add_argument("--processed", help="Processed data directory (overrides config)")
    g.add_argument("--n-neighbors", type=int, default=20, help="Number of neighbor profiles")
    g.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs (-1 = all cores)")
    g.set_defaults(func=cmd_impute_stockouts)
    
    # forecast (NEW)
    g = sp.add_parser("forecast", help="Train density forecast models")
    g.add_argument("--config", default="configs/forecast.yaml", help="Config file")
    g.add_argument("--pilot", action="store_true", help="Run pilot test on subset of SKUs")
    g.add_argument("--test", action="store_true", help="Test mode: train on 1 SKU only")
    g.add_argument("--n-jobs", type=int, default=1, help="Number of parallel workers")
    g.set_defaults(func=cmd_forecast)
    
    # eval-models (NEW)
    g = sp.add_parser("eval-models", help="Evaluate trained models with cost-based ranking")
    g.add_argument("--holdout", type=int, default=8, help="Number of rolling-origin folds")
    g.add_argument("--n-sims", type=int, default=500, help="Monte Carlo samples")
    g.add_argument("--cpu-fraction", type=float, default=0.5, help="Fraction of CPU cores to use")
    g.add_argument("--n-jobs", type=int, help="Number of parallel workers (overrides cpu-fraction)")
    g.add_argument("--batch-size", type=int, default=2000, help="Tasks per batch")
    g.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    g.add_argument("--aggregate", action="store_true", help="Only run aggregation")
    g.add_argument("--use-sip-optimization", action="store_true", help="Use SIP-based optimization")
    g.add_argument("--sip-grain", type=int, default=1000, help="PMF grain for SIP (max support)")
    g.add_argument("--exclude-week1-cost", action="store_true", help="Exclude week 1 costs from ranking")
    g.add_argument("--out-suffix", type=str, default="", help="Output file suffix (e.g., 'v4')")
    g.add_argument("--models", type=str, nargs='+', help="Specific models to evaluate (default: all)")
    g.set_defaults(func=cmd_eval_models)
    
    # eda-info (NEW)
    g = sp.add_parser("eda-info", help="Run information decomposition EDA (Imin PID)")
    g.add_argument("--config", default="configs/forecast.yaml", help="Config file")
    g.add_argument("--n-skus", type=int, default=10, help="Number of SKUs to sample")
    g.add_argument("--seed", type=int, default=0, help="Random seed")
    g.add_argument("--output", type=str, default="models/results/pid_imin_sample.parquet", help="Output path")
    g.set_defaults(func=cmd_eda_info)
    
    # ensemble-eval
    g = sp.add_parser("ensemble-eval", help="Build and evaluate ensemble models")
    g.add_argument("--stage", type=str, choices=['selector', 'cohort', 'decision'], required=True,
                   help="Ensemble stage: selector, cohort, or decision")
    g.add_argument("--eval-folds", type=str, default="models/results/eval_folds_v4_sip.parquet",
                   help="Path to per-model fold results")
    g.add_argument("--selector-map", type=str, default="models/results/per_sku_selector_map.parquet",
                   help="Path to per-SKU selector map")
    g.add_argument("--cohort-features", type=str, default=None,
                   help="Path to cohort features parquet (for cohort stage)")
    g.add_argument("--out-suffix", type=str, default="_v4_ens_selector",
                   help="Output suffix for ensemble results")
    g.add_argument("--aggregate", action="store_true",
                   help="Aggregate existing fold parts and produce leaderboard")
    g.add_argument("--append-to-combined", action="store_true",
                   help="Append ensemble leaderboard to v4_all combined leaderboard")
    g.set_defaults(func=cmd_ensemble_eval)
    
    # w8-eval
    g = sp.add_parser("w8-eval", help="Compute exact 8-fold per-SKU costs with Q=0 fallback")
    g.add_argument("--run-tag", type=str, default="v4full",
                   help="Run tag for output files (default: v4full)")
    g.add_argument("--folds-path", type=str, default="models/results/eval_folds__v4_sip_full.parquet",
                   help="Path to folds parquet")
    g.add_argument("--demand-path", type=str, default="data/processed/demand_long.parquet",
                   help="Path to demand data")
    g.add_argument("--state-path", type=str, default="data/interim/state.parquet",
                   help="Path to state data")
    g.add_argument("--out-dir", type=str, default="models/results",
                   help="Output directory")
    g.add_argument("--n-jobs", type=int, default=12,
                   help="Number of parallel jobs")
    g.add_argument("--holding-cost", type=float, default=0.2,
                   help="Holding cost per unit")
    g.add_argument("--shortage-cost", type=float, default=1.0,
                   help="Shortage cost per unit")
    g.set_defaults(func=cmd_w8_eval)
    
    # sequential-eval (NEW)
    g = sp.add_parser("sequential-eval", help="Run sequential L=2 evaluation over H=12 epochs")
    g.add_argument("--checkpoints", type=str, default="models/checkpoints",
                   help="Path to model checkpoints directory")
    g.add_argument("--demand", type=str, default="data/processed/demand_long.parquet",
                   help="Path to demand data")
    g.add_argument("--state", type=str, default="data/interim/state.parquet",
                   help="Path to state data")
    g.add_argument("--out-dir", type=str, default="models/results",
                   help="Output directory")
    g.add_argument("--run-tag", type=str, default="seq12",
                   help="Run tag for output files")
    g.add_argument("--n-jobs", type=int, default=12,
                   help="Number of parallel jobs")
    g.add_argument("--cu", type=float, default=1.0,
                   help="Shortage cost per unit (cu)")
    g.add_argument("--co", type=float, default=0.2,
                   help="Holding cost per unit (co)")
    g.add_argument("--sip-grain", type=int, default=500,
                   help="PMF grain (max support)")
    g.add_argument("--holdout", type=int, default=12,
                   help="Number of holdout weeks (epochs)")
    g.set_defaults(func=cmd_sequential_eval)
    
    # today-order (NEW)
    g = sp.add_parser("today-order", help="Generate today's order using latest forecasts")
    g.add_argument("--checkpoints", type=str, default="models/checkpoints",
                   help="Path to model checkpoints directory")
    g.add_argument("--state", type=str, default="data/interim/state.parquet",
                   help="Path to state data")
    g.add_argument("--model", type=str, required=True,
                   help="Model name to use for forecasts")
    g.add_argument("--out", type=str, required=True,
                   help="Output CSV file for orders")
    g.add_argument("--cu", type=float, default=1.0,
                   help="Shortage cost per unit")
    g.add_argument("--co", type=float, default=0.2,
                   help="Holding cost per unit")
    g.add_argument("--sip-grain", type=int, default=500,
                   help="PMF grain (max support)")
    g.set_defaults(func=cmd_today_order)
    
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

