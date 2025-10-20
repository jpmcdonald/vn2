"""
Model evaluation with cost-based ranking over rolling-origin holdout.

Evaluates trained forecast models using:
- Standard metrics: MAE, MAPE, MASE, Bias, pinball, CRPS, coverage/width
- Density shape: PI width, curvature, moments (mean, std, skew, kurtosis)
- Cost-based value: expected cost under base-stock or MC optimization policy

Supports:
- Parallel execution with configurable resources
- Checkpointing and resume
- Background execution
- Atomic batch writes
"""

import argparse
import json
import pickle
import time
import signal
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from vn2.forecast.features import prepare_train_test_split
from vn2.forecast.evaluation import (
    pinball_loss, coverage_metrics, crps_score,
    average_pinball_loss, average_crps
)
from vn2.policy.base_stock import base_stock_orders
from vn2.sim import Simulator, Costs, LeadTime
from vn2.analyze.sip_opt import (
    quantiles_to_pmf, convolve_inventory, optimize_order,
    compute_realized_metrics, Costs as SIPCosts
)


def build_ensemble_from_folds(
    stage: str,
    eval_folds_path: Path,
    selector_map_path: Optional[Path] = None,
    cohort_features_path: Optional[Path] = None,
    cohort_rules: Optional[Dict] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Build ensemble fold-level results by selecting per-SKU models (post-hoc).
    
    Args:
        stage: 'selector' or 'cohort'
        eval_folds_path: Path to per-model fold results (e.g., eval_folds_v4_sip.parquet)
        selector_map_path: Path to per_sku_selector_map.parquet (for 'selector' stage)
        cohort_features_path: Path to cohort features (for 'cohort' stage)
        cohort_rules: Dict mapping cohort tuples to model names (for 'cohort' stage)
        output_path: Optional path to save ensemble folds
    
    Returns:
        DataFrame with ensemble fold-level results
    """
    print(f"🔧 Building ensemble (stage={stage}) from folds...")
    
    # Load per-model folds
    df = pd.read_parquet(eval_folds_path)
    
    if stage == 'selector':
        if selector_map_path is None:
            raise ValueError("selector_map_path required for stage='selector'")
        
        # Load selector map
        selector = pd.read_parquet(selector_map_path)
        
        # Check column names (v1 has 'model_name', v2 has 'selected_model')
        if 'selected_model' in selector.columns:
            selector_single = selector[['store', 'product', 'selected_model']].copy()
        elif 'model_name' in selector.columns:
            # For ties, pick first (or could sample/blend)
            selector_single = selector.groupby(['store', 'product']).first().reset_index()
            selector_single = selector_single[['store', 'product', 'model_name']].rename(
                columns={'model_name': 'selected_model'}
            )
        else:
            raise ValueError(f"Selector map must have 'model_name' or 'selected_model' column")
        
        # Join and filter
        merged = df.merge(selector_single, on=['store', 'product'], how='inner')
        ensemble_folds = merged[merged['model_name'] == merged['selected_model']].copy()
        ensemble_folds['model_name'] = 'ensemble_selector'
        ensemble_folds = ensemble_folds.drop(columns=['selected_model'])
        
        # Coverage fallback: fill missing (store, product, fold_idx) with next-best models
        keys = ['store', 'product', 'fold_idx']
        have = set(map(tuple, ensemble_folds[keys].values))
        all_keys = set(map(tuple, df[keys].drop_duplicates().values))
        
        if len(have) < len(all_keys):
            print(f"⚠️  Coverage gap: {len(all_keys) - len(have)} missing rows; applying fallback...")
            # Load per-SKU ranking (exported by selector function)
            rank_path = selector_map_path.with_name(selector_map_path.stem + '_ranking.parquet')
            if rank_path.exists():
                rk = pd.read_parquet(rank_path)
                missing = pd.DataFrame(list(all_keys - have), columns=keys)
                
                filled_rows = []
                for _, row in missing.iterrows():
                    sp_cand = rk[(rk['store'] == row['store']) & (rk['product'] == row['product'])].sort_values('model_rank')
                    for _, r in sp_cand.iterrows():
                        m = r['model_name']
                        cand_row = df[(df['store'] == row['store']) & (df['product'] == row['product']) &
                                      (df['fold_idx'] == row['fold_idx']) & (df['model_name'] == m)]
                        if len(cand_row) == 1:
                            x = cand_row.iloc[0].to_dict()
                            x['model_name'] = 'ensemble_selector'  # keep ensemble name
                            filled_rows.append(x)
                            break
                
                if filled_rows:
                    ensemble_folds = pd.concat([ensemble_folds, pd.DataFrame(filled_rows)], ignore_index=True)
                    print(f"✅ Filled {len(filled_rows)} missing rows via fallback")
            else:
                print(f"⚠️  Ranking file not found: {rank_path}; skipping fallback")
        
    elif stage == 'cohort':
        if cohort_features_path is None or cohort_rules is None:
            raise ValueError("cohort_features_path and cohort_rules required for stage='cohort'")
        
        # Load cohort features
        cohort_df = pd.read_parquet(cohort_features_path)
        
        # Apply rules to get selected model per SKU
        def lookup(row):
            key = tuple(row[f] for f in ['rate_bin', 'zero_bin', 'cv_bin', 'stockout_bin'])
            return cohort_rules.get(key, 'qrf')  # fallback to qrf
        
        cohort_df['selected_model'] = cohort_df.apply(lookup, axis=1)
        cohort_single = cohort_df[['store', 'product', 'selected_model']]
        
        # Join and filter
        merged = df.merge(cohort_single, on=['store', 'product'], how='inner')
        ensemble_folds = merged[merged['model_name'] == merged['selected_model']].copy()
        ensemble_folds['model_name'] = 'ensemble_cohort'
        ensemble_folds = ensemble_folds.drop(columns=['selected_model'])
        
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    
    print(f"✅ Built {len(ensemble_folds)} ensemble fold rows")
    
    if output_path:
        ensemble_folds.to_parquet(output_path, index=False)
        print(f"✅ Saved to {output_path}")
    
    return ensemble_folds


# Global flag for graceful shutdown
INTERRUPTED = False


def signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown"""
    global INTERRUPTED
    print(f"\n⚠️  Received signal {signum}, finishing current batch...")
    INTERRUPTED = True


@dataclass
class EvalTask:
    """Single evaluation task"""
    model_name: str
    store: int
    product: int
    fold_idx: int
    
    def key(self) -> str:
        return f"{self.model_name}_{self.store}_{self.product}_{self.fold_idx}"


class ProgressTracker:
    """Track and persist evaluation progress"""
    
    def __init__(self, progress_file: Path):
        self.progress_file = progress_file
        self.completed = set()
        self._load()
    
    def _load(self):
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                self.completed = set(data.get('completed', []))
    
    def save(self):
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump({'completed': list(self.completed)}, f, indent=2)
    
    def is_complete(self, task: EvalTask) -> bool:
        return task.key() in self.completed
    
    def mark_complete(self, task: EvalTask):
        self.completed.add(task.key())


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict]:
    """Load a model checkpoint"""
    if not checkpoint_path.exists():
        return None
    try:
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {checkpoint_path}: {e}")
        return None


def compute_point_metrics(y_true: np.ndarray, quantiles_df: pd.DataFrame) -> Dict[str, float]:
    """Compute point forecast metrics from median"""
    metrics = {}
    
    if 0.5 not in quantiles_df.columns:
        return metrics
    
    median_preds = quantiles_df[0.5].values[:len(y_true)]
    
    # MAE
    metrics['mae'] = float(np.mean(np.abs(y_true - median_preds)))
    
    # MAPE with epsilon to avoid division by zero
    epsilon = 1.0
    metrics['mape'] = float(np.mean(np.abs((y_true - median_preds) / (y_true + epsilon))) * 100)
    
    # Bias
    metrics['bias'] = float(np.mean(median_preds - y_true))
    
    # RMSE
    metrics['rmse'] = float(np.sqrt(np.mean((y_true - median_preds) ** 2)))
    
    return metrics


def compute_mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, season_length: int = 52) -> float:
    """Compute Mean Absolute Scaled Error"""
    # MAE of forecast
    mae_forecast = np.mean(np.abs(y_true - y_pred))
    
    # MAE of seasonal naive on training set
    if len(y_train) <= season_length:
        return np.nan
    
    naive_errors = []
    for i in range(season_length, len(y_train)):
        naive_errors.append(abs(y_train[i] - y_train[i - season_length]))
    
    mae_naive = np.mean(naive_errors) if naive_errors else 1.0
    
    if mae_naive == 0:
        return np.nan
    
    return mae_forecast / mae_naive


def compute_shape_metrics(quantiles_df: pd.DataFrame, n_sims: int = 500, seed: int = 42) -> Dict[str, float]:
    """Compute density shape diagnostics"""
    metrics = {}
    rng = np.random.default_rng(seed)
    quantile_levels = quantiles_df.columns.values
    
    # Width at different coverage levels
    for level in [0.8, 0.9, 0.95]:
        lower_q = (1 - level) / 2
        upper_q = 1 - lower_q
        
        lower_idx = np.argmin(np.abs(quantile_levels - lower_q))
        upper_idx = np.argmin(np.abs(quantile_levels - upper_q))
        
        for step in [1, 2]:
            if step in quantiles_df.index:
                lower_val = quantiles_df.iloc[step-1, lower_idx]
                upper_val = quantiles_df.iloc[step-1, upper_idx]
                metrics[f'width_{int(level*100)}_h{step}'] = float(upper_val - lower_val)
    
    # Curvature: ratio of h=2 to h=1 width
    if 1 in quantiles_df.index and 2 in quantiles_df.index:
        for level in [0.8, 0.9]:
            w1_key = f'width_{int(level*100)}_h1'
            w2_key = f'width_{int(level*100)}_h2'
            if w1_key in metrics and w2_key in metrics and metrics[w1_key] > 0:
                metrics[f'curvature_{int(level*100)}'] = metrics[w2_key] / metrics[w1_key]
    
    # Sample from distribution to estimate moments
    for step in [1, 2]:
        if step in quantiles_df.index:
            q_vals = quantiles_df.iloc[step-1].values
            # Inverse transform sampling
            u = rng.uniform(0, 1, n_sims)
            samples = np.interp(u, quantile_levels, q_vals)
            
            metrics[f'mean_h{step}'] = float(np.mean(samples))
            metrics[f'std_h{step}'] = float(np.std(samples))
            metrics[f'skew_h{step}'] = float(stats.skew(samples))
            metrics[f'kurtosis_h{step}'] = float(stats.kurtosis(samples))
    
    return metrics


def compute_newsvendor_metrics(
    y_true: np.ndarray,
    quantiles_df: pd.DataFrame,
    costs: Costs
) -> Dict[str, float]:
    """
    Compute newsvendor-specific metrics focused on critical fractile performance.
    
    Args:
        y_true: Actual realized demand
        quantiles_df: Forecast quantiles (rows=steps, columns=quantile levels)
        costs: Cost parameters (holding, shortage)
    
    Returns:
        Dictionary with newsvendor metrics
    """
    metrics = {}
    quantile_levels = quantiles_df.columns.values
    
    # Critical fractile for newsvendor
    critical_fractile = costs.shortage / (costs.holding + costs.shortage)
    metrics['critical_fractile'] = float(critical_fractile)
    
    # Find closest quantile to critical fractile
    cf_idx = np.argmin(np.abs(quantile_levels - critical_fractile))
    cf_actual = quantile_levels[cf_idx]
    
    # Pinball loss at critical fractile
    if 1 in quantiles_df.index and len(y_true) >= 1:
        q_cf_h1 = quantiles_df.iloc[0, cf_idx]
        error = y_true[0] - q_cf_h1
        pinball_cf = error * critical_fractile if error > 0 else -error * (1 - critical_fractile)
        metrics['pinball_cf_h1'] = float(pinball_cf)
        
        # Calibration: did actual exceed forecast at critical fractile?
        metrics['hit_cf_h1'] = float(y_true[0] >= q_cf_h1)
    
    if 2 in quantiles_df.index and len(y_true) >= 2:
        q_cf_h2 = quantiles_df.iloc[1, cf_idx]
        error = y_true[1] - q_cf_h2
        pinball_cf = error * critical_fractile if error > 0 else -error * (1 - critical_fractile)
        metrics['pinball_cf_h2'] = float(pinball_cf)
        metrics['hit_cf_h2'] = float(y_true[1] >= q_cf_h2)
    
    # Local sharpness around critical fractile (80th to 90th percentiles)
    idx_80 = np.argmin(np.abs(quantile_levels - 0.80))
    idx_90 = np.argmin(np.abs(quantile_levels - 0.90))
    
    for step in [1, 2]:
        if step in quantiles_df.index:
            q_80 = quantiles_df.iloc[step-1, idx_80]
            q_90 = quantiles_df.iloc[step-1, idx_90]
            metrics[f'local_width_h{step}'] = float(q_90 - q_80)
            
            # Quantile gradient (steepness)
            if q_90 - q_80 > 0:
                metrics[f'quantile_gradient_h{step}'] = float((q_90 - q_80) / 0.10)
    
    # Curvature around critical fractile
    idx_75 = np.argmin(np.abs(quantile_levels - 0.75))
    idx_87 = np.argmin(np.abs(quantile_levels - 0.87))
    
    for step in [1, 2]:
        if step in quantiles_df.index:
            q_75 = quantiles_df.iloc[step-1, idx_75]
            q_cf = quantiles_df.iloc[step-1, cf_idx]
            q_87 = quantiles_df.iloc[step-1, idx_87]
            
            lower_span = q_cf - q_75
            upper_span = q_87 - q_cf
            
            if lower_span > 0 and upper_span > 0:
                # Ratio should be ~1 for symmetric, <1 for left-skewed, >1 for right-skewed
                metrics[f'cf_asymmetry_h{step}'] = float(upper_span / lower_span)
    
    # Asymmetric loss (weighted by cost ratio)
    if len(y_true) >= 1 and 0.5 in quantiles_df.columns:
        median_idx = np.argmin(np.abs(quantile_levels - 0.5))
        forecast = quantiles_df.iloc[0, median_idx]
        error = y_true[0] - forecast
        
        if error > 0:  # Underprediction (shortage)
            asym_loss = costs.shortage * error
        else:  # Overprediction (holding)
            asym_loss = costs.holding * abs(error)
        
        metrics['asymmetric_loss_h1'] = float(asym_loss)
    
    if len(y_true) >= 2 and 0.5 in quantiles_df.columns:
        median_idx = np.argmin(np.abs(quantile_levels - 0.5))
        forecast = quantiles_df.iloc[1, median_idx]
        error = y_true[1] - forecast
        
        if error > 0:
            asym_loss = costs.shortage * error
        else:
            asym_loss = costs.holding * abs(error)
        
        metrics['asymmetric_loss_h2'] = float(asym_loss)
    
    return metrics


def compute_cost_metric(
    y_true: np.ndarray,
    quantiles_df: pd.DataFrame,
    initial_state: pd.DataFrame,
    costs: Costs,
    lt: LeadTime,
    n_sims: int = 500,
    seed: int = 42,
    realized_cost: bool = True
) -> Dict[str, float]:
    """
    Compute cost-based metric using base-stock policy.
    
    Simulates ordering decision at fold origin and measures costs over horizon.
    
    Args:
        y_true: Actual realized demand for h=1, h=2
        quantiles_df: Forecast quantiles
        initial_state: Starting inventory state
        costs: Cost parameters
        lt: Lead time parameters
        n_sims: Number of Monte Carlo simulations (for forecast-based cost only)
        seed: Random seed
        realized_cost: If True, use y_true as demand; if False, sample from forecast
    
    Returns:
        Dictionary with cost metrics
    """
    rng = np.random.default_rng(seed)
    quantile_levels = quantiles_df.columns.values
    horizon = len(y_true)
    
    # Estimate mu and sigma from h=1 forecast for base-stock
    if 1 in quantiles_df.index:
        q_vals_h1 = quantiles_df.loc[1].values
        u_sample = rng.uniform(0, 1, 1000)
        samples_h1 = np.interp(u_sample, quantile_levels, q_vals_h1)
        mu = np.mean(samples_h1)
        sigma = np.std(samples_h1)
    else:
        mu = np.mean(y_true)
        sigma = np.std(y_true)
    
    # Base-stock order calculation
    critical_fractile = costs.shortage / (costs.holding + costs.shortage)
    L = lt.lead_weeks + lt.review_weeks
    z = stats.norm.ppf(critical_fractile)
    S = mu * L + z * sigma * np.sqrt(L)
    
    # Inventory position at fold origin
    position = initial_state["on_hand"].iloc[0] + initial_state["intransit_1"].iloc[0] + initial_state["intransit_2"].iloc[0]
    order_qty_cont = max(0, S - position)
    
    # INTEGERIZE order quantity (round up to be conservative given asymmetric costs)
    order_qty = int(np.ceil(order_qty_cont))
    
    # Simulate costs
    sim = Simulator(costs, lt)
    
    if realized_cost:
        # Use actual realized demand (single deterministic path)
        # INTEGERIZE demand (round actual demand)
        y_true_int = np.round(y_true).astype(int)
        
        state = initial_state.copy()
        total_cost = 0.0
        total_shortage = 0
        total_holding = 0
        stockouts = 0
        total_demand = 0
        total_satisfied = 0
        
        for step in range(horizon):
            demand_t = pd.Series([float(y_true_int[step])], index=state.index)
            total_demand += y_true_int[step]
            
            # Week 0: place computed order; after: zero recourse
            if step == 0:
                order_t = pd.Series([float(order_qty)], index=state.index)
            else:
                order_t = pd.Series([0.0], index=state.index)
            
            state, cost_dict = sim.step(state, demand_t, order_t)
            total_cost += cost_dict["total"]
            
            # Track component costs and service metrics
            if "shortage" in cost_dict:
                shortage_cost = cost_dict["shortage"]
                holding_cost = cost_dict.get("holding", 0)
                total_shortage += shortage_cost
                total_holding += holding_cost
                
                # Stockout if we had any shortage cost
                if shortage_cost > 0:
                    stockouts += 1
                    
                # Satisfied demand = actual demand - shortage units
                shortage_units = shortage_cost / costs.shortage if costs.shortage > 0 else 0
                satisfied = max(0, y_true_int[step] - shortage_units)
                total_satisfied += satisfied
        
        # Decision-based metrics
        service_level = 1.0 - (stockouts / horizon) if horizon > 0 else 1.0
        fill_rate = total_satisfied / total_demand if total_demand > 0 else 1.0
        
        # Optimal order (if we knew actual demand)
        optimal_order = int(np.sum(y_true_int))
        regret_abs = abs(order_qty - optimal_order)
        
        # Cost should be exact multiple of 0.2
        # (shortage cost = 1.0 * integer units, holding = 0.2 * integer units)
        
        return {
            'expected_cost': float(total_cost),
            'cost_std': 0.0,
            'cost_q05': float(total_cost),
            'cost_q95': float(total_cost),
            'order_qty': float(order_qty),
            'base_stock_level': float(S),
            'shortage_cost': float(total_shortage),
            'holding_cost': float(total_holding),
            'service_level': float(service_level),
            'fill_rate': float(fill_rate),
            'regret_qty': float(regret_abs),
            'optimal_qty': float(optimal_order)
        }
    else:
        # Sample demand scenarios from forecast quantiles (original v1 logic)
        demand_sims = np.zeros((n_sims, horizon))
        
        for step in range(horizon):
            if step + 1 in quantiles_df.index:
                q_vals = quantiles_df.loc[step + 1].values
                u = rng.uniform(0, 1, n_sims)
                demand_sims[:, step] = np.interp(u, quantile_levels, q_vals)
            else:
                demand_sims[:, step] = y_true[step]  # Fallback
        
        total_costs = []
        
        for sim_idx in range(n_sims):
            state = initial_state.copy()
            sim_cost = 0.0
            
            for step in range(horizon):
                demand_t = pd.Series([demand_sims[sim_idx, step]], index=state.index)
                
                # Week 0: place computed order; after: zero recourse
                if step == 0:
                    order_t = pd.Series([order_qty], index=state.index)
                else:
                    order_t = pd.Series([0], index=state.index)
                
                state, cost_dict = sim.step(state, demand_t, order_t)
                sim_cost += cost_dict["total"]
            
            total_costs.append(sim_cost)
        
        return {
            'expected_cost': float(np.mean(total_costs)),
            'cost_std': float(np.std(total_costs)),
            'cost_q05': float(np.quantile(total_costs, 0.05)),
            'cost_q95': float(np.quantile(total_costs, 0.95)),
            'order_qty': float(order_qty),
            'base_stock_level': float(S)
        }


def is_degenerate_forecast(quantiles_df: pd.DataFrame) -> bool:
    """Check if forecast is degenerate (all quantiles identical or all zero)"""
    if quantiles_df.empty:
        return True
    
    # Check if all quantiles are identical (zero width)
    for step in [1, 2]:
        if step in quantiles_df.index:
            q_vals = quantiles_df.loc[step].values
            if len(np.unique(q_vals)) == 1:  # All same value
                return True
            # Check if range is effectively zero
            if np.ptp(q_vals) < 1e-6:
                return True
    
    return False


def compute_sip_cost_metric(
    y_true: np.ndarray,
    quantiles_df: pd.DataFrame,
    initial_state: pd.DataFrame,
    costs: Costs,
    sip_grain: int = 1000,
    exclude_week1: bool = True
) -> Dict[str, float]:
    """
    Compute cost-based metrics using SIP optimization.
    
    Uses PMF convolution to optimize integer order quantity for week 2,
    accounting for uncertainty in both demand and inventory levels.
    
    Args:
        y_true: Actual realized demand [y1, y2]
        quantiles_df: Forecast quantiles (index=step, columns=quantile_levels)
        initial_state: Starting inventory (on_hand, intransit_1, intransit_2)
        costs: Cost parameters
        sip_grain: Maximum support for PMF (0 to grain)
        exclude_week1: If True, only compute costs for week 2
    
    Returns:
        Dictionary with SIP-based cost metrics
    """
    # Extract initial state
    I0 = int(initial_state["on_hand"].iloc[0])
    Q1 = int(initial_state["intransit_1"].iloc[0])  # Arrives at start of week 1
    Q2 = int(initial_state["intransit_2"].iloc[0])  # Arrives at start of week 2
    
    # Convert costs to SIP format
    sip_costs = SIPCosts(holding=costs.holding, shortage=costs.shortage)
    
    # Extract quantile levels and values
    quantile_levels = quantiles_df.columns.values
    
    # Build PMFs for h=1 and h=2
    if 1 not in quantiles_df.index or 2 not in quantiles_df.index:
        return {
            'sip_order_qty': None,
            'sip_expected_cost': None,
            'sip_realized_cost_w2': None,
            'sip_holding_cost_w2': None,
            'sip_shortage_cost_w2': None,
            'sip_service_level_w2': None,
            'sip_fill_rate_w2': None,
            'sip_regret_qty': None
        }
    
    q1_vals = quantiles_df.loc[1].values
    q2_vals = quantiles_df.loc[2].values
    
    pmf_D1 = quantiles_to_pmf(q1_vals, quantile_levels, grain=sip_grain)
    pmf_D2 = quantiles_to_pmf(q2_vals, quantile_levels, grain=sip_grain)
    
    # Compute end-of-week-1 inventory distribution
    pmf_I1_end = convolve_inventory(I0, Q1, pmf_D1)
    
    # Optimize order quantity for week 2
    Q_opt, expected_cost = optimize_order(pmf_I1_end, Q2, pmf_D2, sip_costs, max_Q=sip_grain)
    
    # Compute realized metrics using actual demand
    y1_true = int(np.round(y_true[0]))
    y2_true = int(np.round(y_true[1])) if len(y_true) > 1 else 0
    
    realized_metrics = compute_realized_metrics(
        Q_opt, I0, Q1, Q2, y1_true, y2_true, sip_costs
    )
    
    return {
        'sip_order_qty': float(Q_opt),
        'sip_expected_cost': float(expected_cost),
        'sip_realized_cost_w2': realized_metrics['realized_cost_w2'],
        'sip_holding_cost_w2': realized_metrics['holding_cost_w2'],
        'sip_shortage_cost_w2': realized_metrics['shortage_cost_w2'],
        'sip_service_level_w2': realized_metrics['service_level_w2'],
        'sip_fill_rate_w2': realized_metrics['fill_rate_w2'],
        'sip_regret_qty': realized_metrics['regret_qty']
    }


def evaluate_one(
    task: EvalTask,
    df: pd.DataFrame,
    master_df: Optional[pd.DataFrame],
    checkpoint_dir: Path,
    holdout_weeks: int,
    costs: Costs,
    lt: LeadTime,
    n_sims: int,
    seed: int,
    realized_cost: bool = True,
    skip_degenerate: bool = True,
    use_sip: bool = False,
    sip_grain: int = 1000,
    state_df: Optional[pd.DataFrame] = None
) -> Optional[Dict[str, Any]]:
    """Evaluate a single (model, SKU, fold) combination"""
    
    # Load checkpoint
    checkpoint_path = checkpoint_dir / task.model_name / f"{task.store}_{task.product}" / f"fold_{task.fold_idx}.pkl"
    checkpoint = load_checkpoint(checkpoint_path)
    
    if checkpoint is None:
        return None
    
    quantiles_df = checkpoint.get('quantiles')
    if quantiles_df is None or quantiles_df.empty:
        return None
    
    # Check for degenerate forecast
    if skip_degenerate and is_degenerate_forecast(quantiles_df):
        return None
    
    # Reconstruct test data
    try:
        y_train, X_train, y_test, X_test = prepare_train_test_split(
            df,
            (task.store, task.product),
            holdout_weeks=holdout_weeks,
            fold_idx=task.fold_idx,
            master_df=master_df
        )
        
        if y_train is None or y_test is None or len(y_test) == 0:
            return None
        
        y_true = y_test.values
        
    except Exception as e:
        print(f"Warning: Failed to reconstruct data for {task.key()}: {e}")
        return None
    
    # Compute metrics
    result = {
        'model_name': task.model_name,
        'store': task.store,
        'product': task.product,
        'fold_idx': task.fold_idx,
        'horizon': len(y_true)
    }
    
    try:
        # Point metrics
        point_metrics = compute_point_metrics(y_true, quantiles_df)
        result.update(point_metrics)
        
        # MASE
        if 0.5 in quantiles_df.columns:
            median_preds = quantiles_df[0.5].values[:len(y_true)]
            mase = compute_mase(y_true, median_preds, y_train.values)
            result['mase'] = float(mase) if not np.isnan(mase) else None
        
        # Distribution metrics
        result['pinball_loss'] = float(average_pinball_loss(y_true, quantiles_df))
        result['crps'] = float(average_crps(y_true, quantiles_df))
        
        coverage_results = coverage_metrics(y_true, quantiles_df)
        result.update(coverage_results)
        
        # Shape metrics
        shape_metrics = compute_shape_metrics(quantiles_df, n_sims=min(n_sims, 500), seed=seed)
        result.update(shape_metrics)
        
        # Newsvendor-specific metrics
        newsvendor_metrics = compute_newsvendor_metrics(y_true, quantiles_df, costs)
        result.update(newsvendor_metrics)
        
        # Cost metrics - determine initial state
        if use_sip and state_df is not None:
            # Use actual initial state from state.parquet
            # State is indexed by (store, product, week)
            # For fold_idx, we need the state at the origin of that fold
            fold_origin_week = task.fold_idx
            state_key = (task.store, task.product, fold_origin_week)
            
            if state_key in state_df.index:
                state_row = state_df.loc[state_key]
                initial_state = pd.DataFrame({
                    'on_hand': [state_row['on_hand']],
                    'intransit_1': [state_row.get('intransit_1', 0)],
                    'intransit_2': [state_row.get('intransit_2', 0)]
                }, index=[(task.store, task.product)])
            else:
                # Fallback to zero state if not found
                initial_state = pd.DataFrame({
                    'on_hand': [0],
                    'intransit_1': [0],
                    'intransit_2': [0]
                }, index=[(task.store, task.product)])
        else:
            # Zero inventory baseline for non-SIP evaluation
            initial_state = pd.DataFrame({
                'on_hand': [0],
                'intransit_1': [0],
                'intransit_2': [0]
            }, index=[(task.store, task.product)])
        
        # Compute SIP-based cost if requested
        if use_sip:
            sip_metrics = compute_sip_cost_metric(
                y_true, quantiles_df, initial_state, costs, sip_grain, exclude_week1=True
            )
            result.update(sip_metrics)
        
        # Also compute traditional cost metrics for comparison
        cost_metrics = compute_cost_metric(
            y_true, quantiles_df, initial_state, costs, lt, n_sims, seed + task.fold_idx, realized_cost
        )
        result.update(cost_metrics)
        
    except Exception as e:
        print(f"Warning: Metrics computation failed for {task.key()}: {e}")
        return None
    
    return result


def generate_tasks(
    models: List[str],
    checkpoint_dir: Path,
    holdout_weeks: int,
    progress: ProgressTracker
) -> List[EvalTask]:
    """Generate all evaluation tasks from existing checkpoints"""
    tasks = []
    
    for model_name in models:
        model_dir = checkpoint_dir / model_name
        if not model_dir.exists():
            continue
        
        for sku_dir in model_dir.iterdir():
            if not sku_dir.is_dir():
                continue
            
            try:
                store, product = map(int, sku_dir.name.split('_'))
            except:
                continue
            
            for fold_idx in range(holdout_weeks):
                task = EvalTask(model_name, store, product, fold_idx)
                
                if not progress.is_complete(task):
                    tasks.append(task)
    
    return tasks


def run_evaluation(
    checkpoint_dir: Path,
    demand_path: Path,
    master_path: Optional[Path],
    output_dir: Path,
    progress_file: Path,
    holdout_weeks: int = 8,
    n_jobs: int = 6,
    batch_size: int = 2000,
    n_sims: int = 500,
    resume: bool = True,
    costs_dict: Dict[str, float] = None,
    lead_weeks: int = 2,
    review_weeks: int = 1,
    realized_cost: bool = True,
    skip_degenerate: bool = True,
    out_suffix: str = "",
    use_sip: bool = False,
    sip_grain: int = 1000,
    state_path: Optional[Path] = None
):
    """Run full evaluation with batching and checkpointing"""
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load data
    print("📊 Loading data...")
    df = pd.read_parquet(demand_path)
    master_df = pd.read_parquet(master_path) if master_path and master_path.exists() else None
    
    # Load state data if using SIP
    state_df = None
    if use_sip and state_path and state_path.exists():
        print(f"📦 Loading initial state from {state_path}...")
        state_df = pd.read_parquet(state_path)
        # Ensure proper indexing for fast lookup
        if not isinstance(state_df.index, pd.MultiIndex):
            # Assume columns are store, product, week
            if 'store' in state_df.columns and 'product' in state_df.columns:
                state_df = state_df.set_index(['store', 'product', 'week'])
    
    # Setup costs
    if costs_dict is None:
        costs_dict = {'holding': 0.2, 'shortage': 1.0}
    costs = Costs(**costs_dict)
    lt = LeadTime(lead_weeks=lead_weeks, review_weeks=review_weeks)
    
    # Load progress
    progress = ProgressTracker(progress_file)
    
    # Find all models (or filter if specified)
    all_models = [d.name for d in checkpoint_dir.iterdir() if d.is_dir()]
    # Model filtering would need to be passed as parameter - for now use all
    models = all_models
    print(f"🤖 Found {len(models)} models: {models}")
    
    # Generate tasks
    print(f"📋 Generating tasks (holdout={holdout_weeks} folds)...")
    tasks = generate_tasks(models, checkpoint_dir, holdout_weeks, progress)
    
    print(f"✅ Total tasks: {len(tasks)}")
    print(f"✅ Already completed: {len(progress.completed)}")
    print(f"⏳ To run: {len(tasks)}")
    
    if len(tasks) == 0:
        print("All tasks complete!")
        return
    
    # Process in batches
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    
    for batch_start in range(0, len(tasks), batch_size):
        if INTERRUPTED:
            print("⚠️  Interrupted, saving progress...")
            break
        
        batch_end = min(batch_start + batch_size, len(tasks))
        batch_tasks = tasks[batch_start:batch_end]
        
        print(f"\n{'='*60}")
        print(f"📦 Batch {batch_start//batch_size + 1}: tasks {batch_start+1}-{batch_end} of {len(tasks)}")
        print(f"{'='*60}")
        
        # Parallel evaluation
        batch_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
            delayed(evaluate_one)(
                task, df, master_df, checkpoint_dir, holdout_weeks,
                costs, lt, n_sims, 42, realized_cost, skip_degenerate,
                use_sip, sip_grain, state_df
            ) for task in batch_tasks
        )
        
        # Filter out None results and update progress
        valid_results = []
        for task, result in zip(batch_tasks, batch_results):
            if result is not None:
                valid_results.append(result)
                progress.mark_complete(task)
        
        all_results.extend(valid_results)
        
        # Save batch results atomically
        if valid_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pid = os.getpid()
            suffix = f"_{out_suffix}" if out_suffix else ""
            batch_file = output_dir / f"eval_folds{suffix}_part-{timestamp}-{pid}-{batch_start//batch_size}.parquet"
            temp_file = batch_file.with_suffix('.parquet.tmp')
            
            batch_df = pd.DataFrame(valid_results)
            batch_df.to_parquet(temp_file)
            temp_file.rename(batch_file)
            
            print(f"✅ Saved {len(valid_results)} results to {batch_file.name}")
        
        # Save progress
        progress.save()
        
        print(f"✓ Progress: {len(progress.completed)} / {len(progress.completed) + len(tasks)} tasks")
    
    # Consolidate all results
    print(f"\n{'='*60}")
    print("📊 Consolidating results...")
    print(f"{'='*60}")
    
    if all_results:
        final_df = pd.DataFrame(all_results)
        suffix = f"_{out_suffix}" if out_suffix else ""
        final_path = output_dir / f"eval_folds{suffix}.parquet"
        final_df.to_parquet(final_path)
        print(f"✅ Saved consolidated results: {final_path}")
        print(f"   Total evaluations: {len(final_df)}")
        print(f"   Models: {final_df['model_name'].nunique()}")
        print(f"   SKUs: {final_df[['store', 'product']].drop_duplicates().shape[0]}")


def aggregate_results(input_path: Path, output_dir: Path, out_suffix: str = ""):
    """Aggregate per-fold results to per-SKU and overall leaderboards"""
    print("📊 Aggregating results...")
    
    df = pd.read_parquet(input_path)
    
    # Aggregate per (SKU, model)
    # Use SUMS for cost (not means), means for accuracy metrics
    agg_funcs = {
        'mae': 'mean',
        'mape': 'mean',
        'mase': 'mean',
        'bias': 'mean',
        'rmse': 'mean',
        'pinball_loss': 'mean',
        'crps': 'mean',
        'coverage_80': 'mean',
        'coverage_90': 'mean',
        'coverage_95': 'mean',
        'width_80': 'mean',
        'width_90': 'mean',
        'width_95': 'mean',
        'expected_cost': 'sum',  # TOTAL cost across folds
        'shortage_cost': 'sum',
        'holding_cost': 'sum',
        'service_level': 'mean',
        'fill_rate': 'mean',
        'regret_qty': 'sum',
        'pinball_cf_h1': 'mean',  # Newsvendor metrics
        'pinball_cf_h2': 'mean',
        'hit_cf_h1': 'mean',  # Calibration at critical fractile
        'hit_cf_h2': 'mean',
        'local_width_h1': 'mean',
        'local_width_h2': 'mean',
        'quantile_gradient_h1': 'mean',
        'quantile_gradient_h2': 'mean',
        'cf_asymmetry_h1': 'mean',
        'cf_asymmetry_h2': 'mean',
        'asymmetric_loss_h1': 'sum',
        'asymmetric_loss_h2': 'sum',
        # SIP metrics
        'sip_expected_cost': 'sum',
        'sip_realized_cost_w2': 'sum',
        'sip_holding_cost_w2': 'sum',
        'sip_shortage_cost_w2': 'sum',
        'sip_service_level_w2': 'mean',
        'sip_fill_rate_w2': 'mean',
        'sip_regret_qty': 'sum',
        'fold_idx': 'count'  # Number of folds
    }
    
    # Filter to only existing columns
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}
    
    agg_df = df.groupby(['store', 'product', 'model_name']).agg(agg_funcs).reset_index()
    agg_df.rename(columns={'fold_idx': 'n_folds'}, inplace=True)
    
    # Save aggregated
    suffix = f"_{out_suffix}" if out_suffix else ""
    agg_path = output_dir / f"eval_agg{suffix}.parquet"
    agg_df.to_parquet(agg_path)
    print(f"✅ Saved aggregated results: {agg_path}")
    
    # Create leaderboards
    # Overall leaderboard: SUM for costs, MEAN for accuracy metrics
    overall_agg = {}
    for col in agg_df.columns:
        if col in ['model_name', 'store', 'product', 'n_folds']:
            continue
        elif col in ['expected_cost', 'shortage_cost', 'holding_cost', 'regret_qty', 
                     'asymmetric_loss_h1', 'asymmetric_loss_h2',
                     'sip_expected_cost', 'sip_realized_cost_w2', 
                     'sip_holding_cost_w2', 'sip_shortage_cost_w2', 'sip_regret_qty']:
            # TOTAL cost across all SKUs
            overall_agg[col] = 'sum'
        else:
            # AVERAGE accuracy/density metrics
            overall_agg[col] = 'mean'
    
    overall = agg_df.groupby('model_name').agg(overall_agg).reset_index()
    
    # Rank by SIP cost if available, otherwise expected_cost
    rank_col = 'sip_realized_cost_w2' if 'sip_realized_cost_w2' in overall.columns else 'expected_cost'
    if rank_col in overall.columns:
        overall = overall.sort_values(rank_col)
    
    leaderboard_path = output_dir / f"leaderboards{suffix}.parquet"
    overall.to_parquet(leaderboard_path)
    print(f"✅ Saved leaderboard: {leaderboard_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    rank_label = "SIP realized cost (week 2)" if rank_col == 'sip_realized_cost_w2' else "TOTAL expected cost"
    print(f"🏆 Overall Leaderboard (by {rank_label})")
    print(f"{'='*60}")
    
    # Build display columns
    display_cols = ['model_name']
    if 'sip_realized_cost_w2' in overall.columns:
        display_cols.extend(['sip_realized_cost_w2', 'sip_service_level_w2', 'sip_fill_rate_w2'])
    if 'expected_cost' in overall.columns:
        display_cols.append('expected_cost')
    if 'mae' in overall.columns:
        display_cols.append('mae')
    if 'pinball_cf_h1' in overall.columns:
        display_cols.append('pinball_cf_h1')
    if 'service_level' in overall.columns:
        display_cols.append('service_level')
    if 'coverage_90' in overall.columns:
        display_cols.append('coverage_90')
    
    display_cols = [c for c in display_cols if c in overall.columns]
    print(overall[display_cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Evaluate forecast models with cost-based ranking")
    
    # Paths
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('models/checkpoints'))
    parser.add_argument('--demand-path', type=Path, default=Path('data/processed/demand_imputed.parquet'))
    parser.add_argument('--master-path', type=Path, default=Path('data/processed/master.parquet'))
    parser.add_argument('--output-dir', type=Path, default=Path('models/results'))
    parser.add_argument('--progress-file', type=Path, default=Path('models/results/eval_progress.json'))
    
    # Evaluation config
    parser.add_argument('--holdout', type=int, default=8, help='Number of rolling-origin folds')
    parser.add_argument('--n-sims', type=int, default=500, help='Monte Carlo samples for cost evaluation')
    
    # Resource control
    parser.add_argument('--n-jobs', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--cpu-fraction', type=float, default=0.5, help='Fraction of CPU cores to use')
    parser.add_argument('--omp-threads', type=int, default=1, help='BLAS threads per worker')
    parser.add_argument('--batch-size', type=int, default=2000, help='Tasks per batch')
    
    # Execution
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--aggregate', action='store_true', help='Only run aggregation step')
    
    # V2 features
    parser.add_argument('--realized-cost', action='store_true', default=True, help='Use realized demand for cost (default: True)')
    parser.add_argument('--forecast-cost', dest='realized_cost', action='store_false', help='Use forecast samples for cost (v1 behavior)')
    parser.add_argument('--skip-degenerate', action='store_true', default=True, help='Skip degenerate forecasts (default: True)')
    parser.add_argument('--include-degenerate', dest='skip_degenerate', action='store_false', help='Include degenerate forecasts')
    parser.add_argument('--out-suffix', type=str, default='', help='Output file suffix (e.g., "v2")')
    
    # Cost parameters
    parser.add_argument('--holding-cost', type=float, default=0.2)
    parser.add_argument('--shortage-cost', type=float, default=1.0)
    parser.add_argument('--lead-weeks', type=int, default=2)
    parser.add_argument('--review-weeks', type=int, default=1)
    
    args = parser.parse_args()
    
    # Set BLAS threads
    os.environ['OMP_NUM_THREADS'] = str(args.omp_threads)
    os.environ['MKL_NUM_THREADS'] = str(args.omp_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(args.omp_threads)
    
    # Compute n_jobs if not specified
    if args.n_jobs is None:
        import subprocess
        result = subprocess.run(['sysctl', '-n', 'hw.logicalcpu'], capture_output=True, text=True)
        total_cores = int(result.stdout.strip())
        args.n_jobs = max(1, int(total_cores * args.cpu_fraction))
    
    print(f"🚀 Model Evaluation")
    print(f"   Holdout: {args.holdout} folds")
    print(f"   Workers: {args.n_jobs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Simulations: {args.n_sims}")
    print(f"   Costs: holding={args.holding_cost}, shortage={args.shortage_cost}")
    
    if args.aggregate:
        # Only aggregate existing results
        suffix = f"_{args.out_suffix}" if args.out_suffix else ""
        input_path = args.output_dir / f"eval_folds{suffix}.parquet"
        if not input_path.exists():
            print(f"❌ No results file found at {input_path}")
            return
        aggregate_results(input_path, args.output_dir, args.out_suffix)
    else:
        # Run evaluation
        costs_dict = {'holding': args.holding_cost, 'shortage': args.shortage_cost}
        
        run_evaluation(
            checkpoint_dir=args.checkpoint_dir,
            demand_path=args.demand_path,
            master_path=args.master_path,
            output_dir=args.output_dir,
            progress_file=args.progress_file,
            holdout_weeks=args.holdout,
            n_jobs=args.n_jobs,
            batch_size=args.batch_size,
            n_sims=args.n_sims,
            resume=args.resume,
            costs_dict=costs_dict,
            lead_weeks=args.lead_weeks,
            review_weeks=args.review_weeks,
            realized_cost=args.realized_cost,
            skip_degenerate=args.skip_degenerate,
            out_suffix=args.out_suffix
        )
        
        # Auto-aggregate if evaluation completed
        suffix = f"_{args.out_suffix}" if args.out_suffix else ""
        final_path = args.output_dir / f"eval_folds{suffix}.parquet"
        if final_path.exists():
            aggregate_results(final_path, args.output_dir, args.out_suffix)


if __name__ == '__main__':
    main()

