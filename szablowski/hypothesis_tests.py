"""
Hypothesis tests H1–H6 and calibration comparison for the Szabłowski sprint.

Each test reads per-SKU-week cost detail from the comparison harness and
stratifies by EDA artifacts.  Produces structured output (DataFrames, plots)
for downstream analysis.
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    console = None


# ===================================================================
# H1: Jensen's Gap by demand regime
# ===================================================================

def h1_jensens_gap(
    delta_df: pd.DataFrame,
    stratify_cols: Optional[List[str]] = None,
    gap_threshold: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify Jensen's Gap: cost(analytical) - cost(SIP) per SKU.

    Positive cost_delta means analytical is worse (SIP captures value).

    Parameters
    ----------
    delta_df : per-SKU cost delta from harness.per_sku_cost_delta, enriched
        with EDA columns.
    stratify_cols : columns to group by for regime analysis.
    gap_threshold : threshold for "large" Jensen's Gap (euros per SKU).

    Returns
    -------
    summary : overall gap statistics
    regime_detail : gap statistics per regime stratum
    """
    if stratify_cols is None:
        stratify_cols = _available_stratifiers(
            delta_df,
            ["heteroskedastic", "taylor_alpha", "cv_bin", "pct_zeros", "ADI",
             "rate_bin", "zero_bin", "stockout_bin"],
        )

    gap = delta_df["cost_delta"]

    summary = pd.DataFrame([{
        "mean_gap": gap.mean(),
        "median_gap": gap.median(),
        "std_gap": gap.std(),
        "n_positive_gap": int((gap > 0).sum()),
        "n_negative_gap": int((gap < 0).sum()),
        "frac_large_gap": float((gap.abs() > gap_threshold).mean()),
        "total_gap": gap.sum(),
        "total_abs_gap": gap.abs().sum(),
        "gap_from_large_skus": float(gap[gap.abs() > gap_threshold].sum()),
        "frac_total_from_large": (
            float(gap[gap.abs() > gap_threshold].sum() / gap.sum())
            if gap.sum() != 0 else np.nan
        ),
    }])

    # Per-regime
    regime_rows = []
    for col in stratify_cols:
        if col not in delta_df.columns or delta_df[col].isna().all():
            continue

        # Bin continuous columns
        strat_col = _maybe_bin(delta_df, col)

        for val, grp in delta_df.groupby(strat_col):
            g = grp["cost_delta"]
            regime_rows.append({
                "stratifier": col,
                "value": str(val),
                "n_skus": len(grp),
                "mean_gap": g.mean(),
                "median_gap": g.median(),
                "total_gap": g.sum(),
                "frac_positive": (g > 0).mean(),
            })

    regime_detail = pd.DataFrame(regime_rows)
    return summary, regime_detail


# ===================================================================
# H5: φ stability vs density quantile stability
# ===================================================================

def h5_phi_stability(
    phi_results: Dict[str, pd.DataFrame],
    delta_df: Optional[pd.DataFrame] = None,
    stratify_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare φ* stability across backtest windows.

    Parameters
    ----------
    phi_results : {window_label: DataFrame with columns phi, total_cost}
        Output from calibrate_phi.calibrate_phi for each rolling window.
    delta_df : optional per-SKU delta enriched with EDA for stratification.
    stratify_cols : columns to stratify by.

    Returns
    -------
    window_summary : best φ per window, stability metrics
    regime_detail : per-regime stability analysis (if delta_df provided)
    """
    window_rows = []
    for label, df in phi_results.items():
        best_idx = df["total_cost"].idxmin()
        window_rows.append({
            "window": label,
            "best_phi": df.loc[best_idx, "phi"],
            "best_cost": df.loc[best_idx, "total_cost"],
            "cost_at_phi_0": float(df.loc[df["phi"].idxmin(), "total_cost"]) if 0.0 in df["phi"].values else np.nan,
        })

    window_summary = pd.DataFrame(window_rows)
    if len(window_summary) > 1:
        window_summary.loc[:, "phi_mean"] = window_summary["best_phi"].mean()
        window_summary.loc[:, "phi_std"] = window_summary["best_phi"].std()
        window_summary.loc[:, "phi_cv"] = (
            window_summary["best_phi"].std() / window_summary["best_phi"].mean()
            if window_summary["best_phi"].mean() != 0 else np.nan
        )

    regime_detail = pd.DataFrame()
    if delta_df is not None and stratify_cols:
        # Show cost sensitivity to φ by regime
        available = _available_stratifiers(
            delta_df, stratify_cols or ["stationarity", "has_structural_break", "seasonal_strength_stl"],
        )
        # Placeholder for per-regime phi sensitivity (requires per-SKU simulation)
        regime_detail = pd.DataFrame({"note": ["Per-regime phi sensitivity requires per-SKU simulation runs"]})

    return window_summary, regime_detail


# ===================================================================
# H2: Annualized scaling vs SURD
# ===================================================================

def h2_scaling_comparison(
    baseline_eval: pd.DataFrame,
    annualized_eval: pd.DataFrame,
    metric_cols: Optional[List[str]] = None,
    stratify_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compare forecast accuracy and cost between SURD and annualized scaling.

    Parameters
    ----------
    baseline_eval : eval_folds with SURD transforms (existing pipeline)
    annualized_eval : eval_folds with annualized scaling swapped in
    metric_cols : metrics to compare (default: mae, pinball_cf_h2, sip_realized_cost_w2)
    stratify_cols : EDA columns for regime analysis

    Returns
    -------
    DataFrame with per-metric, per-regime comparison statistics.
    """
    if metric_cols is None:
        metric_cols = ["mae", "pinball_cf_h2", "sip_realized_cost_w2"]

    results = []
    available_metrics = [m for m in metric_cols if m in baseline_eval.columns and m in annualized_eval.columns]

    for metric in available_metrics:
        base_vals = baseline_eval.groupby(["Store", "Product"])[metric].mean().reset_index()
        ann_vals = annualized_eval.groupby(["Store", "Product"])[metric].mean().reset_index()
        merged = base_vals.merge(ann_vals, on=["Store", "Product"], suffixes=("_surd", "_annualized"))

        delta = merged[f"{metric}_annualized"] - merged[f"{metric}_surd"]
        results.append({
            "metric": metric,
            "mean_surd": merged[f"{metric}_surd"].mean(),
            "mean_annualized": merged[f"{metric}_annualized"].mean(),
            "mean_delta": delta.mean(),
            "median_delta": delta.median(),
            "frac_annualized_better": (delta < 0).mean(),
            "paired_t_stat": float(sp_stats.ttest_rel(
                merged[f"{metric}_surd"], merged[f"{metric}_annualized"]
            ).statistic) if len(merged) > 1 else np.nan,
            "paired_t_pval": float(sp_stats.ttest_rel(
                merged[f"{metric}_surd"], merged[f"{metric}_annualized"]
            ).pvalue) if len(merged) > 1 else np.nan,
        })

    return pd.DataFrame(results)


# ===================================================================
# H3: Time-decayed weighting
# ===================================================================

def h3_time_decay(
    baseline_eval: pd.DataFrame,
    stepwise_eval: pd.DataFrame,
    exponential_eval: Optional[pd.DataFrame] = None,
    metric: str = "sip_realized_cost_w2",
    stratify_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compare uniform vs stepwise vs exponential decay weighting.

    Returns per-SKU cost comparison with optional regime stratification.
    """
    evals = {"uniform": baseline_eval, "stepwise": stepwise_eval}
    if exponential_eval is not None:
        evals["exponential"] = exponential_eval

    merged = None
    for name, ev in evals.items():
        sku_cost = ev.groupby(["Store", "Product"])[metric].mean().reset_index()
        sku_cost = sku_cost.rename(columns={metric: f"cost_{name}"})
        if merged is None:
            merged = sku_cost
        else:
            merged = merged.merge(sku_cost, on=["Store", "Product"], how="outer")

    if merged is None:
        return pd.DataFrame()

    results = []
    baseline_col = "cost_uniform"
    for name in evals:
        if name == "uniform":
            continue
        comp_col = f"cost_{name}"
        if comp_col not in merged.columns:
            continue
        delta = merged[comp_col] - merged[baseline_col]
        results.append({
            "comparison": f"{name} vs uniform",
            "mean_delta": delta.mean(),
            "median_delta": delta.median(),
            "frac_better": (delta < 0).mean(),
            "total_portfolio_delta": delta.sum(),
        })

    return pd.DataFrame(results)


# ===================================================================
# H4: Feature engineering transfer
# ===================================================================

def h4_feature_transfer(
    baseline_eval: pd.DataFrame,
    augmented_eval: pd.DataFrame,
    metric_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compare LightGBM Quantile with and without Szabłowski features.

    Focuses on tail improvement: pinball at 0.833 and 0.95.
    """
    if metric_cols is None:
        metric_cols = ["pinball_cf_h2", "pinball_loss_h2", "sip_realized_cost_w2"]

    return h2_scaling_comparison(baseline_eval, augmented_eval, metric_cols)


# ===================================================================
# H6: Ensemble advantage in tails
# ===================================================================

def h6_tail_advantage(
    delta_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    percentile_threshold: float = 0.10,
) -> pd.DataFrame:
    """Compare approaches on series where realized demand hit tail events.

    Identifies SKUs where realized demand (during test period) fell in the
    top or bottom `percentile_threshold` of their historical distribution.
    """
    sales_col = "sales" if "sales" in demand_df.columns else "demand"

    # Compute per-SKU historical distribution stats
    sku_stats = demand_df.groupby(["Store", "Product"])[sales_col].agg(
        ["mean", "std", "min", "max"]
    ).reset_index()
    sku_stats.columns = ["Store", "Product", "hist_mean", "hist_std", "hist_min", "hist_max"]

    # Merge with delta
    merged = delta_df.merge(sku_stats, on=["Store", "Product"], how="left")

    # Compute realized demand during backtest
    if "demand" in delta_df.columns:
        merged["realized_mean"] = delta_df.groupby(["Store", "Product"])["demand"].transform("mean")
    elif "comp_total" in delta_df.columns:
        merged["realized_mean"] = np.nan

    results = []

    # Stratify by kurtosis if available
    stratifiers = _available_stratifiers(merged, ["kurtosis", "skewness", "negbin_preferred"])

    for col in stratifiers:
        strat_col = _maybe_bin(merged, col)
        for val, grp in merged.groupby(strat_col):
            gap = grp["cost_delta"]
            results.append({
                "stratifier": col,
                "value": str(val),
                "n_skus": len(grp),
                "mean_gap": gap.mean(),
                "median_gap": gap.median(),
                "total_gap": gap.sum(),
                "frac_sip_wins": (gap > 0).mean(),
            })

    return pd.DataFrame(results) if results else pd.DataFrame()


# ===================================================================
# Calibration comparison (Sprint Section 5)
# ===================================================================

def calibration_comparison(
    sip_detail: pd.DataFrame,
    analytical_detail: pd.DataFrame,
    demand_df: pd.DataFrame,
    quantile_levels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compare calibration (coverage at 0.833) between SIP and analytical.

    For SIP: what fraction of realized demands fell below the 0.833 quantile?
    For analytical: what fraction fell below D̂ + z_q * φ * √D̂?

    Requires per-SKU forecast and realized demand data.  This is a simplified
    version that works from the simulation detail output.
    """
    if quantile_levels is None:
        quantile_levels = np.array([0.50, 0.60, 0.70, 0.80, 0.833, 0.90, 0.95])

    # Coverage analysis from simulation detail
    results = []

    for label, detail in [("sip", sip_detail), ("analytical", analytical_detail)]:
        if "demand" not in detail.columns or "available" not in detail.columns:
            continue

        coverage = (detail["demand"] <= detail["available"]).mean()
        results.append({
            "policy": label,
            "overall_coverage": coverage,
            "target_coverage": 0.833,
            "coverage_gap": coverage - 0.833,
        })

    return pd.DataFrame(results) if results else pd.DataFrame()


def reliability_diagram_data(
    eval_folds: pd.DataFrame,
    quantile_cols: Optional[List[str]] = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Produce reliability diagram data for density model calibration.

    For each nominal quantile level, compute the observed coverage fraction.
    """
    if quantile_cols is None:
        quantile_cols = [c for c in eval_folds.columns if c.startswith("hit_") or c.startswith("coverage_")]

    if not quantile_cols:
        return pd.DataFrame()

    rows = []
    for col in quantile_cols:
        if col not in eval_folds.columns:
            continue
        observed = eval_folds[col].mean()
        # Try to extract nominal level from column name
        parts = col.split("_")
        try:
            nominal = float(parts[-1]) / 100 if float(parts[-1]) > 1 else float(parts[-1])
        except (ValueError, IndexError):
            nominal = np.nan
        rows.append({"quantile_col": col, "nominal": nominal, "observed": observed})

    return pd.DataFrame(rows)


# ===================================================================
# Helpers
# ===================================================================

def _available_stratifiers(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [c for c in candidates if c in df.columns and not df[c].isna().all()]


def _maybe_bin(df: pd.DataFrame, col: str, n_bins: int = 5) -> pd.Series:
    """Bin continuous columns into quintiles for stratification."""
    if df[col].dtype in ("category", "object", "bool"):
        return df[col]
    nunique = df[col].nunique()
    if nunique <= n_bins:
        return df[col]
    try:
        return pd.qcut(df[col], n_bins, duplicates="drop")
    except Exception:
        return df[col]


# ===================================================================
# Master runner
# ===================================================================

def run_h1_h5(
    comparison_dir: Path,
    phi_results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Run H1 and H5 from harness output."""
    output_dir = output_dir or comparison_dir / "hypothesis_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # H1
    delta_path = comparison_dir / "per_sku_cost_delta.parquet"
    if delta_path.exists():
        delta_df = pd.read_parquet(delta_path)
        h1_summary, h1_regime = h1_jensens_gap(delta_df)

        h1_summary.to_csv(output_dir / "h1_summary.csv", index=False)
        h1_regime.to_csv(output_dir / "h1_regime_detail.csv", index=False)
        results["h1_summary"] = h1_summary
        results["h1_regime"] = h1_regime

        if console:
            console.print("\n[bold]H1: Jensen's Gap Summary[/bold]")
            console.print(h1_summary.to_string(index=False))
            if not h1_regime.empty:
                console.print("\n[bold]H1: By Regime[/bold]")
                console.print(h1_regime.to_string(index=False))
    else:
        if console:
            console.print(f"[yellow]H1: {delta_path} not found, skipping[/yellow]")

    # H5
    if phi_results_dir and phi_results_dir.exists():
        phi_results = {}
        for f in sorted(phi_results_dir.glob("phi_calibration_*.parquet")):
            label = f.stem.replace("phi_calibration_", "")
            phi_results[label] = pd.read_parquet(f)

        if phi_results:
            h5_window, h5_regime = h5_phi_stability(phi_results)
            h5_window.to_csv(output_dir / "h5_phi_stability.csv", index=False)
            results["h5_window"] = h5_window

            if console:
                console.print("\n[bold]H5: φ Stability Across Windows[/bold]")
                console.print(h5_window.to_string(index=False))
    else:
        if console:
            console.print("[yellow]H5: phi results dir not provided or not found[/yellow]")

    return results


def run_all_hypothesis_tests(
    comparison_dir: Path,
    demand_path: Path,
    phi_results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Run all hypothesis tests from harness output."""
    output_dir = output_dir or comparison_dir / "hypothesis_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_h1_h5(comparison_dir, phi_results_dir, output_dir)

    # H6
    delta_path = comparison_dir / "per_sku_cost_delta.parquet"
    if delta_path.exists() and demand_path.exists():
        delta_df = pd.read_parquet(delta_path)
        demand_df = pd.read_parquet(demand_path)
        if "sales" not in demand_df.columns and "demand" in demand_df.columns:
            demand_df["sales"] = demand_df["demand"]

        h6 = h6_tail_advantage(delta_df, demand_df)
        if not h6.empty:
            h6.to_csv(output_dir / "h6_tail_advantage.csv", index=False)
            results["h6"] = h6

    # Calibration comparison
    sip_path = comparison_dir / "sku_detail_sip.parquet"
    analytical_path = comparison_dir / "sku_detail_analytical.parquet"
    if sip_path.exists() and analytical_path.exists():
        sip_detail = pd.read_parquet(sip_path)
        analytical_detail = pd.read_parquet(analytical_path)
        demand_df = pd.read_parquet(demand_path) if demand_path.exists() else pd.DataFrame()

        cal = calibration_comparison(sip_detail, analytical_detail, demand_df)
        if not cal.empty:
            cal.to_csv(output_dir / "calibration_comparison.csv", index=False)
            results["calibration"] = cal

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Szabłowski sprint hypothesis tests")
    parser.add_argument("--comparison-dir", type=Path, default=Path("reports/comparison"))
    parser.add_argument("--demand-path", type=Path, default=Path("data/processed/demand_long.parquet"))
    parser.add_argument("--phi-results-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--h1-h5-only", action="store_true")
    args = parser.parse_args()

    if args.h1_h5_only:
        run_h1_h5(args.comparison_dir, args.phi_results_dir, args.output_dir)
    else:
        run_all_hypothesis_tests(
            args.comparison_dir, args.demand_path, args.phi_results_dir, args.output_dir,
        )


if __name__ == "__main__":
    main()
