"""
Residual diagnostics for forecast error characterization.

Analyses CatBoost (and optionally other model) residuals to gate the
neural sprint decision: should we wrap CatBoost in SLURP bootstrap or
build native distributional outputs?

Diagnostics:
  A. Per-SKU residual autocorrelation (Ljung-Box)
  B. Heteroskedasticity of residuals (Taylor's law, Breusch-Pagan)
  C. Tail behaviour (kurtosis, Jarque-Bera, sigma-exceedance)
  D. Cross-model comparison via existing eval_folds
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats as sp_stats

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
except ImportError:
    acorr_ljungbox = None

try:
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
except ImportError:
    het_breuschpagan = None
    OLS = None
    add_constant = None


# ---------------------------------------------------------------------------
# A. Per-SKU residual ACF
# ---------------------------------------------------------------------------

def _ljung_box_one_sku(
    store: int,
    product: int,
    residuals: np.ndarray,
    lags: List[int],
) -> dict:
    row: dict = {"Store": store, "Product": product, "n_obs": len(residuals)}
    if acorr_ljungbox is None or len(residuals) < max(lags) + 2:
        for lag in lags:
            row[f"lb_stat_{lag}"] = np.nan
            row[f"lb_pval_{lag}"] = np.nan
        row["any_significant_acf"] = False
        return row

    try:
        result = acorr_ljungbox(residuals, lags=lags, return_df=True)
        any_sig = False
        for lag in lags:
            if lag in result.index:
                row[f"lb_stat_{lag}"] = float(result.loc[lag, "lb_stat"])
                row[f"lb_pval_{lag}"] = float(result.loc[lag, "lb_pvalue"])
                if result.loc[lag, "lb_pvalue"] < 0.05:
                    any_sig = True
            else:
                row[f"lb_stat_{lag}"] = np.nan
                row[f"lb_pval_{lag}"] = np.nan
        row["any_significant_acf"] = any_sig
    except Exception:
        for lag in lags:
            row[f"lb_stat_{lag}"] = np.nan
            row[f"lb_pval_{lag}"] = np.nan
        row["any_significant_acf"] = False

    return row


def residual_acf(
    residuals_df: pd.DataFrame,
    lags: Optional[List[int]] = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Ljung-Box test on per-SKU residuals at specified lags.

    Parameters
    ----------
    residuals_df : DataFrame
        Columns: Store, Product, week, residual (actual - forecast).
    lags : list of int
        Lags to test (default [1, 4, 12]).
    """
    if lags is None:
        lags = [1, 4, 12]

    groups = []
    for (s, p), grp in residuals_df.groupby(["Store", "Product"]):
        r = grp.sort_values("week")["residual"].values
        groups.append((s, p, r))

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_ljung_box_one_sku)(s, p, r, lags)
        for s, p, r in groups
    )
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# B. Heteroskedasticity
# ---------------------------------------------------------------------------

def _hetero_one_sku(
    store: int,
    product: int,
    demand: np.ndarray,
    residuals: np.ndarray,
) -> dict:
    row: dict = {"Store": store, "Product": product}
    n = len(residuals)

    row["resid_mean"] = float(np.mean(residuals))
    row["resid_std"] = float(np.std(residuals, ddof=1)) if n > 1 else np.nan
    row["resid_var"] = float(np.var(residuals, ddof=1)) if n > 1 else np.nan
    row["demand_mean"] = float(np.mean(demand))

    # Taylor's law on residuals: Var(resid) vs E[demand]
    # Computed at aggregate level; per-SKU is a single point
    row["abs_resid_mean"] = float(np.mean(np.abs(residuals)))

    # Breusch-Pagan test: residual^2 ~ demand level
    if het_breuschpagan is not None and n >= 5:
        try:
            resid_sq = residuals ** 2
            X = add_constant(demand)
            result = het_breuschpagan(resid_sq, X)
            row["bp_stat"] = float(result[0])
            row["bp_pval"] = float(result[1])
            row["heteroskedastic_resid"] = bool(result[1] < 0.05)
        except Exception:
            row["bp_stat"] = np.nan
            row["bp_pval"] = np.nan
            row["heteroskedastic_resid"] = False
    else:
        row["bp_stat"] = np.nan
        row["bp_pval"] = np.nan
        row["heteroskedastic_resid"] = False

    return row


def residual_heteroskedasticity(
    residuals_df: pd.DataFrame,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Breusch-Pagan test and variance-vs-demand for each SKU."""
    groups = []
    for (s, p), grp in residuals_df.groupby(["Store", "Product"]):
        grp = grp.sort_values("week")
        groups.append((s, p, grp["demand"].values, grp["residual"].values))

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_hetero_one_sku)(s, p, d, r)
        for s, p, d, r in groups
    )
    return pd.DataFrame(results)


def taylors_law_on_residuals(
    hetero_df: pd.DataFrame,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Bin SKUs by demand_mean, compute Var(residual) per bin.

    Returns a DataFrame with bin, mean_demand, var_residual, log-log slope.
    """
    df = hetero_df.dropna(subset=["demand_mean", "resid_var"])
    df = df[df["demand_mean"] > 0]
    if len(df) < n_bins:
        return pd.DataFrame()

    df["demand_bin"] = pd.qcut(df["demand_mean"], n_bins, duplicates="drop")
    binned = df.groupby("demand_bin", observed=True).agg(
        mean_demand=("demand_mean", "mean"),
        mean_resid_var=("resid_var", "mean"),
        n_skus=("Store", "count"),
    ).reset_index()

    binned = binned[binned["mean_resid_var"] > 0]
    if len(binned) >= 2:
        log_d = np.log(binned["mean_demand"].values)
        log_v = np.log(binned["mean_resid_var"].values)
        slope, intercept, r, p, se = sp_stats.linregress(log_d, log_v)
        binned["taylor_alpha"] = slope / 2.0
        binned["taylor_r2"] = r ** 2
    else:
        binned["taylor_alpha"] = np.nan
        binned["taylor_r2"] = np.nan

    return binned


# ---------------------------------------------------------------------------
# C. Tail behaviour
# ---------------------------------------------------------------------------

def _tail_one_sku(
    store: int,
    product: int,
    residuals: np.ndarray,
) -> dict:
    row: dict = {"Store": store, "Product": product}
    n = len(residuals)

    if n < 4:
        row.update({
            "kurtosis": np.nan, "skewness": np.nan,
            "jb_stat": np.nan, "jb_pval": np.nan,
            "frac_beyond_2sigma": np.nan, "frac_beyond_3sigma": np.nan,
            "normal_residuals": False,
        })
        return row

    row["kurtosis"] = float(sp_stats.kurtosis(residuals, fisher=True))
    row["skewness"] = float(sp_stats.skew(residuals))

    try:
        jb_stat, jb_pval = sp_stats.jarque_bera(residuals)
        row["jb_stat"] = float(jb_stat)
        row["jb_pval"] = float(jb_pval)
        row["normal_residuals"] = bool(jb_pval >= 0.05)
    except Exception:
        row["jb_stat"] = np.nan
        row["jb_pval"] = np.nan
        row["normal_residuals"] = False

    std = np.std(residuals, ddof=1)
    if std > 0:
        z = np.abs(residuals - np.mean(residuals)) / std
        row["frac_beyond_2sigma"] = float(np.mean(z > 2))
        row["frac_beyond_3sigma"] = float(np.mean(z > 3))
    else:
        row["frac_beyond_2sigma"] = 0.0
        row["frac_beyond_3sigma"] = 0.0

    return row


def residual_tails(
    residuals_df: pd.DataFrame,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Kurtosis, skewness, Jarque-Bera, sigma-exceedance per SKU."""
    groups = []
    for (s, p), grp in residuals_df.groupby(["Store", "Product"]):
        groups.append((s, p, grp.sort_values("week")["residual"].values))

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_tail_one_sku)(s, p, r)
        for s, p, r in groups
    )
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# D. Cross-model comparison from eval_folds
# ---------------------------------------------------------------------------

def cross_model_comparison(
    eval_folds_path: Path,
) -> pd.DataFrame:
    """Aggregate residual-relevant metrics per model from existing eval_folds.

    Uses MAE, bias, CRPS, pinball_cf, and coverage metrics as proxies
    for residual structure since we have these pre-computed.
    """
    ef = pd.read_parquet(eval_folds_path)

    agg = ef.groupby("model_name").agg(
        mean_mae=("mae", "mean"),
        mean_bias=("bias", "mean"),
        std_bias=("bias", "std"),
        mean_crps=("crps", "mean"),
        mean_pinball=("pinball_loss", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_coverage_80=("coverage_80", "mean"),
        mean_coverage_90=("coverage_90", "mean"),
        n_folds=("fold_idx", "count"),
    ).reset_index()

    agg["bias_ratio"] = agg["mean_bias"] / agg["mean_mae"].clip(lower=0.01)
    agg["rmse_mae_ratio"] = agg["mean_rmse"] / agg["mean_mae"].clip(lower=0.01)

    return agg.sort_values("mean_mae")


# ---------------------------------------------------------------------------
# Build CatBoost residuals DataFrame
# ---------------------------------------------------------------------------

def build_catboost_residuals(
    forecasts_path: Path,
    actuals_path: Path,
) -> pd.DataFrame:
    """Merge CatBoost competition forecasts with actuals to produce residuals."""
    fc = pd.read_parquet(forecasts_path)
    act = pd.read_parquet(actuals_path)

    merged = fc.merge(act, on=["Store", "Product", "week"])
    merged["residual"] = merged["actual_demand"] - merged["h1"]
    merged["demand"] = merged["actual_demand"]
    return merged[["Store", "Product", "week", "demand", "h1", "residual"]]


# ---------------------------------------------------------------------------
# Generate integration recommendation
# ---------------------------------------------------------------------------

def generate_recommendation(
    acf_df: pd.DataFrame,
    hetero_df: pd.DataFrame,
    tail_df: pd.DataFrame,
    taylor_df: pd.DataFrame,
    cross_model_df: Optional[pd.DataFrame] = None,
) -> str:
    """Produce a markdown text summarising findings and the gating decision."""
    lines = ["# Residual Diagnostics: Integration Recommendation\n"]

    # ACF summary
    n_sig_acf = acf_df["any_significant_acf"].sum()
    n_total = len(acf_df)
    pct_acf = 100 * n_sig_acf / max(n_total, 1)
    lines.append("## A. Autocorrelation of Residuals\n")
    lines.append(f"- {n_sig_acf}/{n_total} SKUs ({pct_acf:.1f}%) show significant "
                 f"autocorrelation (Ljung-Box, p<0.05 at any tested lag)")
    if pct_acf > 30:
        lines.append("- **Finding**: Substantial residual autocorrelation. "
                      "Bootstrap on i.i.d. residuals (SLURP approach) would "
                      "understate uncertainty for these SKUs.")
    else:
        lines.append("- **Finding**: Residual autocorrelation is limited. "
                      "Bootstrap approaches are likely adequate.")
    lines.append("")

    # Heteroskedasticity summary
    n_hetero = hetero_df["heteroskedastic_resid"].sum()
    pct_hetero = 100 * n_hetero / max(len(hetero_df), 1)
    lines.append("## B. Heteroskedasticity\n")
    lines.append(f"- {n_hetero}/{len(hetero_df)} SKUs ({pct_hetero:.1f}%) show "
                 f"significant heteroskedasticity (Breusch-Pagan, p<0.05)")

    if not taylor_df.empty and "taylor_alpha" in taylor_df.columns:
        alpha_vals = taylor_df["taylor_alpha"].dropna()
        if len(alpha_vals) > 0:
            lines.append(f"- Taylor's law on residuals: alpha = "
                         f"{alpha_vals.iloc[0]:.3f}")
    if pct_hetero > 30:
        lines.append("- **Finding**: Residual variance scales with demand level. "
                      "Constant-variance bootstrap will be miscalibrated for "
                      "high/low demand SKUs.")
    else:
        lines.append("- **Finding**: Residual variance is relatively homogeneous.")
    lines.append("")

    # Tail summary
    n_normal = tail_df["normal_residuals"].sum()
    pct_normal = 100 * n_normal / max(len(tail_df), 1)
    mean_kurt = tail_df["kurtosis"].mean()
    mean_3sig = tail_df["frac_beyond_3sigma"].mean()
    lines.append("## C. Tail Behaviour\n")
    lines.append(f"- {n_normal}/{len(tail_df)} SKUs ({pct_normal:.1f}%) have "
                 f"normally-distributed residuals (Jarque-Bera, p>=0.05)")
    lines.append(f"- Mean excess kurtosis: {mean_kurt:.2f} "
                 f"(0 = Normal, positive = heavy tails)")
    lines.append(f"- Mean fraction beyond 3-sigma: {mean_3sig:.3f} "
                 f"(Normal expectation: 0.003)")
    if mean_kurt > 1.0 or mean_3sig > 0.01:
        lines.append("- **Finding**: Heavy-tailed residuals. Normal-based "
                      "prediction intervals will be too narrow.")
    else:
        lines.append("- **Finding**: Tail behaviour is near-Normal.")
    lines.append("")

    # Cross-model
    if cross_model_df is not None and not cross_model_df.empty:
        lines.append("## D. Cross-Model Comparison\n")
        lines.append("| Model | MAE | Bias | CRPS | RMSE/MAE ratio |")
        lines.append("|---|---|---|---|---|")
        for _, r in cross_model_df.iterrows():
            lines.append(
                f"| {r['model_name']} | {r['mean_mae']:.3f} | "
                f"{r['mean_bias']:.3f} | {r['mean_crps']:.3f} | "
                f"{r['rmse_mae_ratio']:.2f} |"
            )
        lines.append("")

    # Gating decision
    lines.append("## Integration Decision\n")
    problems = []
    if pct_acf > 30:
        problems.append("autocorrelated residuals")
    if pct_hetero > 30:
        problems.append("heteroskedastic residuals")
    if mean_kurt > 1.0 or mean_3sig > 0.01:
        problems.append("heavy-tailed residuals")

    if len(problems) >= 2:
        lines.append(
            f"**Recommendation: Native distributional outputs needed.** "
            f"Residuals exhibit {', '.join(problems)}, which violate "
            f"the i.i.d. Normal assumptions underlying bootstrap-based "
            f"uncertainty (SLURP). A distributional output head (NegBin, "
            f"quantile, or mixture density) trained under CRPS would "
            f"directly model these structures."
        )
    elif len(problems) == 1:
        lines.append(
            f"**Recommendation: Cautious — consider hybrid approach.** "
            f"Residuals exhibit {problems[0]}. SLURP bootstrap may work "
            f"for most SKUs but consider native distributional outputs "
            f"for the affected cohort."
        )
    else:
        lines.append(
            "**Recommendation: SLURP bootstrap wrapping is viable.** "
            "Residuals are approximately i.i.d. with near-Normal tails. "
            "Wrapping CatBoost in SLURP's bootstrap should produce "
            "well-calibrated densities."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Run all diagnostics
# ---------------------------------------------------------------------------

def run_all(
    forecasts_path: Path,
    actuals_path: Path,
    output_dir: Path,
    eval_folds_path: Optional[Path] = None,
    n_jobs: int = 12,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all residual diagnostics and write outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building CatBoost residuals ...")
    resid = build_catboost_residuals(forecasts_path, actuals_path)
    resid.to_parquet(output_dir / "catboost_residuals.parquet", index=False)
    print(f"  {len(resid)} residual rows, {resid[['Store','Product']].drop_duplicates().shape[0]} SKUs")

    print("A. Residual autocorrelation ...")
    acf_df = residual_acf(resid, n_jobs=n_jobs)
    acf_df.to_parquet(output_dir / "acf_diagnostics.parquet", index=False)

    print("B. Heteroskedasticity ...")
    hetero_df = residual_heteroskedasticity(resid, n_jobs=n_jobs)
    hetero_df.to_parquet(output_dir / "hetero_diagnostics.parquet", index=False)

    taylor_df = taylors_law_on_residuals(hetero_df)
    if not taylor_df.empty:
        taylor_df.to_csv(output_dir / "taylor_law_residuals.csv", index=False)

    print("C. Tail behaviour ...")
    tail_df = residual_tails(resid, n_jobs=n_jobs)
    tail_df.to_parquet(output_dir / "tail_diagnostics.parquet", index=False)

    # D. Cross-model comparison
    cross_model_df = None
    if eval_folds_path is not None and eval_folds_path.exists():
        print("D. Cross-model comparison ...")
        cross_model_df = cross_model_comparison(eval_folds_path)
        cross_model_df.to_csv(output_dir / "cross_model_comparison.csv", index=False)

    # Summary
    summary_rows = []
    for label, df_diag in [("acf", acf_df), ("hetero", hetero_df), ("tails", tail_df)]:
        summary_rows.append({
            "diagnostic": label,
            "n_skus": len(df_diag),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "residual_summary.csv", index=False)

    # Recommendation
    rec_text = generate_recommendation(acf_df, hetero_df, tail_df, taylor_df,
                                       cross_model_df)
    with open(output_dir / "integration_recommendation.md", "w") as f:
        f.write(rec_text)
    print(f"\nRecommendation written to {output_dir / 'integration_recommendation.md'}")

    return acf_df, hetero_df, tail_df, taylor_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Residual diagnostics for CatBoost forecasts")
    parser.add_argument("--forecasts", type=Path, required=True,
                        help="Parquet with competition forecasts (Store, Product, week, h1, h2, h3)")
    parser.add_argument("--actuals", type=Path, required=True,
                        help="Parquet with actual demand (Store, Product, week, actual_demand)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("reports/residual_diagnostics"))
    parser.add_argument("--eval-folds", type=Path, default=None,
                        help="Optional eval_folds parquet for cross-model comparison")
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()

    run_all(args.forecasts, args.actuals, args.output_dir,
            eval_folds_path=args.eval_folds, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
