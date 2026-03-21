"""
EDA extensions for backtesting stratification.

Adds per-SKU diagnostics that the existing EDA notebook did not compute:
  - Variance-mean power law (Taylor's law)
  - Seasonal decomposition strength (STL-based)
  - Structural break detection (CUSUM)
  - Ljung-Box autocorrelation test
  - Count distribution goodness-of-fit (NegBin vs Poisson)

Results are designed to be merged into the existing
data/processed/summary_statistics.parquet and stationarity_tests.parquet.
"""

import argparse
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats as sp_stats

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
except ImportError:
    acorr_ljungbox = None

try:
    from statsmodels.tsa.seasonal import STL
except ImportError:
    STL = None


# ---------------------------------------------------------------------------
# 1. Variance-mean power law (Taylor's law)
# ---------------------------------------------------------------------------

def _taylor_one_sku(store, product, y, window, min_windows):
    if len(y) < window * min_windows:
        return {"Store": store, "Product": product,
                "taylor_alpha": np.nan, "taylor_r2": np.nan}

    n_windows = len(y) // window
    means, variances = [], []
    for i in range(n_windows):
        chunk = y[i * window : (i + 1) * window]
        m = np.mean(chunk)
        v = np.var(chunk, ddof=1) if len(chunk) > 1 else 0
        if m > 0 and v > 0:
            means.append(np.log(m))
            variances.append(np.log(v))

    if len(means) < min_windows:
        return {"Store": store, "Product": product,
                "taylor_alpha": np.nan, "taylor_r2": np.nan}

    slope, intercept, r, p, se = sp_stats.linregress(means, variances)
    return {
        "Store": store, "Product": product,
        "taylor_alpha": slope / 2.0,
        "taylor_r2": r ** 2,
    }


def taylor_alpha_per_sku(
    df: pd.DataFrame,
    window: int = 13,
    min_windows: int = 4,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Fit log(var) ~ alpha * log(mean) per SKU using rolling windows.

    Szabłowski's σ = φ√D implies alpha = 0.5 (Poisson-like scaling).
    Returns DataFrame with Store, Product, taylor_alpha, taylor_r2.
    """
    sales_col = "sales" if "sales" in df.columns else "demand"
    groups = [(s, p, g[sales_col].dropna().values) for (s, p), g in df.groupby(["Store", "Product"])]

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_taylor_one_sku)(s, p, y, window, min_windows) for s, p, y in groups
    )
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 2. Seasonal decomposition strength (STL)
# ---------------------------------------------------------------------------

def _stl_one_sku(store, product, y, period, min_obs):
    if len(y) < min_obs:
        return {"Store": store, "Product": product, "seasonal_strength_stl": np.nan}
    try:
        stl_result = STL(y, period=period, robust=True).fit()
        seasonal = stl_result.seasonal
        resid = stl_result.resid
        var_sr = np.var(seasonal + resid)
        var_r = np.var(resid)
        strength = max(0, 1.0 - var_r / var_sr) if var_sr > 0 else 0.0
    except Exception:
        strength = np.nan
    return {"Store": store, "Product": product, "seasonal_strength_stl": strength}


def seasonal_strength_stl(
    df: pd.DataFrame,
    period: int = 52,
    min_obs: int = 104,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """STL-based seasonal strength: 1 - Var(remainder) / Var(remainder + seasonal).

    Returns DataFrame with Store, Product, seasonal_strength_stl.
    """
    if STL is None:
        warnings.warn("statsmodels.tsa.seasonal.STL not available; skipping STL")
        return pd.DataFrame(columns=["Store", "Product", "seasonal_strength_stl"])

    sales_col = "sales" if "sales" in df.columns else "demand"
    groups = [(s, p, g[sales_col].dropna().values.astype(float))
              for (s, p), g in df.groupby(["Store", "Product"])]

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_stl_one_sku)(s, p, y, period, min_obs) for s, p, y in groups
    )
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 3. Structural break detection (CUSUM)
# ---------------------------------------------------------------------------

def _cusum_one_sku(store, product, y, significance):
    n = len(y)
    if n < 20:
        return {"Store": store, "Product": product,
                "has_structural_break": np.nan, "cusum_max_stat": np.nan}

    mean_y = np.mean(y)
    std_y = np.std(y, ddof=1)
    if std_y == 0:
        return {"Store": store, "Product": product,
                "has_structural_break": False, "cusum_max_stat": 0.0}

    cusum = np.cumsum(y - mean_y) / (std_y * np.sqrt(n))
    max_stat = np.max(np.abs(cusum))

    crit = {0.01: 1.628, 0.05: 1.358, 0.10: 1.224}.get(significance, 1.358)
    has_break = bool(max_stat > crit)

    return {
        "Store": store, "Product": product,
        "has_structural_break": has_break,
        "cusum_max_stat": float(max_stat),
    }


def cusum_break_test(
    df: pd.DataFrame,
    significance: float = 0.05,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """CUSUM test for structural breaks in the mean.

    Uses the Brown-Durbin-Evans approach: cumulative sum of recursive residuals.
    A break is flagged if the CUSUM path crosses the ±(a + b*t) boundary.

    Returns DataFrame with Store, Product, has_structural_break, cusum_max_stat.
    """
    sales_col = "sales" if "sales" in df.columns else "demand"
    groups = [(s, p, g[sales_col].dropna().values.astype(float))
              for (s, p), g in df.groupby(["Store", "Product"])]

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_cusum_one_sku)(s, p, y, significance) for s, p, y in groups
    )
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 4. Ljung-Box autocorrelation test
# ---------------------------------------------------------------------------

def _ljung_box_one_sku(store, product, y, lags_short, lags_long):
    n = len(y)
    row = {"Store": store, "Product": product,
           "ljung_box_pval_12": np.nan, "ljung_box_pval_52": np.nan}

    if n > lags_short + 5:
        try:
            lb_result = acorr_ljungbox(y, lags=[lags_short], return_df=True)
            row["ljung_box_pval_12"] = float(lb_result["lb_pvalue"].iloc[0])
        except Exception:
            pass

    if n > lags_long + 5:
        try:
            lb_result = acorr_ljungbox(y, lags=[lags_long], return_df=True)
            row["ljung_box_pval_52"] = float(lb_result["lb_pvalue"].iloc[0])
        except Exception:
            pass

    return row


def ljung_box_test(
    df: pd.DataFrame,
    lags_short: int = 12,
    lags_long: int = 52,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Ljung-Box test at two lag horizons.

    Returns DataFrame with Store, Product, ljung_box_pval_12, ljung_box_pval_52.
    """
    if acorr_ljungbox is None:
        warnings.warn("statsmodels acorr_ljungbox not available; skipping Ljung-Box")
        return pd.DataFrame(columns=["Store", "Product", "ljung_box_pval_12", "ljung_box_pval_52"])

    sales_col = "sales" if "sales" in df.columns else "demand"
    groups = [(s, p, g[sales_col].dropna().values.astype(float))
              for (s, p), g in df.groupby(["Store", "Product"])]

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_ljung_box_one_sku)(s, p, y, lags_short, lags_long) for s, p, y in groups
    )
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 5. Count distribution GoF (NegBin vs Poisson)
# ---------------------------------------------------------------------------

def _count_gof_one_sku(store, product, y, min_nonzero):
    y_int = np.round(y).astype(int)
    y_int = y_int[y_int >= 0]

    row = {"Store": store, "Product": product,
           "negbin_preferred": np.nan, "zero_inflated": np.nan,
           "dispersion_ratio": np.nan}

    n_nonzero = np.sum(y_int > 0)
    if n_nonzero < min_nonzero:
        return row

    mean_y = np.mean(y_int)
    var_y = np.var(y_int, ddof=1)
    if mean_y <= 0:
        return row

    row["dispersion_ratio"] = float(var_y / mean_y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ll_poisson = float(np.sum(sp_stats.poisson.logpmf(y_int, mu=mean_y)))

    if var_y > mean_y:
        r = mean_y ** 2 / (var_y - mean_y)
        p = mean_y / var_y
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ll_negbin = float(np.sum(sp_stats.nbinom.logpmf(y_int, n=r, p=1 - p)))

        lr_stat = 2 * (ll_negbin - ll_poisson)
        lr_pval = 1.0 - sp_stats.chi2.cdf(max(lr_stat, 0), df=1)
        row["negbin_preferred"] = bool(lr_pval < 0.05)
    else:
        row["negbin_preferred"] = False

    observed_zero_frac = np.mean(y_int == 0)
    expected_zero_frac = np.exp(-mean_y) if mean_y < 700 else 0.0
    if len(y_int) > 0 and expected_zero_frac > 0:
        n_zeros = int(np.sum(y_int == 0))
        binom_pval = (
            sp_stats.binom_test(n_zeros, len(y_int), expected_zero_frac, alternative="greater")
            if hasattr(sp_stats, "binom_test")
            else sp_stats.binomtest(n_zeros, len(y_int), expected_zero_frac, alternative="greater").pvalue
        )
        row["zero_inflated"] = bool(binom_pval < 0.05)
    else:
        row["zero_inflated"] = bool(observed_zero_frac > 0.5)

    return row


def count_distribution_gof(
    df: pd.DataFrame,
    min_nonzero: int = 20,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Test NegBin vs Poisson fit and detect zero-inflation per SKU.

    Uses a likelihood-ratio test: NegBin is preferred if it significantly
    improves over Poisson (LR test p < 0.05).  Zero-inflation is flagged if
    observed zero fraction significantly exceeds the Poisson-predicted fraction.

    Returns DataFrame with Store, Product, negbin_preferred, zero_inflated,
    dispersion_ratio (var/mean).
    """
    sales_col = "sales" if "sales" in df.columns else "demand"
    groups = [(s, p, g[sales_col].dropna().values.astype(float))
              for (s, p), g in df.groupby(["Store", "Product"])]

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_count_gof_one_sku)(s, p, y, min_nonzero) for s, p, y in groups
    )
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_all_extensions(
    demand_path: Path,
    output_dir: Path,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run all EDA extensions and merge into existing artifact structure.

    Returns (summary_extensions, stationarity_extensions).
    """
    print(f"Loading data from {demand_path} ...")
    df = pd.read_parquet(demand_path)
    if "sales" not in df.columns and "demand" in df.columns:
        df["sales"] = df["demand"]

    print("Computing Taylor's law (variance-mean power law) ...")
    taylor = taylor_alpha_per_sku(df, n_jobs=n_jobs)

    print("Computing STL seasonal strength ...")
    seasonal = seasonal_strength_stl(df, n_jobs=n_jobs)

    print("Computing CUSUM structural break test ...")
    cusum = cusum_break_test(df, n_jobs=n_jobs)

    print("Computing Ljung-Box autocorrelation test ...")
    ljungbox = ljung_box_test(df, n_jobs=n_jobs)

    print("Computing count distribution GoF (NegBin vs Poisson) ...")
    countgof = count_distribution_gof(df, n_jobs=n_jobs)

    # Merge into two groups matching existing parquet structure
    summary_ext = taylor.merge(seasonal, on=["Store", "Product"], how="outer")
    summary_ext = summary_ext.merge(countgof, on=["Store", "Product"], how="outer")

    stationarity_ext = cusum.merge(ljungbox, on=["Store", "Product"], how="outer")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary_statistics_ext.parquet"
    summary_ext.to_parquet(summary_path, index=False)
    print(f"  Saved: {summary_path}  ({len(summary_ext)} SKUs)")

    stationarity_path = output_dir / "stationarity_tests_ext.parquet"
    stationarity_ext.to_parquet(stationarity_path, index=False)
    print(f"  Saved: {stationarity_path}  ({len(stationarity_ext)} SKUs)")

    # Also try to merge with existing parquet files if they exist
    _try_merge_existing(output_dir.parent / "processed", summary_ext, stationarity_ext)

    return summary_ext, stationarity_ext


def _try_merge_existing(
    processed_dir: Path,
    summary_ext: pd.DataFrame,
    stationarity_ext: pd.DataFrame,
):
    """Attempt to merge new columns into existing EDA parquets."""
    ss_path = processed_dir / "summary_statistics.parquet"
    if ss_path.exists():
        try:
            existing = pd.read_parquet(ss_path)
            # Drop any columns that would conflict
            new_cols = [c for c in summary_ext.columns if c not in ("Store", "Product")]
            for col in new_cols:
                if col in existing.columns:
                    existing = existing.drop(columns=[col])
            merged = existing.merge(summary_ext, on=["Store", "Product"], how="left")
            merged.to_parquet(ss_path, index=False)
            print(f"  Merged new columns into {ss_path}")
        except Exception as e:
            print(f"  Warning: could not merge into {ss_path}: {e}")

    st_path = processed_dir / "stationarity_tests.parquet"
    if st_path.exists():
        try:
            existing = pd.read_parquet(st_path)
            new_cols = [c for c in stationarity_ext.columns if c not in ("Store", "Product")]
            for col in new_cols:
                if col in existing.columns:
                    existing = existing.drop(columns=[col])
            merged = existing.merge(stationarity_ext, on=["Store", "Product"], how="left")
            merged.to_parquet(st_path, index=False)
            print(f"  Merged new columns into {st_path}")
        except Exception as e:
            print(f"  Warning: could not merge into {st_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run EDA extensions for backtesting stratification")
    parser.add_argument("--demand-path", type=Path, default=Path("data/processed/demand_long.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()

    run_all_extensions(args.demand_path, args.output_dir, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
