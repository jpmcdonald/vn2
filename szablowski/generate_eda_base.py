"""
Generate base EDA parquets (summary_statistics, stationarity_tests) from demand_long.parquet.

Replicates the core EDA notebook computations as a standalone script for
reproducible pipeline execution. Parallelized with joblib across 12 cores.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats as sp_stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_arch


def _summary_for_sku(store: int, product: int, y: np.ndarray) -> dict:
    n = len(y)
    nonzero = y[y > 0]
    row = {
        "Store": store,
        "Product": product,
        "n_obs": n,
        "mean": np.mean(y),
        "std": np.std(y, ddof=1) if n > 1 else 0.0,
        "cv": np.std(y, ddof=1) / np.mean(y) if np.mean(y) > 0 and n > 1 else np.nan,
        "skewness": float(sp_stats.skew(y)) if n > 2 else np.nan,
        "kurtosis": float(sp_stats.kurtosis(y)) if n > 3 else np.nan,
        "pct_zeros": float(np.mean(y == 0)),
        "median": float(np.median(y)),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "iqr": float(np.percentile(y, 75) - np.percentile(y, 25)) if n > 0 else 0.0,
    }

    # Average Demand Interval (ADI) for intermittency classification
    if len(nonzero) > 1:
        nz_indices = np.where(y > 0)[0]
        intervals = np.diff(nz_indices)
        row["ADI"] = float(np.mean(intervals)) if len(intervals) > 0 else np.nan
    else:
        row["ADI"] = np.nan

    # Stockout proxy (already in data as in_stock, but also compute pct_stockout)
    row["pct_stockout"] = np.nan  # filled from in_stock column if available
    return row


def _stationarity_for_sku(store: int, product: int, y: np.ndarray) -> dict:
    row = {
        "Store": store,
        "Product": product,
        "adf_stat": np.nan,
        "adf_pval": np.nan,
        "adf_stationary": np.nan,
        "kpss_stat": np.nan,
        "kpss_pval": np.nan,
        "kpss_stationary": np.nan,
        "arch_stat": np.nan,
        "arch_pval": np.nan,
        "heteroskedastic": np.nan,
    }
    if len(y) < 20 or np.std(y) == 0:
        return row

    try:
        adf_result = adfuller(y, maxlag=min(13, len(y) // 3), autolag="AIC")
        row["adf_stat"] = float(adf_result[0])
        row["adf_pval"] = float(adf_result[1])
        row["adf_stationary"] = bool(adf_result[1] < 0.05)
    except Exception:
        pass

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_result = kpss(y, regression="c", nlags="auto")
        row["kpss_stat"] = float(kpss_result[0])
        row["kpss_pval"] = float(kpss_result[1])
        row["kpss_stationary"] = bool(kpss_result[1] > 0.05)
    except Exception:
        pass

    try:
        arch_result = het_arch(y, nlags=min(12, len(y) // 4))
        row["arch_stat"] = float(arch_result[0])
        row["arch_pval"] = float(arch_result[1])
        row["heteroskedastic"] = bool(arch_result[1] < 0.05)
    except Exception:
        pass

    return row


def generate_eda_base(
    demand_path: Path,
    output_dir: Path,
    n_jobs: int = 12,
) -> tuple:
    print(f"Loading data from {demand_path} ...")
    df = pd.read_parquet(demand_path)
    sales_col = "sales" if "sales" in df.columns else "demand"

    groups = []
    for (store, product), grp in df.groupby(["Store", "Product"]):
        groups.append((store, product, grp[sales_col].values, grp.get("in_stock")))

    print(f"Computing summary statistics for {len(groups)} SKUs ({n_jobs} jobs) ...")
    summary_rows = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_summary_for_sku)(store, product, y) for store, product, y, _ in groups
    )

    # Fill pct_stockout from in_stock column
    for row, (_, _, _, in_stock) in zip(summary_rows, groups):
        if in_stock is not None:
            row["pct_stockout"] = float(1.0 - in_stock.astype(float).mean())

    summary_df = pd.DataFrame(summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary_statistics.parquet"
    summary_df.to_parquet(summary_path, index=False)
    print(f"  Saved: {summary_path}  ({len(summary_df)} SKUs)")

    print(f"Computing stationarity tests for {len(groups)} SKUs ({n_jobs} jobs) ...")
    stationarity_rows = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_stationarity_for_sku)(store, product, y) for store, product, y, _ in groups
    )

    stationarity_df = pd.DataFrame(stationarity_rows)
    stationarity_path = output_dir / "stationarity_tests.parquet"
    stationarity_df.to_parquet(stationarity_path, index=False)
    print(f"  Saved: {stationarity_path}  ({len(stationarity_df)} SKUs)")

    return summary_df, stationarity_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate base EDA parquets from demand_long.parquet"
    )
    parser.add_argument(
        "--demand-path", type=Path, default=Path("data/processed/demand_long.parquet")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()

    generate_eda_base(args.demand_path, args.output_dir, args.n_jobs)


if __name__ == "__main__":
    main()
