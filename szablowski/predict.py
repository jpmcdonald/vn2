"""
Forecast generation for the Szabłowski CatBoost pipeline.

Loads trained h=1,2,3 models and produces point forecasts, post-processed
(rounded to integer, clipped >= 0) and saved as parquet for downstream
policy and comparison harness consumption.

Includes an optional dormancy filter that zeroes forecasts for SKUs whose
recent in-stock demand is entirely zero (likely discontinued / seasonal off).
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import catboost as cb
except ImportError:
    cb = None

from szablowski.features import build_features, _week_col, fold_split
from szablowski.train import build_targets


def load_models(model_dir: Path) -> Dict[int, object]:
    """Load saved CatBoost models for h=1,2,3."""
    if cb is None:
        raise ImportError("catboost is required")
    models = {}
    for h in [1, 2, 3]:
        path = model_dir / f"catboost_h{h}.cbm"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        models[h] = cb.CatBoostRegressor().load_model(str(path))
    return models


def load_feature_cols(model_dir: Path) -> List[str]:
    with open(model_dir / "feature_cols.pkl", "rb") as f:
        return pickle.load(f)


def predict_all(
    df: pd.DataFrame,
    models: Dict[int, object],
    feature_cols: List[str],
) -> pd.DataFrame:
    """Generate point forecasts for all rows, all horizons.

    Returns a DataFrame with Store, Product, week_date, h1, h2, h3, scale_factor.
    Forecasts are in *original* (unscaled) units, rounded and clipped.
    """
    wc = _week_col(df)

    results = []
    for h, model in models.items():
        X = df[feature_cols]
        preds_scaled = model.predict(X)
        preds_orig = preds_scaled * df["scale_factor"].values
        preds_orig = np.clip(np.round(preds_orig), 0, None).astype(int)
        results.append(preds_orig)

    out = df[["Store", "Product", wc, "scale_factor"]].copy()
    out["h1"] = results[0]
    out["h2"] = results[1]
    out["h3"] = results[2]
    return out


# ---------------------------------------------------------------------------
# Dormancy filter
# ---------------------------------------------------------------------------

def apply_dormancy_filter(
    forecasts_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    lookback: int = 10,
) -> pd.DataFrame:
    """Zero out forecasts for SKUs that have been dormant.

    A SKU is dormant if its last ``lookback`` in-stock weeks all had zero
    demand.  For each forecast row the lookback window ends at the week
    *before* the forecast date.

    Adds a boolean ``dormancy_filtered`` column for traceability.
    """
    wc = _week_col(forecasts_df)
    demand_wc = _week_col(demand_df)

    sales_col = "sales" if "sales" in demand_df.columns else "demand"
    in_stock_col = "in_stock" if "in_stock" in demand_df.columns else None

    dormant_skus: set = set()

    for (store, product), grp in demand_df.groupby(["Store", "Product"]):
        grp = grp.sort_values(demand_wc)

        if in_stock_col is not None:
            in_stock_rows = grp[grp[in_stock_col].astype(bool)]
        else:
            in_stock_rows = grp

        if len(in_stock_rows) < lookback:
            continue

        tail = in_stock_rows[sales_col].iloc[-lookback:]
        if (tail == 0).all():
            dormant_skus.add((store, product))

    out = forecasts_df.copy()
    mask = out.apply(
        lambda r: (r["Store"], r["Product"]) in dormant_skus, axis=1
    )
    out["dormancy_filtered"] = mask
    out.loc[mask, ["h1", "h2", "h3"]] = 0

    n_filtered = mask.sum()
    n_skus = len(dormant_skus)
    print(f"  Dormancy filter: {n_skus} SKUs dormant, "
          f"{n_filtered} forecast rows zeroed (lookback={lookback})")
    return out


def generate_forecasts(
    demand_path: Path,
    model_dir: Path,
    output_path: Path,
    dormancy_lookback: int = 0,
) -> pd.DataFrame:
    """End-to-end: load data, build features, predict, save."""
    print(f"Loading data from {demand_path} ...")
    df = pd.read_parquet(demand_path)
    if "sales" not in df.columns and "demand" in df.columns:
        df["sales"] = df["demand"]

    print("Building features ...")
    df, _ = build_features(df)

    print("Loading models ...")
    models = load_models(model_dir)
    feature_cols = load_feature_cols(model_dir)

    print("Generating forecasts ...")
    forecasts = predict_all(df, models, feature_cols)

    if dormancy_lookback > 0:
        raw = pd.read_parquet(demand_path)
        if "sales" not in raw.columns and "demand" in raw.columns:
            raw["sales"] = raw["demand"]
        forecasts = apply_dormancy_filter(forecasts, raw, lookback=dormancy_lookback)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(output_path, index=False)
    print(f"Saved {len(forecasts)} forecast rows to {output_path}")
    return forecasts


def generate_fold_forecasts(
    demand_path: Path,
    model_dir: Path,
    output_path: Path,
    n_folds: int = 8,
    holdout_weeks: int = 18,
) -> pd.DataFrame:
    """Generate per-fold forecasts for backtest evaluation.

    For each fold, builds features on the train portion and predicts the
    3-week test window.  Output has columns: Store, Product, fold_idx, h1, h2, h3.
    """
    print(f"Loading data from {demand_path} ...")
    df = pd.read_parquet(demand_path)
    if "sales" not in df.columns and "demand" in df.columns:
        df["sales"] = df["demand"]

    print("Building features ...")
    df, _ = build_features(df)

    models = load_models(model_dir)
    feature_cols = load_feature_cols(model_dir)

    all_preds = []
    for fold_idx in range(n_folds):
        print(f"  Fold {fold_idx} ...")
        train_df, test_df = fold_split(df, fold_idx=fold_idx, holdout_weeks=holdout_weeks)
        if test_df.empty:
            continue

        preds = predict_all(test_df, models, feature_cols)
        preds["fold_idx"] = fold_idx
        all_preds.append(preds)

    result = pd.concat(all_preds, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"Saved {len(result)} fold-forecast rows to {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate Szabłowski CatBoost forecasts")
    parser.add_argument("--demand-path", type=Path, default=Path("data/processed/demand_long.parquet"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/szablowski"))
    parser.add_argument("--output", type=Path, default=Path("models/szablowski/forecasts.parquet"))
    parser.add_argument("--fold-output", type=Path, default=Path("models/szablowski/fold_forecasts.parquet"))
    parser.add_argument("--n-folds", type=int, default=8)
    parser.add_argument("--holdout-weeks", type=int, default=8)
    parser.add_argument("--folds-only", action="store_true", help="Only generate fold forecasts")
    parser.add_argument("--dormancy-lookback", type=int, default=10,
                        help="Zero forecasts for SKUs with this many consecutive "
                             "zero-demand in-stock weeks (0 to disable)")
    args = parser.parse_args()

    if not args.folds_only:
        generate_forecasts(args.demand_path, args.model_dir, args.output,
                           dormancy_lookback=args.dormancy_lookback)

    generate_fold_forecasts(
        args.demand_path, args.model_dir, args.fold_output,
        n_folds=args.n_folds, holdout_weeks=args.holdout_weeks,
    )


if __name__ == "__main__":
    main()
