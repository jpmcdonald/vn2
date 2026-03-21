"""
CatBoost training pipeline reproducing Szabłowski (arXiv:2601.18919v1).

Three independent CatBoost regressors for h=1, h=2, h=3 with RMSE loss on
scaled targets.  Optuna HPO (100 trials, TPE), early stopping at 500 rounds.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import catboost as cb
except ImportError:
    cb = None

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

from szablowski.features import (
    build_features,
    train_val_test_split,
    _week_col,
)


# ---------------------------------------------------------------------------
# Target builder: direct multi-horizon
# ---------------------------------------------------------------------------

def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns y_h1, y_h2, y_h3 = scaled y_eff shifted by 1, 2, 3 periods."""
    wc = _week_col(df)
    out = df.copy()
    for h in [1, 2, 3]:
        out[f"y_h{h}"] = np.nan

    for (store, product), grp in out.groupby(["Store", "Product"]):
        idx = grp.index
        y_scaled = grp["y_scaled"].values
        for h in [1, 2, 3]:
            shifted = np.full(len(y_scaled), np.nan)
            if h < len(y_scaled):
                shifted[:-h] = y_scaled[h:]
            out.loc[idx, f"y_h{h}"] = shifted

    return out


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _make_objective(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val_orig: np.ndarray,
    scale_val: np.ndarray,
    cat_features: List[int],
):
    """Return an Optuna objective that trains CatBoost and returns MAE in original units."""

    def objective(trial):
        params = {
            "iterations": 3000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "task_type": "CPU",
            "verbose": 0,
            "random_seed": 42,
            "early_stopping_rounds": 500,
        }

        train_pool = cb.Pool(X_train, y_train, weight=w_train, cat_features=cat_features)
        val_pool = cb.Pool(X_val, label=y_val_orig / scale_val, cat_features=cat_features)

        model = cb.CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, verbose=0)

        # Evaluate MAE in original units
        preds_scaled = model.predict(X_val)
        preds_orig = preds_scaled * scale_val
        mae = np.mean(np.abs(preds_orig - y_val_orig))
        return mae

    return objective


# ---------------------------------------------------------------------------
# Train single horizon
# ---------------------------------------------------------------------------

def train_horizon(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    horizon: int,
    feature_cols: List[str],
    n_trials: int = 100,
    seed: int = 42,
) -> Tuple[object, dict]:
    """Train a CatBoost model for a single horizon with Optuna HPO.

    Returns (model, info_dict).
    """
    if cb is None:
        raise ImportError("catboost is required: pip install catboost")
    if optuna is None:
        raise ImportError("optuna is required: pip install optuna")

    target_col = f"y_h{horizon}"

    # Drop rows without target
    tr = train_df.dropna(subset=[target_col])
    vl = val_df.dropna(subset=[target_col])

    if len(tr) == 0 or len(vl) == 0:
        raise ValueError(f"No valid samples for horizon {horizon}")

    X_train = tr[feature_cols]
    y_train = tr[target_col].values
    w_train = tr["sample_weight"].values

    X_val = vl[feature_cols]
    y_val_scaled = vl[target_col].values
    scale_val = vl["scale_factor"].values
    y_val_orig = y_val_scaled * scale_val

    cat_features = [i for i, c in enumerate(feature_cols) if str(X_train[c].dtype) == "category"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        _make_objective(X_train, y_train, w_train, X_val, y_val_orig, scale_val, cat_features),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    best.update({
        "iterations": 3000,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "task_type": "CPU",
        "verbose": 100,
        "random_seed": seed,
        "early_stopping_rounds": 500,
    })

    train_pool = cb.Pool(X_train, y_train, weight=w_train, cat_features=cat_features)
    val_pool = cb.Pool(X_val, label=y_val_scaled, cat_features=cat_features)

    final_model = cb.CatBoostRegressor(**best)
    final_model.fit(train_pool, eval_set=val_pool, verbose=100)

    # Final MAE
    preds_orig = final_model.predict(X_val) * scale_val
    mae = float(np.mean(np.abs(preds_orig - y_val_orig)))

    info = {
        "horizon": horizon,
        "best_params": study.best_params,
        "val_mae_orig": mae,
        "n_trials": n_trials,
        "train_rows": len(tr),
        "val_rows": len(vl),
        "best_iteration": final_model.get_best_iteration(),
    }

    return final_model, info


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def train_all_horizons(
    demand_path: Path,
    output_dir: Path,
    test_weeks: int = 18,
    val_frac: float = 0.10,
    n_trials: int = 100,
    seed: int = 42,
) -> Dict[int, object]:
    """End-to-end: load data, build features, train h=1,2,3 models, save."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {demand_path} ...")
    df = pd.read_parquet(demand_path)
    if "sales" not in df.columns and "demand" in df.columns:
        df["sales"] = df["demand"]

    print("Building features ...")
    df, feature_cols = build_features(df)

    print("Building multi-horizon targets ...")
    df = build_targets(df)

    print("Splitting train / val / test ...")
    train_df, val_df, test_df = train_val_test_split(df, test_weeks=test_weeks, val_frac=val_frac)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    models: Dict[int, object] = {}
    infos: Dict[int, dict] = {}

    for h in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"Training horizon h={h}  ({n_trials} Optuna trials)")
        print(f"{'='*60}")
        model, info = train_horizon(train_df, val_df, h, feature_cols, n_trials, seed)
        models[h] = model
        infos[h] = info

        model_path = output_dir / f"catboost_h{h}.cbm"
        model.save_model(str(model_path))
        print(f"  Saved: {model_path}")
        print(f"  Val MAE (orig units): {info['val_mae_orig']:.4f}")

    # Save metadata
    meta = {
        "feature_cols": feature_cols,
        "test_weeks": test_weeks,
        "val_frac": val_frac,
        "n_trials": n_trials,
        "seed": seed,
        "horizons": {h: infos[h] for h in [1, 2, 3]},
    }
    meta_path = output_dir / "training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\nMetadata saved: {meta_path}")

    # Also pickle the feature_cols list for downstream use
    with open(output_dir / "feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    return models


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Szabłowski CatBoost h=1,2,3")
    parser.add_argument(
        "--demand-path", type=Path,
        default=Path("data/processed/demand_long.parquet"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("models/szablowski"),
    )
    parser.add_argument("--test-weeks", type=int, default=18)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_all_horizons(
        args.demand_path,
        args.output_dir,
        args.test_weeks,
        args.val_frac,
        args.n_trials,
        args.seed,
    )


if __name__ == "__main__":
    main()
