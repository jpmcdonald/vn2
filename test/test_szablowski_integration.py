"""Integration tests for the szablowski pipeline — end-to-end on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from szablowski.features import build_features
from szablowski.train import build_targets

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from szablowski.calibrate_phi import simulate_with_phi


@pytest.fixture
def synthetic_demand_df():
    """3 SKUs, 80 weeks, enough for feature warm-up and target creation."""
    np.random.seed(42)
    rows = []
    for store in [0]:
        for product in [100, 200, 300]:
            base = 10 if product == 100 else 50 if product == 200 else 5
            for w in range(80):
                week_date = pd.Timestamp("2022-01-03") + pd.Timedelta(weeks=w)
                demand = max(0, int(base + np.random.normal(0, max(base * 0.3, 1))))
                rows.append({
                    "Store": store,
                    "Product": product,
                    "week": week_date,
                    "demand": demand,
                    "in_stock": True,
                })
    return pd.DataFrame(rows)


@pytest.mark.skipif(not HAS_CATBOOST, reason="catboost not installed")
class TestEndToEnd:
    def test_features_targets_train_predict(self, synthetic_demand_df):
        """build_features -> build_targets -> CatBoost train (1 trial) -> predict."""
        df, feature_cols = build_features(synthetic_demand_df)
        df = build_targets(df)

        # Simple train/test split on last 10 weeks
        weeks_sorted = np.sort(df["week"].unique())
        cutoff = weeks_sorted[-10]
        train = df[df["week"] < cutoff].dropna(subset=["y_h1"])
        test = df[df["week"] >= cutoff]

        assert len(train) > 0
        assert len(test) > 0

        numeric_feats = [c for c in feature_cols if c not in ("store_id", "product_id")]
        cat_feats = [i for i, c in enumerate(feature_cols) if c in ("store_id", "product_id")]

        X_train = train[feature_cols]
        y_train = train["y_h1"].values
        w_train = train["sample_weight"].values

        model = cb.CatBoostRegressor(
            iterations=50,
            depth=4,
            learning_rate=0.1,
            loss_function="RMSE",
            verbose=0,
            random_seed=42,
        )
        pool = cb.Pool(X_train, y_train, weight=w_train, cat_features=cat_feats)
        model.fit(pool)

        X_test = test[feature_cols]
        preds_scaled = model.predict(X_test)
        preds_orig = np.clip(np.round(preds_scaled * test["scale_factor"].values), 0, None).astype(int)

        assert len(preds_orig) == len(test)
        assert np.all(preds_orig >= 0)
        assert preds_orig.dtype == int


class TestSimulateWithPhi:
    def test_cost_positive_on_synthetic_data(self):
        forecasts_df = pd.DataFrame({
            "Store": [0, 0, 0] * 3,
            "Product": [100, 200, 300] * 3,
            "week": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "h1": [10, 50, 5] * 3,
            "h2": [10, 50, 5] * 3,
            "h3": [10, 50, 5] * 3,
        })

        actuals_df = pd.DataFrame({
            "Store": [0, 0, 0] * 3,
            "Product": [100, 200, 300] * 3,
            "week": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "actual_demand": [12, 55, 3, 8, 48, 7, 10, 52, 4],
        })

        initial_states = {
            (0, 100): {"on_hand": 20, "in_transit": [10, 10, 0]},
            (0, 200): {"on_hand": 100, "in_transit": [50, 50, 0]},
            (0, 300): {"on_hand": 10, "in_transit": [5, 5, 0]},
        }

        h, s, total, detail = simulate_with_phi(
            forecasts_df, actuals_df, initial_states,
            phi=1.0, n_weeks=3,
        )

        assert total >= 0
        assert len(detail) == 9  # 3 SKUs * 3 weeks
        assert "holding_cost" in detail.columns
        assert "shortage_cost" in detail.columns
