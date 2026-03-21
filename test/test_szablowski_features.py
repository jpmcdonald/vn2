"""Unit tests for szablowski.features — feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from szablowski.features import (
    build_features,
    competition_split,
    compute_effective_sales,
    compute_sample_weights,
    compute_scale_factors,
    rolling_cv_folds,
    train_val_test_split,
    two_level_median_imputation,
    _lag_features,
    _seasonality_features,
    _spike_features,
)
from szablowski.predict import apply_dormancy_filter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df():
    """3 SKUs, 60 weeks of synthetic demand data."""
    np.random.seed(42)
    rows = []
    for store in [0]:
        for product in [100, 200, 300]:
            base = 10 if product == 100 else 50 if product == 200 else 0
            for w in range(60):
                week_date = pd.Timestamp("2022-01-03") + pd.Timedelta(weeks=w)
                demand = max(0, int(base + np.random.normal(0, 3)))
                in_stock = not (product == 100 and 20 <= w <= 22)
                rows.append({
                    "Store": store,
                    "Product": product,
                    "week": week_date,
                    "demand": demand,
                    "in_stock": in_stock,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeEffectiveSales:
    def test_stockout_weeks_become_nan(self, synthetic_df):
        result = compute_effective_sales(synthetic_df)
        sku100 = result[(result["Store"] == 0) & (result["Product"] == 100)]
        stockout_rows = sku100[(sku100.index >= 0) & (~sku100["in_stock"])]
        assert stockout_rows["y_eff"].isna().all()

    def test_in_stock_weeks_preserved(self, synthetic_df):
        result = compute_effective_sales(synthetic_df)
        in_stock_rows = result[result["in_stock"]]
        assert in_stock_rows["y_eff"].notna().all()


class TestComputeScaleFactors:
    def test_constant_series_scale(self):
        n = 60
        df = pd.DataFrame({
            "Store": [0] * n,
            "Product": [1] * n,
            "week": pd.date_range("2022-01-03", periods=n, freq="W-MON"),
            "demand": [10] * n,
            "in_stock": [True] * n,
        })
        df = compute_effective_sales(df)
        sf = compute_scale_factors(df)
        # For a constant series of 10, scale = 53 * 10 = 530
        last_scale = sf["scale_factor"].iloc[-1]
        assert last_scale == pytest.approx(530.0, rel=0.01)

    def test_scale_floor_is_one(self):
        n = 60
        df = pd.DataFrame({
            "Store": [0] * n,
            "Product": [1] * n,
            "week": pd.date_range("2022-01-03", periods=n, freq="W-MON"),
            "demand": [0] * n,
            "in_stock": [True] * n,
        })
        df = compute_effective_sales(df)
        sf = compute_scale_factors(df)
        assert (sf["scale_factor"] >= 1.0).all()


class TestLagFeatures:
    def test_lag0_is_current(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        feats = _lag_features(y)
        np.testing.assert_array_equal(feats["lag_0"], y)

    def test_lag1_is_previous(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        feats = _lag_features(y)
        assert np.isnan(feats["lag_1"][0])
        np.testing.assert_array_equal(feats["lag_1"][1:], [1.0, 2.0, 3.0, 4.0])

    def test_nan_at_boundaries(self):
        y = np.array([1.0, 2.0, 3.0])
        feats = _lag_features(y)
        assert np.isnan(feats["lag_51"][0])
        assert np.isnan(feats["lag_52"][0])
        assert np.isnan(feats["lag_53"][0])


class TestSeasonalityFeatures:
    def test_fourier_range(self):
        woy = np.arange(1, 53, dtype=float)
        y_eff = np.random.rand(52)
        feats = _seasonality_features(woy, y_eff, n_harmonics=3)
        for k in range(1, 4):
            assert np.all(feats[f"fourier_sin_{k}"] >= -1.0)
            assert np.all(feats[f"fourier_sin_{k}"] <= 1.0)
            assert np.all(feats[f"fourier_cos_{k}"] >= -1.0)
            assert np.all(feats[f"fourier_cos_{k}"] <= 1.0)

    def test_seasonality_strength_bounded(self):
        woy = np.tile(np.arange(1, 53, dtype=float), 3)
        y_eff = np.random.rand(156)
        feats = _seasonality_features(woy, y_eff, n_harmonics=3)
        ss = feats["seasonality_strength"]
        valid = ss[~np.isnan(ss)]
        assert np.all(valid >= -1.0)
        assert np.all(valid <= 1.0)


class TestSpikeFeatures:
    def test_time_since_spike_resets(self):
        # Need varied data so that rolling MAD is nonzero, and the spike has |z| > 3
        np.random.seed(99)
        base = np.random.randint(5, 15, size=30).astype(float)
        base[20] = 200.0  # large spike relative to local median ~10
        y = pd.Series(base)
        feats = _spike_features(y)
        tss = feats["time_since_spike"]
        spike_idx = np.where(~np.isnan(tss) & (tss == 0))[0]
        assert len(spike_idx) > 0


class TestTwoLevelMedianImputation:
    def test_no_nans_remain(self, synthetic_df):
        df = compute_effective_sales(synthetic_df)
        df["test_col"] = df["y_eff"]
        result = two_level_median_imputation(df, ["test_col"])
        assert result["test_col"].isna().sum() == 0


class TestComputeSampleWeights:
    def test_recent_weeks_weight_one(self, synthetic_df):
        df = compute_effective_sales(synthetic_df)
        weights = compute_sample_weights(df)
        sku = df[(df["Store"] == 0) & (df["Product"] == 100)]
        sku_weights = weights[sku.index]
        assert sku_weights[-1] == 1.0

    def test_all_weights_positive(self, synthetic_df):
        df = compute_effective_sales(synthetic_df)
        weights = compute_sample_weights(df)
        assert np.all(weights > 0)


class TestBuildFeatures:
    def test_returns_expected_columns(self, synthetic_df):
        df, feature_cols = build_features(synthetic_df)
        assert "y_scaled" in df.columns
        assert "scale_factor" in df.columns
        assert "sample_weight" in df.columns
        assert "store_id" in feature_cols
        assert "product_id" in feature_cols
        assert len(feature_cols) > 10

    def test_no_nans_after_imputation(self, synthetic_df):
        df, feature_cols = build_features(synthetic_df)
        numeric_feats = [c for c in feature_cols if c not in ("store_id", "product_id")]
        nan_counts = df[numeric_feats].isna().sum()
        assert nan_counts.sum() == 0, f"NaNs found: {nan_counts[nan_counts > 0]}"


class TestTrainValTestSplit:
    def test_chronological_ordering(self, synthetic_df):
        df, _ = build_features(synthetic_df)
        train, val, test = train_val_test_split(df, test_weeks=10, val_frac=0.10)
        wc = "week"
        if not train.empty and not val.empty:
            assert train[wc].max() < val[wc].min()
        if not val.empty and not test.empty:
            assert val[wc].max() < test[wc].min()

    def test_no_data_leakage(self, synthetic_df):
        df, _ = build_features(synthetic_df)
        train, val, test = train_val_test_split(df, test_weeks=10, val_frac=0.10)
        if not train.empty and not test.empty:
            train_weeks = set(train["week"].unique())
            test_weeks = set(test["week"].unique())
            assert train_weeks.isdisjoint(test_weeks)


# ---------------------------------------------------------------------------
# New split functions
# ---------------------------------------------------------------------------

@pytest.fixture
def longer_df():
    """3 SKUs, 165 weeks mirroring the real data timeline."""
    np.random.seed(42)
    rows = []
    start = pd.Timestamp("2021-04-12")
    for store in [0]:
        for product in [100, 200]:
            for w in range(165):
                week_date = start + pd.Timedelta(weeks=w)
                demand = max(0, int(10 + np.random.normal(0, 3)))
                rows.append({
                    "Store": store,
                    "Product": product,
                    "week": week_date,
                    "demand": demand,
                    "in_stock": True,
                })
    return pd.DataFrame(rows)


class TestCompetitionSplit:
    def test_test_set_size(self, longer_df):
        df, _ = build_features(longer_df)
        _, _, comp_test = competition_split(df, n_competition_weeks=8)
        n_test_weeks = comp_test["week"].nunique()
        assert n_test_weeks == 8

    def test_no_overlap(self, longer_df):
        df, _ = build_features(longer_df)
        train_all, early_stop, comp_test = competition_split(df, n_competition_weeks=8)
        train_weeks = set(train_all["week"].unique())
        es_weeks = set(early_stop["week"].unique())
        test_weeks = set(comp_test["week"].unique())
        assert train_weeks.isdisjoint(test_weeks)
        assert es_weeks.isdisjoint(test_weeks)
        assert train_weeks.isdisjoint(es_weeks)

    def test_train_contains_recent_data(self, longer_df):
        """Train set should include data right up to the competition start."""
        df, _ = build_features(longer_df)
        train_all, _, comp_test = competition_split(df, n_competition_weeks=8)
        comp_start = comp_test["week"].min()
        # The prior-year window is removed from train, but weeks between
        # the prior-year window end and comp_start should be in train.
        recent_in_train = train_all[train_all["week"] >= comp_start - pd.Timedelta(weeks=10)]
        assert len(recent_in_train) > 0


class TestRollingCVFolds:
    def test_fold_count(self, longer_df):
        df, _ = build_features(longer_df)
        folds = rolling_cv_folds(df, n_folds=5, n_val_weeks=3,
                                 n_competition_weeks=8)
        assert len(folds) == 5

    def test_no_future_leak(self, longer_df):
        df, _ = build_features(longer_df)
        folds = rolling_cv_folds(df, n_folds=3, n_val_weeks=3,
                                 n_competition_weeks=8)
        for train_fold, val_fold in folds:
            assert train_fold["week"].max() < val_fold["week"].min()

    def test_val_weeks_correct_size(self, longer_df):
        df, _ = build_features(longer_df)
        folds = rolling_cv_folds(df, n_folds=3, n_val_weeks=3,
                                 n_competition_weeks=8)
        for _, val_fold in folds:
            assert val_fold["week"].nunique() == 3

    def test_no_competition_data(self, longer_df):
        df, _ = build_features(longer_df)
        weeks_sorted = np.sort(df["week"].unique())
        comp_start = weeks_sorted[-8]
        folds = rolling_cv_folds(df, n_folds=3, n_val_weeks=3,
                                 n_competition_weeks=8)
        for train_fold, val_fold in folds:
            assert train_fold["week"].max() < comp_start
            assert val_fold["week"].max() < comp_start


# ---------------------------------------------------------------------------
# Dormancy filter
# ---------------------------------------------------------------------------

class TestDormancyFilter:
    def test_dormant_sku_zeroed(self):
        """SKU with all-zero recent demand should get forecasts zeroed."""
        demand_df = pd.DataFrame({
            "Store": [0] * 20,
            "Product": [1] * 20,
            "week": pd.date_range("2023-01-02", periods=20, freq="W-MON"),
            "demand": [5] * 10 + [0] * 10,
            "in_stock": [True] * 20,
        })
        forecasts_df = pd.DataFrame({
            "Store": [0],
            "Product": [1],
            "week": [pd.Timestamp("2023-05-22")],
            "h1": [10],
            "h2": [12],
            "h3": [8],
        })
        result = apply_dormancy_filter(forecasts_df, demand_df, lookback=10)
        assert result["h1"].iloc[0] == 0
        assert result["h2"].iloc[0] == 0
        assert result["h3"].iloc[0] == 0
        assert result["dormancy_filtered"].iloc[0] == True

    def test_active_sku_untouched(self):
        """SKU with recent non-zero demand should be unchanged."""
        demand_df = pd.DataFrame({
            "Store": [0] * 20,
            "Product": [1] * 20,
            "week": pd.date_range("2023-01-02", periods=20, freq="W-MON"),
            "demand": [5] * 20,
            "in_stock": [True] * 20,
        })
        forecasts_df = pd.DataFrame({
            "Store": [0],
            "Product": [1],
            "week": [pd.Timestamp("2023-05-22")],
            "h1": [10],
            "h2": [12],
            "h3": [8],
        })
        result = apply_dormancy_filter(forecasts_df, demand_df, lookback=10)
        assert result["h1"].iloc[0] == 10
        assert result["dormancy_filtered"].iloc[0] == False

    def test_short_history_not_filtered(self):
        """SKU with fewer weeks than lookback should not be filtered."""
        demand_df = pd.DataFrame({
            "Store": [0] * 5,
            "Product": [1] * 5,
            "week": pd.date_range("2023-01-02", periods=5, freq="W-MON"),
            "demand": [0] * 5,
            "in_stock": [True] * 5,
        })
        forecasts_df = pd.DataFrame({
            "Store": [0],
            "Product": [1],
            "week": [pd.Timestamp("2023-02-06")],
            "h1": [10],
            "h2": [12],
            "h3": [8],
        })
        result = apply_dormancy_filter(forecasts_df, demand_df, lookback=10)
        assert result["h1"].iloc[0] == 10
