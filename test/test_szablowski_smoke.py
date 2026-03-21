"""Smoke tests on real data (first 10 SKUs) for the szablowski pipeline."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DEMAND_PATH = Path("data/processed/demand_long.parquet")
HAS_DATA = DEMAND_PATH.exists()


@pytest.fixture
def first_10_skus():
    """Load first 10 SKUs from demand_long.parquet."""
    df = pd.read_parquet(DEMAND_PATH)
    skus = df.groupby(["Store", "Product"]).ngroups
    unique_skus = df[["Store", "Product"]].drop_duplicates().head(10)
    subset = df.merge(unique_skus, on=["Store", "Product"])
    return subset


@pytest.mark.skipif(not HAS_DATA, reason="data/processed/demand_long.parquet not found")
class TestSmokeFeatures:
    def test_build_features_no_exceptions(self, first_10_skus):
        from szablowski.features import build_features
        df, feature_cols = build_features(first_10_skus)
        assert len(df) > 0
        assert len(feature_cols) > 10

    def test_expected_columns_present(self, first_10_skus):
        from szablowski.features import build_features
        df, feature_cols = build_features(first_10_skus)
        assert "y_scaled" in df.columns
        assert "scale_factor" in df.columns
        assert "sample_weight" in df.columns
        for col in ["lag_0", "lag_1", "rmean_3", "fourier_sin_1", "robust_zscore"]:
            assert col in feature_cols or col in df.columns, f"Missing {col}"


@pytest.mark.skipif(not HAS_DATA, reason="data/processed/demand_long.parquet not found")
class TestSmokeEDABase:
    def test_generate_eda_base_no_exceptions(self, first_10_skus, tmp_path):
        from szablowski.generate_eda_base import _summary_for_sku, _stationarity_for_sku

        grp = first_10_skus.groupby(["Store", "Product"]).first()
        store, product = grp.index[0]
        y = first_10_skus[
            (first_10_skus["Store"] == store) & (first_10_skus["Product"] == product)
        ]["demand"].values

        summary = _summary_for_sku(store, product, y)
        assert "mean" in summary
        assert "cv" in summary
        assert summary["Store"] == store

        stationarity = _stationarity_for_sku(store, product, y)
        assert "adf_stat" in stationarity
        assert stationarity["Store"] == store


@pytest.mark.skipif(not HAS_DATA, reason="data/processed/demand_long.parquet not found")
class TestSmokeEDAExtensions:
    def test_taylor_on_real_data(self, first_10_skus):
        from szablowski.eda_extensions import taylor_alpha_per_sku
        result = taylor_alpha_per_sku(first_10_skus)
        assert len(result) == 10
        assert "taylor_alpha" in result.columns

    def test_cusum_on_real_data(self, first_10_skus):
        from szablowski.eda_extensions import cusum_break_test
        result = cusum_break_test(first_10_skus)
        assert len(result) == 10
        assert "has_structural_break" in result.columns

    def test_ljung_box_on_real_data(self, first_10_skus):
        from szablowski.eda_extensions import ljung_box_test
        result = ljung_box_test(first_10_skus)
        assert len(result) == 10
        assert "ljung_box_pval_12" in result.columns

    def test_count_gof_on_real_data(self, first_10_skus):
        from szablowski.eda_extensions import count_distribution_gof
        result = count_distribution_gof(first_10_skus)
        assert len(result) == 10
        assert "negbin_preferred" in result.columns
