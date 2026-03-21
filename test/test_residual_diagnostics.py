"""Unit tests for szablowski.residual_diagnostics."""

import numpy as np
import pandas as pd
import pytest

from szablowski.residual_diagnostics import (
    build_catboost_residuals,
    cross_model_comparison,
    generate_recommendation,
    residual_acf,
    residual_heteroskedasticity,
    residual_tails,
    taylors_law_on_residuals,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_residuals():
    """Residuals DataFrame with 3 SKUs and 20 weeks each."""
    np.random.seed(42)
    rows = []
    for store in [0]:
        for product in [100, 200, 300]:
            if product == 100:
                resids = np.random.normal(0, 2, 20)
                demand = np.random.poisson(10, 20).astype(float)
            elif product == 200:
                resids = np.random.normal(5, 10, 20)
                demand = np.random.poisson(50, 20).astype(float)
            else:
                resids = np.concatenate([np.zeros(15), [30, -20, 25, -15, 10]])
                demand = np.concatenate([np.zeros(15), [5, 0, 8, 0, 3]])

            for w in range(20):
                rows.append({
                    "Store": store,
                    "Product": product,
                    "week": w + 1,
                    "demand": demand[w],
                    "h1": demand[w] - resids[w],
                    "residual": resids[w],
                })
    return pd.DataFrame(rows)


@pytest.fixture
def forecasts_and_actuals(tmp_path):
    """Create forecast and actual parquet files for testing."""
    np.random.seed(42)
    fc = pd.DataFrame({
        "Store": [0] * 8 + [0] * 8,
        "Product": [100] * 8 + [200] * 8,
        "week": list(range(1, 9)) * 2,
        "h1": np.random.randint(5, 15, 16),
        "h2": np.random.randint(5, 15, 16),
        "h3": np.random.randint(5, 15, 16),
    })
    act = pd.DataFrame({
        "Store": [0] * 8 + [0] * 8,
        "Product": [100] * 8 + [200] * 8,
        "week": list(range(1, 9)) * 2,
        "actual_demand": np.random.randint(3, 18, 16),
    })
    fc_path = tmp_path / "forecasts.parquet"
    act_path = tmp_path / "actuals.parquet"
    fc.to_parquet(fc_path, index=False)
    act.to_parquet(act_path, index=False)
    return fc_path, act_path


# ---------------------------------------------------------------------------
# Tests: ACF
# ---------------------------------------------------------------------------

class TestResidualACF:
    def test_returns_one_row_per_sku(self, synthetic_residuals):
        result = residual_acf(synthetic_residuals, lags=[1, 4])
        assert len(result) == 3

    def test_has_expected_columns(self, synthetic_residuals):
        result = residual_acf(synthetic_residuals, lags=[1, 4])
        assert "lb_stat_1" in result.columns
        assert "lb_pval_1" in result.columns
        assert "lb_stat_4" in result.columns
        assert "any_significant_acf" in result.columns

    def test_pvalues_in_range(self, synthetic_residuals):
        result = residual_acf(synthetic_residuals, lags=[1, 4])
        for col in ["lb_pval_1", "lb_pval_4"]:
            vals = result[col].dropna()
            assert (vals >= 0).all()
            assert (vals <= 1).all()


# ---------------------------------------------------------------------------
# Tests: Heteroskedasticity
# ---------------------------------------------------------------------------

class TestResidualHeteroskedasticity:
    def test_returns_one_row_per_sku(self, synthetic_residuals):
        result = residual_heteroskedasticity(synthetic_residuals)
        assert len(result) == 3

    def test_has_expected_columns(self, synthetic_residuals):
        result = residual_heteroskedasticity(synthetic_residuals)
        for col in ["resid_mean", "resid_std", "resid_var",
                     "demand_mean", "bp_stat", "bp_pval",
                     "heteroskedastic_resid"]:
            assert col in result.columns

    def test_variance_non_negative(self, synthetic_residuals):
        result = residual_heteroskedasticity(synthetic_residuals)
        assert (result["resid_var"].dropna() >= 0).all()


class TestTaylorsLaw:
    def test_returns_bins(self, synthetic_residuals):
        hetero = residual_heteroskedasticity(synthetic_residuals)
        taylor = taylors_law_on_residuals(hetero, n_bins=2)
        assert len(taylor) >= 1
        assert "mean_demand" in taylor.columns
        assert "mean_resid_var" in taylor.columns


# ---------------------------------------------------------------------------
# Tests: Tail behaviour
# ---------------------------------------------------------------------------

class TestResidualTails:
    def test_returns_one_row_per_sku(self, synthetic_residuals):
        result = residual_tails(synthetic_residuals)
        assert len(result) == 3

    def test_has_expected_columns(self, synthetic_residuals):
        result = residual_tails(synthetic_residuals)
        for col in ["kurtosis", "skewness", "jb_stat", "jb_pval",
                     "frac_beyond_2sigma", "frac_beyond_3sigma",
                     "normal_residuals"]:
            assert col in result.columns

    def test_sigma_fractions_bounded(self, synthetic_residuals):
        result = residual_tails(synthetic_residuals)
        for col in ["frac_beyond_2sigma", "frac_beyond_3sigma"]:
            vals = result[col].dropna()
            assert (vals >= 0).all()
            assert (vals <= 1).all()

    def test_normal_residuals_detected(self):
        """Large sample from N(0,1) should pass Jarque-Bera."""
        np.random.seed(42)
        resid = np.random.normal(0, 1, 200)
        df = pd.DataFrame({
            "Store": [0] * 200,
            "Product": [1] * 200,
            "week": range(200),
            "residual": resid,
        })
        result = residual_tails(df)
        assert result["normal_residuals"].iloc[0] == True


# ---------------------------------------------------------------------------
# Tests: Build residuals
# ---------------------------------------------------------------------------

class TestBuildCatboostResiduals:
    def test_produces_residuals(self, forecasts_and_actuals):
        fc_path, act_path = forecasts_and_actuals
        result = build_catboost_residuals(fc_path, act_path)
        assert "residual" in result.columns
        assert len(result) == 16

    def test_residual_is_actual_minus_forecast(self, forecasts_and_actuals):
        fc_path, act_path = forecasts_and_actuals
        result = build_catboost_residuals(fc_path, act_path)
        expected = result["demand"] - result["h1"]
        np.testing.assert_array_equal(result["residual"].values, expected.values)


# ---------------------------------------------------------------------------
# Tests: Recommendation
# ---------------------------------------------------------------------------

class TestGenerateRecommendation:
    def test_produces_nonempty_string(self, synthetic_residuals):
        acf = residual_acf(synthetic_residuals, lags=[1, 4])
        hetero = residual_heteroskedasticity(synthetic_residuals)
        tails = residual_tails(synthetic_residuals)
        taylor = taylors_law_on_residuals(hetero, n_bins=2)
        rec = generate_recommendation(acf, hetero, tails, taylor)
        assert len(rec) > 100
        assert "Recommendation" in rec

    def test_recommendation_with_cross_model(self, synthetic_residuals):
        acf = residual_acf(synthetic_residuals, lags=[1, 4])
        hetero = residual_heteroskedasticity(synthetic_residuals)
        tails = residual_tails(synthetic_residuals)
        taylor = taylors_law_on_residuals(hetero, n_bins=2)
        cross = pd.DataFrame({
            "model_name": ["model_a", "model_b"],
            "mean_mae": [1.0, 2.0],
            "mean_bias": [0.1, -0.5],
            "mean_crps": [0.5, 1.0],
            "rmse_mae_ratio": [1.2, 1.5],
        })
        rec = generate_recommendation(acf, hetero, tails, taylor, cross)
        assert "model_a" in rec
