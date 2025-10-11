"""Tests for submission builder"""

import pytest
import pandas as pd
from pathlib import Path
from vn2.submit import build_submission, validate_submission


@pytest.fixture
def simple_index():
    """Simple canonical index"""
    return pd.MultiIndex.from_tuples(
        [(0, 100), (0, 101), (1, 100)],
        names=["Store", "Product"]
    )


def test_build_submission(simple_index, tmp_path):
    """Test submission building with correct format"""
    orders = pd.Series([10, 20, 5], index=simple_index)
    
    out_file = tmp_path / "test.csv"
    build_submission(simple_index, orders, str(out_file))
    
    # Read back
    df = pd.read_csv(out_file)
    
    assert len(df) == 3
    assert list(df.columns) == ["Store", "Product", "0"]
    assert df["0"].tolist() == [10, 20, 5]


def test_validate_submission_success(simple_index):
    """Test validation passes for correct submission"""
    df = pd.DataFrame({
        "Store": [0, 0, 1],
        "Product": [100, 101, 100],
        "0": [10, 20, 5]
    })
    
    # Should not raise
    validate_submission(df, simple_index)


def test_validate_submission_wrong_length(simple_index):
    """Test validation fails for wrong row count"""
    df = pd.DataFrame({
        "Store": [0, 0],
        "Product": [100, 101],
        "0": [10, 20]
    })
    
    with pytest.raises(ValueError, match="rows"):
        validate_submission(df, simple_index)


def test_validate_submission_negative(simple_index):
    """Test validation fails for negative orders"""
    df = pd.DataFrame({
        "Store": [0, 0, 1],
        "Product": [100, 101, 100],
        "0": [10, -5, 5]
    })
    
    with pytest.raises(ValueError, match="negative"):
        validate_submission(df, simple_index)


def test_validate_submission_nan(simple_index):
    """Test validation fails for NaN values"""
    df = pd.DataFrame({
        "Store": [0, 0, 1],
        "Product": [100, 101, 100],
        "0": [10, None, 5]
    })
    
    with pytest.raises(ValueError, match="NaN"):
        validate_submission(df, simple_index)

