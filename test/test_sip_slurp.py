"""Tests for SIP/SLURP"""

import pytest
import numpy as np
import pandas as pd
from vn2.uncertainty import SLURP, make_sip_from_uniform_threshold


def test_sip_from_uniform_threshold():
    """Test SIP creation from uniform with threshold"""
    u = np.array([0.02, 0.05, 0.10, 0.50])
    threshold = 0.06
    
    result = make_sip_from_uniform_threshold(u, threshold, 50000, 0)
    
    # 0.02 <= 0.06: success
    # 0.05 <= 0.06: success
    # 0.10 > 0.06: fail
    # 0.50 > 0.06: fail
    expected = np.array([50000, 50000, 0, 0])
    
    np.testing.assert_array_equal(result, expected)


def test_slurp_creation():
    """Test SLURP creation from dict"""
    sips = {
        "A": [1, 2, 3, 4],
        "B": [10, 20, 30, 40],
        "C": [100, 200, 300, 400]
    }
    
    slurp = SLURP.from_dict(sips, provenance="test")
    
    assert slurp.n_scenarios == 4
    assert list(slurp.names) == ["A", "B", "C"]
    assert slurp.provenance == "test"


def test_slurp_sample_preserves_relationships():
    """Test that row sampling preserves relationships"""
    # Create correlated SIPs
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 1000)
    
    sips = {
        "X": x,
        "Y": x * 2 + 1,  # Perfect linear relationship
    }
    
    slurp = SLURP.from_dict(sips)
    
    # Sample rows
    samples = slurp.sample_rows(100, seed=42)
    
    # Check relationship preserved
    corr = samples["X"].corr(samples["Y"])
    assert corr > 0.99  # Should be near perfect correlation


def test_slurp_sample_different_lengths_fail():
    """Test SLURP rejects SIPs of different lengths"""
    sips = {
        "A": [1, 2, 3],
        "B": [10, 20]  # Wrong length
    }
    
    with pytest.raises(ValueError, match="same length"):
        SLURP.from_dict(sips)

