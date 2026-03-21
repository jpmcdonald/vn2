"""
Szabłowski CatBoost pipeline reproduction for VN2 backtesting comparison.

Implements the winning VN2 solution (arXiv:2601.18919v1):
  - Global CatBoost forecaster with stockout-aware features and dynamic scaling
  - Analytical newsvendor ordering policy with φ√D uncertainty proxy

Kept separate from the main vn2 package to avoid contamination.
"""
