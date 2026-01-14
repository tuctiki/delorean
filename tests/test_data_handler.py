import pytest
import pandas as pd
import numpy as np
from delorean.data import ETFDataHandler

def test_custom_factors_structure():
    """Test that get_custom_factors returns correct tuple structure."""
    exprs, names = ETFDataHandler.get_custom_factors()
    
    assert isinstance(exprs, list)
    assert isinstance(names, list)
    assert len(exprs) == len(names)
    assert len(names) == 7, f"Expected 7 factors, got {len(names)}"
    
def test_custom_factors_contains_expected_factors():
    """Test that all expected factor names are present (2026-01-14 Optimized 7-Factor Library)."""
    _, names = ETFDataHandler.get_custom_factors()
    
    # Optimized 7-factor library after 2026-01-14 audit
    # Removed: VOL60 (negative IC), Mom20_VolAdj (redundant), Accel_Rev (redundant)
    expected_factors = [
        "MarketCap_Liquidity",
        "MOM60",
        "MOM120", 
        "Trend_Efficiency",
        "Gap_Fill",
        "Mom_Persistence",
        "Acceleration",
    ]
    
    for factor in expected_factors:
        assert factor in names, f"Expected factor {factor} not found in custom factors"

def test_custom_factors_expressions_are_valid_strings():
    """Test that all expressions are non-empty strings."""
    exprs, _ = ETFDataHandler.get_custom_factors()
    
    for expr in exprs:
        assert isinstance(expr, str)
        assert len(expr) > 0
        # Basic sanity check - should contain $ for Qlib field references
        assert "$" in expr, f"Expression '{expr}' doesn't appear to use Qlib field syntax"
