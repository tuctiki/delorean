import pytest
import pandas as pd
import numpy as np
from delorean.data import ETFDataHandler

def test_custom_factors_structure():
    """Test that get_custom_factors returns correct tuple structure."""
    exprs, names = ETFDataHandler.get_custom_factors()
    
    assert isinstance(exprs, (list, tuple))
    assert isinstance(names, (list, tuple))
    assert len(exprs) == len(names)
    assert len(names) == 5, f"Expected 5 factors, got {len(names)}"
    
def test_custom_factors_contains_expected_factors():
    """Test that all expected factor names are present."""
    _, names = ETFDataHandler.get_custom_factors()
    
    expected_factors = [
        "MOM120",
        "Mom_Persistence",
        "Money_Flow_20",
        "RSI_Div_Cond",
        "Alpha_Gen_8",
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
