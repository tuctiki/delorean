
import pytest
from delorean.alphas.factors import get_production_factors
from delorean.conf import ETF_LIST

def test_production_factors_syntax():
    """Verify all production factors use valid Qlib expression syntax."""
    exprs, names = get_production_factors()
    
    assert len(exprs) > 0
    assert len(exprs) == len(names)
    
    for expr in exprs:
        assert isinstance(expr, str)
        # Check for balanced parentheses
        assert expr.count("(") == expr.count(")")
        # Check for core operators or fields
        assert "$" in expr or "Ref" in expr or "Sum" in expr or "Mean" in expr

def test_production_factors_unique_names():
    """Verify all factor names are unique."""
    _, names = get_production_factors()
    assert len(names) == len(set(names)), "Duplicate factor names found"
