import pytest
import pandas as pd
import numpy as np
from delorean.utils import smooth_predictions, calculate_rank_ic


class TestSmoothPredictions:
    """Tests for the prediction smoothing utility."""
    
    def test_smooth_empty_series(self):
        """Test handling of empty series."""
        empty = pd.Series(dtype=float)
        result = smooth_predictions(empty)
        assert result.empty
    
    def test_smooth_basic_structure(self):
        """Test that smoothing preserves index structure."""
        # Create mock prediction with MultiIndex (datetime, instrument)
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        instruments = ['A', 'B']
        
        idx = pd.MultiIndex.from_product(
            [dates, instruments], 
            names=['datetime', 'instrument']
        )
        pred = pd.Series(np.random.randn(20), index=idx)
        
        result = smooth_predictions(pred, halflife=3)
        
        assert not result.empty
        assert 'datetime' in result.index.names
        assert 'instrument' in result.index.names
        assert len(result) <= len(pred)
    
    def test_smooth_reduces_variance(self):
        """Test that smoothing reduces signal variance."""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        idx = pd.MultiIndex.from_product(
            [dates, ['A']], 
            names=['datetime', 'instrument']
        )
        # High variance raw signal
        pred = pd.Series(np.random.randn(50) * 10, index=idx)
        
        result = smooth_predictions(pred, halflife=10)
        
        # Smoothed signal should have lower variance
        assert result.std() < pred.std()


class TestCalculateRankIC:
    """Tests for Rank IC calculation."""
    
    def test_ic_perfect_correlation(self):
        """Test IC with perfectly correlated predictions."""
        idx = pd.MultiIndex.from_product(
            [pd.date_range('2020-01-01', periods=5), ['A', 'B']],
            names=['datetime', 'instrument']
        )
        pred = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=idx)
        labels = pred.copy()  # Perfect correlation
        
        ic = calculate_rank_ic(pred, labels)
        assert ic == pytest.approx(1.0, abs=0.01)
    
    def test_ic_no_common_indices(self):
        """Test IC when predictions and labels have no overlap."""
        idx1 = pd.MultiIndex.from_arrays(
            [pd.to_datetime(['2020-01-01']), ['A']],
            names=['datetime', 'instrument']
        )
        idx2 = pd.MultiIndex.from_arrays(
            [pd.to_datetime(['2020-01-02']), ['B']],
            names=['datetime', 'instrument']
        )
        
        pred = pd.Series([1.0], index=idx1)
        labels = pd.Series([1.0], index=idx2)
        
        ic = calculate_rank_ic(pred, labels)
        assert ic == 0.0
