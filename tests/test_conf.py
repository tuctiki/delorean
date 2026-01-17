
import pytest
import datetime
from delorean.conf import assets, system, model, strategy

def test_assets_config():
    """Verify assets config exports expected lists."""
    assert hasattr(assets, "ETF_LIST")
    assert isinstance(assets.ETF_LIST, list)
    assert len(assets.ETF_LIST) > 0
    assert hasattr(assets, "BENCHMARK")
    assert isinstance(assets.BENCHMARK, str)

def test_system_config():
    """Verify system config exports paths and time."""
    assert hasattr(system, "QLIB_PROVIDER_URI")
    assert hasattr(system, "START_TIME")
    # Verify date format YYYY-MM-DD
    datetime.datetime.strptime(system.START_TIME, "%Y-%m-%d")

def test_strategy_config():
    """Verify strategy parameters."""
    assert hasattr(strategy, "DEFAULT_BACKTEST_PARAMS")
    params = strategy.DEFAULT_BACKTEST_PARAMS
    assert "topk" in params
    assert "target_vol" in params
    assert isinstance(params["topk"], int)

def test_backward_compatibility():
    """Verify old delorean.config still works."""
    from delorean import config
    assert config.ETF_LIST == assets.ETF_LIST
    assert config.QLIB_PROVIDER_URI == system.QLIB_PROVIDER_URI
