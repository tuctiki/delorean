
"""
Alpha Factor Registry.
Central source of truth for all alpha factors used in the strategy.
"""

# ULTRA 8 (Inherently Stable): Redesigned for Low Turnover
# Using longer lookback windows (20-60 days) and MA ratios
# Last Updated: 2026-01-18 (Fundamental Redesign for Stability)
PRODUCTION_FACTORS = [
    # (Expression, Name)
    
    # === MOMENTUM (STABLE) ===
    # Momentum_20: Simple 20-day momentum - changes slowly
    ("$close / Ref($close, 20) - 1", "Momentum_20"),
    
    # Momentum_60: Long-term 60-day momentum
    ("$close / Ref($close, 60) - 1", "Momentum_60"),
    
    # === MOVING AVERAGE RATIOS (VERY STABLE) ===
    # MA_Ratio_20_60: Price relative to moving averages (changes slowly)
    ("Mean($close, 20) / (Mean($close, 60) + 1e-6) - 1", "MA_Ratio_20_60"),
    
    # === VOLATILITY-ADJUSTED RETURNS ===
    # RiskAdjReturn_60: 60-day return / 60-day volatility
    ("($close / Ref($close, 60) - 1) / (Std($close / Ref($close, 1), 60) + 0.01)", "RiskAdjReturn_60"),
    
    # === PRICE POSITION (RANGE-BASED, STABLE) ===
    # PricePosition_60: Where price is within its 60-day range
    ("($close - Min($close, 60)) / (Max($close, 60) - Min($close, 60) + 1e-6)", "PricePosition_60"),
]


def get_production_factors():
    """Returns (expressions, names) tuple for production factors."""
    return list(zip(*PRODUCTION_FACTORS))
