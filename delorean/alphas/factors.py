
"""
Alpha Factor Registry.
Central source of truth for all alpha factors used in the strategy.
"""

# ULTRA 6 (1-Day Optimized): High-IC Selection Library
# Focused on finding leadership and stability for daily rebalancing.
# Last Updated: 2026-01-18
PRODUCTION_FACTORS = [
    # (Expression, Name)
    
    # === MOMENTUM & SKEW ===
    # Vol_Skew_20: Anti-lottery effect. 2022-2025 IC: +0.054
    ("-1 * Skew($close / Ref($close, 1) - 1, 20)", "Vol_Skew_20"),
    # Selection_Trend: GP-Mined momentum multiplier. 2022-2025 IC: +0.028
    ("Log(Abs($close / Ref($close, 20) - 1) + 1.0001) * Power(($close / Mean($close, 60) - 1), 2)", "Selection_Trend"),
    
    # === GP-MINED CORE ===
    # Alpha_Gen_8: GP-Mined price-range signal. 2022-2025 IC: +0.029
    ("-1 * (Sum(-1 * (Log($open + 1e-5)), 5) + Std($high, 5))", "Alpha_Gen_8"),
    
    # === VOL-PRICE DYNAMICS ===
    # Vol_Price_Div_Rev: Mean-reversion confirmation. 2022-2025 IC: +0.029
    ("Mean(-1 * Corr($close / Ref($close, 1), $volume / Ref($volume, 1), 10), 5)", "Vol_Price_Div_Rev"),
    
    # === SHORT-TERM MEAN REVERSION (REVERSED) ===
    # Smart_Flow_Rev: Captures liquidity exhaustion. 2022-2025 IC: +0.032
    ("-1 * ($close - $low) / ($high - $low + 0.001) * (Mean($volume, 5) / Mean($volume, 20))", "Smart_Flow_Rev"),
    # Gap_Fill_Rev: Short-term price reversal from gap extremes. 2022-2025 IC: +0.027
    ("-1 * ($close - $open) / (Abs($open - Ref($close, 1)) + 0.001)", "Gap_Fill_Rev"),
]



def get_production_factors():
    """Returns (expressions, names) tuple for production factors."""
    return list(zip(*PRODUCTION_FACTORS))
