
"""
Alpha Factor Registry.
Central source of truth for all alpha factors used in the strategy.
"""

# REFINED 6: Includes Range_Pos_20 for structural bull markets
# Last Updated: 2026-01-17
PRODUCTION_FACTORS = [
    # (Expression, Name)
    
    # === MOMENTUM ===
    ("$close / Ref($close, 60) - 1", "MOM60"),
    ("$close / Ref($close, 120) - 1", "MOM120"),
    
    # === VOLUME / FLOW ===
    # Vol_Price_Div_Rev: Sign-flipped to capture mean-reversion in modern choppy markets
    ("Mean(-1 * Corr($close / Ref($close, 1), $volume / Ref($volume, 1), 10), 5)", "Vol_Price_Div_Rev"),
    # Smoothed Money Flow
    ("Mean(Sum((($close - $low) - ($high - $close)) / ($high - $low + 0.001) * $volume, 20) / Sum($volume, 20), 5)", "Money_Flow_20"),

    # === BULL MARKET / STRUCTURAL ===
    # Range_Pos_20: Captures structural breakouts in trending markets
    ("($close - Min($low, 20)) / (Max($high, 20) - Min($low, 20) + 1e-4)", "Range_Pos_20"),

    # === MINED FACTORS ===
    ("-1 * (Sum(-1 * (Log($open)), 5) + Std($high, 5))", "Alpha_Gen_8"),
]



def get_production_factors():
    """Returns (expressions, names) tuple for production factors."""
    return list(zip(*PRODUCTION_FACTORS))
