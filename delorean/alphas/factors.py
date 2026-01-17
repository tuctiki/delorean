
"""
Alpha Factor Registry.
Central source of truth for all alpha factors used in the strategy.
"""

# Current Production Factors (5-Factor Library)
# Last Updated: 2026-01-17
# Factors Audited: 2026-01-15 (Simulated)
PRODUCTION_FACTORS = [
    # (Expression, Name)
    
    # === BASELINE ===
    ("$close / Ref($close, 120) - 1", "MOM120"), # Robust Momentum
    
    # === PERSISTENCE ===
    ("Sum(If($close > Ref($close, 1), 1, 0), 10) / 10", "Mom_Persistence"), # Consistent Up days
    
    # === MONEY FLOW ===
    # Smoothed Money Flow (Robust)
    ("Mean(Sum((($close - $low) - ($high - $close)) / ($high - $low + 0.001) * $volume, 20) / Sum($volume, 20), 5)", "Money_Flow_20"),

    # === CONDITIONAL FACTORS (Flip in Bull Regime) ===
    # RSI_Divergence: Unstable sign. Flip to NEGATIVE in Bull (>1.0).
    # Base: Corr(Price, RSI_Diff, 10)
    ("If($close / Mean($close, 60) > 1, -1 * Corr($close / Ref($close, 1) - 1, (Mean(If($close > Ref($close, 1), $close - Ref($close, 1), 0), 14) / (Mean(Abs($close - Ref($close, 1)), 14) + 0.0001)), 10), Corr($close / Ref($close, 1) - 1, (Mean(If($close > Ref($close, 1), $close - Ref($close, 1), 0), 14) / (Mean(Abs($close - Ref($close, 1)), 14) + 0.0001)), 10))", "RSI_Div_Cond"),
    
    # === MINED FACTORS (2026-01-17) ===
    # Alpha_Gen_8: Inverse Log Open + High Volatility (IC=-0.053) [Flipped]
    ("-1 * (Sum(-1 * (Log($open)), 5) + Std($high, 5))", "Alpha_Gen_8"),
]

def get_production_factors():
    """Returns (expressions, names) tuple for production factors."""
    return list(zip(*PRODUCTION_FACTORS))
