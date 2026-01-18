from typing import List, Dict, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger("delorean.strategy.portfolio")

# Asymmetric Volatility Settings
BULL_TARGET_VOL = 0.30  # Relaxed in bull markets (allow full gains)
BEAR_TARGET_VOL = 0.06  # Aggressive in bear markets (protect capital)

# Hysteresis Thresholds (to reduce turnover from frequent regime switching)
# Set to 1.0 to disable hysteresis (original Phase 9 behavior)
BULL_THRESHOLD = 1.0   # Ratio must exceed this to trigger "Bull" mode
BEAR_THRESHOLD = 1.0   # Ratio must fall below this to trigger "Bear" mode

class PortfolioOptimizer:
    """
    Handles portfolio construction and weighting allocation.
    """
    def __init__(self, topk: int = 5, risk_degree: float = 0.95):
        self.topk = topk
        self.risk_degree = risk_degree
        self._last_regime = "neutral"  # Track last regime for hysteresis
        
    def calculate_weights(self, target_stocks: List[str], current_date: pd.Timestamp, 
                          vol_feature: Optional[pd.Series] = None, target_vol: Optional[float] = None,
                          regime_ratio: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate target weights (Risk Parity if vol available, else Equal Weight).
        
        Args:
            regime_ratio: Market regime indicator (Price/MA60). >1.0 = Bull, <1.0 = Bear.
                          If provided, target_vol is dynamically adjusted with hysteresis.
        """
        weights = {}
        if not target_stocks:
            return weights
        
        # === Asymmetric Target Vol Logic with Hysteresis ===
        effective_target_vol = target_vol
        if regime_ratio is not None and target_vol is not None:
            # Hysteresis Logic: Only switch regime if threshold is exceeded
            if regime_ratio >= BULL_THRESHOLD:
                self._last_regime = "bull"
            elif regime_ratio <= BEAR_THRESHOLD:
                self._last_regime = "bear"
            # else: keep _last_regime unchanged (neutral zone)
            
            if self._last_regime == "bull":
                effective_target_vol = max(target_vol, BULL_TARGET_VOL)
                logger.debug(f"Bull Regime ({regime_ratio:.2f}): target_vol={effective_target_vol}")
            elif self._last_regime == "bear":
                effective_target_vol = min(target_vol, BEAR_TARGET_VOL)
                logger.debug(f"Bear Regime ({regime_ratio:.2f}): target_vol={effective_target_vol}")
            
        # 1. Base Allocation (Risk Parity or Equal Weight)
        avg_vol = 0.0
        if vol_feature is not None:
             # Get volatility for these stocks on this date
             inv_vols = {}
             sum_inv_vol = 0.0
             valid_count = 0
             
             for code in target_stocks:
                 try:
                     vol = vol_feature.loc[(current_date, code)]
                     if pd.isna(vol) or vol <= 0: vol = 0.02 # Fallback low vol 
                 except:
                     vol = 0.02 # Fallback 
                 
                 # Store for Vol Targeting
                 avg_vol += vol
                 valid_count += 1
                 
                 inv_vol = 1.0 / vol
                 inv_vols[code] = inv_vol
                 sum_inv_vol += inv_vol
                 
             if sum_inv_vol > 0:
                 for code in target_stocks:
                     weights[code] = (inv_vols[code] / sum_inv_vol) # Base weight (sum=1)
             else:
                 # Fallback to EW
                 ew = 1.0 / len(target_stocks)
                 for code in target_stocks:
                     weights[code] = ew
                     
             if valid_count > 0:
                 avg_vol /= valid_count
                 
        else:
             # Equal Weight
             ew = 1.0 / len(target_stocks)
             for code in target_stocks:
                 weights[code] = ew
             
             # Warning: Without vol feature, we can't do accurate Vol Targeting
             avg_vol = 0.0

        # 2. Target Volatility Scaling (De-leveraging)
        if effective_target_vol is not None and avg_vol > 0:
            ann_vol_est = avg_vol * (252 ** 0.5)
            
            if ann_vol_est > effective_target_vol:
                scale_factor = effective_target_vol / ann_vol_est
                for code in weights:
                    weights[code] *= scale_factor
                    
        # 3. Dynamic Trend-Based Leverage Cap
        # Continuous scaling based on Regime Ratio (Price / MA60)
        # Ratio <= 0.97 (Bear) -> Cap at 20%
        # Ratio 1.00 (Weak Bull) -> Cap at ~60%
        # Ratio >= 1.03 (Strong Bull) -> Cap at 100%
        if regime_ratio is not None:
             trend_score = max(0, regime_ratio - 0.97)
             # Slope = 16.7: (0.97, 0.0) to (1.03, 1.0)
             # S = (1.0 - 0.0) / (1.03 - 0.97) = 1.0 / 0.06 = 16.67
             dynamic_leverage = 0.0 + (trend_score * 16.67)
             dynamic_leverage = min(1.0, dynamic_leverage)
             
             # Apply the stricter of Vol Scaling or Trend Cap
             # If Vol Scaling (scale_factor) is already lower, it persists.
             # If Vol Scaling allows 95%, but Trend Cap is 20%, we clip to 20%.
             total_weight = sum(weights.values())
             if total_weight > dynamic_leverage:
                 cap_factor = dynamic_leverage / total_weight
                 for code in weights:
                     weights[code] *= cap_factor

        # Apply Risk Degree (Cash buffer)
        for code in weights:
            weights[code] *= self.risk_degree

        return weights

