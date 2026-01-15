from typing import List, Dict, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger("delorean.strategy.portfolio")

class PortfolioOptimizer:
    """
    Handles portfolio construction and weighting allocation.
    """
    def __init__(self, topk: int = 5, risk_degree: float = 0.95):
        self.topk = topk
        self.risk_degree = risk_degree
        
    def calculate_weights(self, target_stocks: List[str], current_date: pd.Timestamp, 
                          vol_feature: Optional[pd.Series] = None, target_vol: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate target weights (Risk Parity if vol available, else Equal Weight).
        """
        weights = {}
        if not target_stocks:
            return weights
            
        # 1. Base Allocation (Risk Parity or Equal Weight)
        if vol_feature is not None:
             # Get volatility for these stocks on this date
             inv_vols = {}
             sum_inv_vol = 0.0
             avg_vol = 0.0
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
             # We assume a default vol if not provided? No, just skip scaling if no vol data
             avg_vol = 0.0

        # 2. Target Volatility Scaling (De-leveraging)
        # If Target Vol is enabled and we have vol data
        if target_vol is not None and avg_vol > 0:
            # Annualize the daily vol (assuming input vol is daily std)
            # vol_feature typicaly is Std(Returns, 20). 
            # Annualized Vol = Daily Vol * sqrt(252)
            ann_vol_est = avg_vol * (252 ** 0.5)
            
            if ann_vol_est > target_vol:
                # Scale down exposure
                scale_factor = target_vol / ann_vol_est
                # Cap at 1.0 (No leverage) -> Already implied since scale_factor < 1 if vol > target
                
                # Apply scale
                for code in weights:
                    weights[code] *= scale_factor
                    
        # Apply Risk Degree (Cash buffer)
        for code in weights:
            weights[code] *= self.risk_degree

        return weights
