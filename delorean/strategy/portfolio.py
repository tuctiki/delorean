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
                          vol_feature: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate target weights (Risk Parity if vol available, else Equal Weight).
        """
        weights = {}
        if not target_stocks:
            return weights
            
        if vol_feature is not None:
             # Get volatility for these stocks on this date
             inv_vols = {}
             sum_inv_vol = 0.0
             
             for code in target_stocks:
                 try:
                     vol = vol_feature.loc[(current_date, code)]
                     if pd.isna(vol) or vol <= 0: vol = 1.0 
                 except:
                     vol = 1.0 
                     
                 inv_vol = 1.0 / vol
                 inv_vols[code] = inv_vol
                 sum_inv_vol += inv_vol
                 
             if sum_inv_vol > 0:
                 for code in target_stocks:
                     weights[code] = (inv_vols[code] / sum_inv_vol) * self.risk_degree
             else:
                 # Fallback to EW
                 ew = (1.0 / len(target_stocks)) * self.risk_degree
                 for code in target_stocks:
                     weights[code] = ew
        else:
             # Equal Weight
             ew = (1.0 / len(target_stocks)) * self.risk_degree
             for code in target_stocks:
                 weights[code] = ew
                 
        return weights
