
from typing import List, Any
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import CSZScoreNorm, DropnaLabel
from qlib.contrib.data.handler import Alpha158
from delorean.alphas.factors import get_production_factors

class ETFDataHandler(DataHandlerLP):
    """
    Custom DataHandler for ETF data.
    
    Inherits from Qlib's DataHandlerLP to manage data loading, processing, and feature generation
    standardized for the ETF strategy.
    
    Uses factors from delorean.alphas.factors.
    """
    @staticmethod
    def get_custom_factors():
        """
        Returns a tuple of (expressions, names) for the custom ETF factors.
        Proxies to the centralized alpha registry.
        """
        return get_production_factors()

    def __init__(self, instruments: List[str], start_time: str, end_time: str, label_horizon: int = 1, **kwargs: Any):
        """
        Initialize the ETFDataHandler.

        Args:
            instruments (List[str]): List of ETF codes (e.g., ["510300.SH"]).
            start_time (str): Start date string (YYYY-MM-DD).
            end_time (str): End date string (YYYY-MM-DD).
            label_horizon (int): Number of days for forward return label (default: 1).
            **kwargs: Additional arguments passed to DataHandlerLP.
        """
        custom_exprs, _ = self.get_custom_factors()
        
        data_loader_config = {
            "feature": custom_exprs, # Use the centralized definitions
            "label": [
                f"Ref($close, -{label_horizon}) / $close - 1"  # Dynamic Horizon Return
            ],
        }
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": data_loader_config,
                "freq": "day",
            }
        }
        processors = [
            DropnaLabel(),
            # Normalize features cross-sectionally to make them comparable across ETFs
            CSZScoreNorm(fields_group="feature"),
        ]
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            learn_processors=processors,
            **kwargs
        )

class ETFAlpha158DataHandler(Alpha158):
    """
    Custom Alpha158 DataHandler for ETF data.
    Overrides label to match the strategy's target.
    """
    def __init__(self, label_horizon: int = 1, **kwargs):
        self.label_horizon = label_horizon
        super().__init__(**kwargs)

    def get_label_config(self):
        return [f"Ref($close, -{self.label_horizon}) / $close - 1"]

class ETFHybridDataHandler(Alpha158):
    """
    Hybrid DataHandler: Alpha158 + Custom Factors.
    """
    def __init__(self, label_horizon: int = 1, **kwargs):
        self.label_horizon = label_horizon
        super().__init__(**kwargs)

    def get_label_config(self):
        return [f"Ref($close, -{self.label_horizon}) / $close - 1"]
        
    def get_feature_config(self):
        # Get Alpha158 features (Tuple of (expressions_list, names_list))
        conf = super().get_feature_config()
        alpha158_exprs, alpha158_names = conf
        
        # Get Custom Factors from ETFDataHandler (Registry)
        custom_exprs, custom_names = ETFDataHandler.get_custom_factors()
        
        # Merge
        return (alpha158_exprs + custom_exprs, alpha158_names + custom_names)
