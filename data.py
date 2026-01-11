from typing import List, Dict, Any, Union, Optional
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import CSZScoreNorm, DropnaLabel
from qlib.contrib.data.handler import Alpha158
from constants import ETF_LIST, START_TIME, END_TIME, TRAIN_END_TIME, TEST_START_TIME

class ETFDataHandler(DataHandlerLP):
    """
    Custom DataHandler for ETF data.
    
    Inherits from Qlib's DataHandlerLP to manage data loading, processing, and feature generation
    standardized for the ETF strategy.
    """
    @staticmethod
    def get_custom_factors():
        """
        Returns a tuple of (expressions, names) for the custom ETF factors.
        """
        custom_exprs = [
            "Log(Mean($volume * $close, 20))",                  
            "$close / Ref($close, 60) - 1",                     
            "$close / Ref($close, 120) - 1",                    
            "($close / Ref($close, 5) - 1) * -1",               
            "Std($close / Ref($close, 1) - 1, 20)",             
            "Std($close / Ref($close, 1) - 1, 60)",             
            "Std($close / Ref($close, 1) - 1, 120)",            
        ]
        
        custom_names = [
            "MarketCap_Liquidity",
            "MOM60",
            "MOM120",
            "REV5",
            "VOL20",
            "VOL60",
            "VOL120",
        ]
        return custom_exprs, custom_names

    def __init__(self, instruments: List[str], start_time: str, end_time: str, **kwargs: Any):
        """
        Initialize the ETFDataHandler.

        Args:
            instruments (List[str]): List of ETF codes (e.g., ["510300.SH"]).
            start_time (str): Start date string (YYYY-MM-DD).
            end_time (str): End date string (YYYY-MM-DD).
            **kwargs: Additional arguments passed to DataHandlerLP.
        """
        custom_exprs, _ = self.get_custom_factors()
        
        data_loader_config = {
            "feature": custom_exprs, # Use the centralized definitions
            "label": [
                "Ref($close, -1) / $close - 1"  # Next Day Return (Restored)
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
    Overrides label to match the strategy's target (Next Day Return).
    """
    def get_label_config(self):
        return ["Ref($close, -1) / $close - 1"]

class ETFHybridDataHandler(Alpha158):
    """
    Hybrid DataHandler: Alpha158 + Custom Factors.
    """
    def get_label_config(self):
        return ["Ref($close, -1) / $close - 1"]
        
    def get_feature_config(self):
        # Get Alpha158 features (Tuple of (expressions_list, names_list))
        conf = super().get_feature_config()
        alpha158_exprs, alpha158_names = conf
        
        # Get Custom Factors from ETFDataHandler
        custom_exprs, custom_names = ETFDataHandler.get_custom_factors()
        
        # Merge
        return (alpha158_exprs + custom_exprs, alpha158_names + custom_names)

class ETFDataLoader:
    """
    Wrapper class to manage Qlib DatasetH creation and splitting.
    """
    def __init__(self, use_alpha158: bool = False, use_hybrid: bool = False):
        self.handler: Union[ETFDataHandler, ETFAlpha158DataHandler, ETFHybridDataHandler, None] = None
        self.dataset: Optional[DatasetH] = None
        self.use_alpha158 = use_alpha158
        self.use_hybrid = use_hybrid

    def load_data(self) -> DatasetH:
        """
        Initialize the DataHandler and create the Qlib DatasetH object.

        Returns:
            DatasetH: The configured Qlib dataset object ready for training/inference.
        """
        if self.use_hybrid:
            print("Initializing ETF Hybrid DataHandler (Alpha158 + Custom)...")
            self.handler = ETFHybridDataHandler(
                instruments=ETF_LIST,
                start_time=START_TIME,
                end_time=END_TIME,
            )
        elif self.use_alpha158:
            print("Initializing ETF Alpha158 DataHandler...")
            self.handler = ETFAlpha158DataHandler(
                instruments=ETF_LIST,
                start_time=START_TIME,
                end_time=END_TIME,
            )
        else:
            print("Initializing Custom ETF DataHandler...")
            self.handler = ETFDataHandler(
                instruments=ETF_LIST,
                start_time=START_TIME,
                end_time=END_TIME,
            )
        
        segments = {
            "train": (START_TIME, TRAIN_END_TIME),
            "test": (TEST_START_TIME, END_TIME),
        }
        
        print("Creating Dataset...")
        self.dataset = DatasetH(
            handler=self.handler,
            segments=segments,
        )
        self._print_data_stats()
        return self.dataset

    def _print_data_stats(self) -> None:
        """
        Print basic statistics about the loaded training data (feature shape, correlation).
        """
        train_features = self.dataset.prepare("train", col_set="feature")
        print(f"Training set shape: {train_features.shape}")
        print(f"Features: {list(train_features.columns)}")
        print("\nTraining set correlation:\n", train_features.corr().round(2))
