from typing import List, Dict, Any, Union, Optional
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import CSZScoreNorm, DropnaLabel
from constants import ETF_LIST, START_TIME, END_TIME, TRAIN_END_TIME, TEST_START_TIME

class ETFDataHandler(DataHandlerLP):
    """
    Custom DataHandler for ETF data.
    
    Inherits from Qlib's DataHandlerLP to manage data loading, processing, and feature generation
    standardized for the ETF strategy.
    """
    def __init__(self, instruments: List[str], start_time: str, end_time: str, **kwargs: Any):
        """
        Initialize the ETFDataHandler.

        Args:
            instruments (List[str]): List of ETF codes (e.g., ["510300.SH"]).
            start_time (str): Start date string (YYYY-MM-DD).
            end_time (str): End date string (YYYY-MM-DD).
            **kwargs: Additional arguments passed to DataHandlerLP.
        """
        data_loader_config = {
            "feature": [
                "Log(Mean($volume * $close, 20))",                  # Market Cap / Liquidity Proxy
                "$close / Ref($close, 60) - 1",                     # Medium-term Momentum (MOM60)
                "$close / Ref($close, 120) - 1",                    # Long-term Momentum (MOM120)
                "($close / Ref($close, 5) - 1) * -1",               # Short-term Reversal (REV5)
                "Std($close / Ref($close, 1) - 1, 20)",             # 20-day Volatility (VOL20)
                "Std($close / Ref($close, 1) - 1, 60)",             # 60-day Volatility (VOL60)
            ],
            "label": [
                "Ref($close, -1) / $close - 1"  # Next Day Return
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

class ETFDataLoader:
    """
    Wrapper class to manage Qlib DatasetH creation and splitting.
    """
    def __init__(self):
        self.handler: Optional[ETFDataHandler] = None
        self.dataset: Optional[DatasetH] = None

    def load_data(self) -> DatasetH:
        """
        Initialize the DataHandler and create the Qlib DatasetH object.

        Returns:
            DatasetH: The configured Qlib dataset object ready for training/inference.
        """
        print("Initializing ETF DataHandler...")
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
