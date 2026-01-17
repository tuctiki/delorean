
from typing import Union, Optional
from qlib.data.dataset import DatasetH
from delorean.conf import ETF_LIST, START_TIME, END_TIME, TRAIN_END_TIME, TEST_START_TIME
from .handlers import ETFDataHandler, ETFAlpha158DataHandler, ETFHybridDataHandler

class ETFDataLoader:
    """
    Wrapper class to manage Qlib DatasetH creation and splitting.
    """
    def __init__(self, use_alpha158: bool = False, use_hybrid: bool = False, label_horizon: int = 1,
                 start_time: str = START_TIME, end_time: str = END_TIME):
        self.handler: Union[ETFDataHandler, ETFAlpha158DataHandler, ETFHybridDataHandler, None] = None
        self.dataset: Optional[DatasetH] = None
        self.use_alpha158 = use_alpha158
        self.use_hybrid = use_hybrid
        self.label_horizon = label_horizon
        self.start_time = start_time
        self.end_time = end_time

    def load_data(self, train_start: str = None, train_end: str = None, test_start: str = None, test_end: str = None) -> DatasetH:
        """
        Initialize the DataHandler and create the Qlib DatasetH object.
        """
        if self.use_hybrid:
            print(f"Initializing ETF Hybrid DataHandler (Alpha158 + Custom, H={self.label_horizon})...")
            self.handler = ETFHybridDataHandler(
                instruments=ETF_LIST,
                start_time=self.start_time,
                end_time=self.end_time,
                label_horizon=self.label_horizon
            )
        elif self.use_alpha158:
            print(f"Initializing ETF Alpha158 DataHandler (H={self.label_horizon})...")
            self.handler = ETFAlpha158DataHandler(
                instruments=ETF_LIST,
                start_time=self.start_time,
                end_time=self.end_time,
                label_horizon=self.label_horizon
            )
        else:
            print(f"Initializing Custom ETF DataHandler (H={self.label_horizon})...")
            self.handler = ETFDataHandler(
                instruments=ETF_LIST,
                start_time=self.start_time,
                end_time=self.end_time,
                label_horizon=self.label_horizon
            )
        
        # Define Segments
        # Use passed arguments if available, otherwise fallback to Config constants
        
        _train_start = train_start if train_start else self.start_time
        _train_end = train_end if train_end else TRAIN_END_TIME
        _test_start = test_start if test_start else TEST_START_TIME
        _test_end = test_end if test_end else self.end_time
        
        segments = {
            "train": (_train_start, _train_end),
            "test": (_test_start, _test_end),
        }
        
        print(f"Dataset Segments: {segments}")
        
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
