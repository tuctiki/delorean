from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import CSZScoreNorm, DropnaLabel
from constants import ETF_LIST, START_TIME, END_TIME, TRAIN_END_TIME, TEST_START_TIME

class ETFDataHandler(DataHandlerLP):
    def __init__(self, instruments, start_time, end_time, **kwargs):
        data_loader_config = {
            "feature": [
                "Log(Mean($volume * $close, 20))",                  # Market Cap / Liquidity Proxy
                "$close / Ref($close, 20) - 1",                 # 20-day Momentum
                "$close / Ref($close, 60) - 1",                 # 60-day Momentum
                "$close / Ref($close, 120) - 1",               # 120-day Momentum
                "($close / Ref($close, 5) - 1) * -1",                # 5-day Reversal
                "Std($close / Ref($close, 1) - 1, 20)",         # 20-day Volatility
                "Std($close / Ref($close, 1) - 1, 60)",         # 60-day Volatility
                "$volume / Mean($volume, 20)",                       # Short-term Volume Ratio
                "$volume / Mean($volume, 60)",                       # Medium-term Volume Ratio
                "Skew($close / Ref($close, 1) - 1, 20)",       # 20-day Skewness
                # New Optimized Features
                "(EMA($close, 12) - EMA($close, 26)) / $close",      # MACD Trend (Normalized)
                "100 * EMA(($close > Ref($close, 1)) * ($close - Ref($close, 1)), 14) / EMA(Abs($close - Ref($close, 1)), 14)", # RSI 14 (Approx)
                "($close / Ref($close, 20) - 1) / Std($close / Ref($close, 1) - 1, 20)", # Vol-Adjusted Momentum
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
            CSZScoreNorm(fields_group="feature"),
        ]
        super().__init__(
            data_loader=data_loader,
            learn_processors=processors,
            **kwargs
        )

class ETFDataLoader:
    def __init__(self):
        self.handler = None
        self.dataset = None

    def load_data(self):
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

    def _print_data_stats(self):
        train_features = self.dataset.prepare("train", col_set="feature")
        print(f"Training set shape: {train_features.shape}")
        print(f"Features: {list(train_features.columns)}")
        print("\nTraining set correlation:\n", train_features.corr().round(2))
