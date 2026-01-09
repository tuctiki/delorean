# etf_analyzer.py

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import DropnaLabel, CSZScoreNorm, DropnaProcessor
import pandas as pd
import traceback

from constants import ETF_LIST

def setup_qlib(provider_uri="~/.qlib/qlib_data/cn_etf_data", region=REG_CN):
    """Initializes Qlib."""
    try:
        qlib.init(provider_uri=provider_uri, region=region)
        print(f"Qlib initialized successfully, using provider: {provider_uri}")
        return True
    except Exception as e:
        print(f"Qlib initialization failed: {e}")
        print("Please ensure you have run the data download and conversion scripts as per the README.")
        return False

class ETFDataHandler(DataHandlerLP):
    def __init__(self, instruments, start_time, end_time, **kwargs):
        data_loader_config = {
            "feature": [
                "$close", "$volume", "Ref($close, 1)", "Mean($volume, 20)",
                "Std($close / Ref($close, 1) - 1, 20)",
            ],
            "label": ["Ref($close, -1) / $close - 1"],
        }
        data_loader = {"class": "QlibDataLoader", "kwargs": {"config": data_loader_config}}
        processors = [DropnaProcessor(fields_group="feature"), DropnaLabel(), CSZScoreNorm()]

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            **kwargs
        )
        self.processors = processors

def get_etf_data_example(etf_ticker, start_time, end_time):
    """Gets and displays a sample of historical data for a given ETF."""
    print(f"\nFetching raw data sample for {etf_ticker}...")
    try:
        features = D.features([etf_ticker], ["$close", "$volume"], start_time=start_time, end_time=end_time)
        if not features.empty:
            print(f"Recent data for {etf_ticker}:\n{features.head()}")
        else:
            print(f"No raw data found for {etf_ticker}.")
        return features
    except Exception as e:
        print(f"Error fetching raw data for {etf_ticker}: {e}")
        return pd.DataFrame()

def main():
    """Main function to demonstrate data preprocessing and access."""
    pd.set_option('display.max_rows', 100)

    # Time range for analysis
    START_TIME = "2015-01-01"
    END_TIME = "2026-01-09"

    # --- 1. Initialize Qlib ---
    if not setup_qlib():
        return

    # --- 2. Demonstrate Raw Data Access ---
    # Fetch data for the first ETF in the list as an example
    if ETF_LIST:
        get_etf_data_example(ETF_LIST[0], START_TIME, END_TIME)

    # --- 3. Preprocess Data with DataHandler ---
    print("\n--- Starting Data Preprocessing ---")
    print("Creating DataHandler instance...")
    handler = ETFDataHandler(
        instruments=ETF_LIST,
        start_time=START_TIME,
        end_time=END_TIME,
    )
    print("DataHandler instance created.")

    # Define data segments for training, validation, and testing
    segments = {
        "train": ("2015-01-01", "2022-12-31"),
        "valid": ("2023-01-01", "2024-12-31"),
        "test": ("2025-01-01", END_TIME),
    }

    print("Creating DatasetH instance for handling segmented data...")
    dataset = DatasetH(handler=handler, segments=segments)
    print("DatasetH instance created.")

    # --- 4. Access and Display Preprocessed Data ---
    print("\n--- Accessing Preprocessed Data ---")
    try:
        # Get and display training data
        print("\nPreparing training data...")
        train_features = dataset.prepare("train", col_set="feature")
        train_label = dataset.prepare("train", col_set="label")
        
        print("Training feature data shape:", train_features.shape)
        if not train_features.empty:
            print("Training feature sample:\n", train_features.head())

        print("\nTraining label data shape:", train_label.shape)
        if not train_label.empty:
            print("Training label sample:\n", train_label.head())

        # Get and display test data
        print("\nPreparing test data...")
        test_data = dataset.prepare("test", col_set=["feature", "label"])
        print("\nTest data shape:", test_data.shape)
        if not test_data.empty:
            print("Test data sample:\n", test_data.head())
            instruments_count = test_data.groupby(level="instrument").size()
            print("\nNumber of samples per ETF in test set:\n", instruments_count)
        else:
            print("Test set is empty. This might be expected if the 'test' date range is in the future.")

    except Exception as e:
        print(f"\nAn error occurred during data preparation: {e}")
        print(traceback.format_exc())

if __name__ == '__main__':
    main()
