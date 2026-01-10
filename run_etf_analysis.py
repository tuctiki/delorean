import datetime
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
import qlib
from qlib.config import REG_CN

# Initialize Qlib
qlib.init(provider_uri='~/.qlib/qlib_data/cn_etf_data', region=REG_CN)


# Define the time range for the analysis
START_TIME = "2018-01-01"
END_TIME = "2022-12-31"
TRAIN_END_TIME = "2020-12-31"
TEST_START_TIME = "2021-01-01"

# Define the segments for training, validation (optional), and testing
segments = {
    "train": (START_TIME, TRAIN_END_TIME),
    # "valid": ("2021-01-01", "2021-12-31"), # Optional, uncomment if needed
    "test": (TEST_START_TIME, END_TIME),
}

if __name__ == "__main__":
    # Import and run the etf_analyzer.py script
    exec(open("etf_analyzer.py").read())
