# Qlib ETF Data Analysis Project

This project focuses on utilizing [Qlib](https://github.com/microsoft/qlib), an AI-driven quantitative investment platform, for the analysis and preprocessing of Chinese ETF data. The goal is to set up an environment where ETF historical data can be efficiently acquired, transformed into Qlib's native binary format, and then preprocessed for quantitative research, model training, and backtesting.

## Project Setup

The project relies on a specific Conda environment (`quant`) and leverages `direnv` for automatic environment activation. Refer to the detailed setup instructions in `GEMINI.md` for installing `direnv`, creating the `quant` environment, and setting up Qlib.

**Key Setup Steps (summarized from `GEMINI.md`):**
1.  **Conda Environment**: Ensure the `quant` conda environment is active.
2.  **Qlib Installation**: Ensure `pyqlib 0.9.7` is installed. The Qlib source code is available under `./vendors/qlib`.

## Data Acquisition and Preprocessing Workflow

The workflow involves three main stages to prepare ETF data for Qlib applications:

### 1. Download ETF Historical Data from AkShare

This step fetches daily historical data for a predefined list of Chinese ETFs using the AkShare library. The data is saved as CSV files.

-   **Script**: `download_etf_data_to_csv.py`
-   **Purpose**: Fetches data for ETFs listed in `ETF_LIST` (defined within the script).
-   **Output**: CSV files for each ETF, stored in `~/.qlib/csv_data/akshare_etf_data/`.

**To run this step:**
```bash
python download_etf_data_to_csv.py
```

### 2. Convert Data to Qlib Binary Format

After downloading, the CSV data needs to be converted into Qlib's efficient binary format. This conversion process is crucial for optimal performance when working with Qlib.

-   **Tool**: `vendors/qlib/scripts/dump_bin.py` (provided by Qlib)
-   **Purpose**: Transforms the raw CSV data into Qlib's proprietary binary format, organizing it for efficient access.
-   **Output**: Qlib-formatted binary data (features, calendars, instruments) located in `~/.qlib/qlib_data/cn_etf_data`.

**To run this step:**
```bash
python vendors/qlib/scripts/dump_bin.py dump_all \
--data_path ~/.qlib/csv_data/akshare_etf_data \
--qlib_dir ~/.qlib/qlib_data/cn_etf_data \
--freq day \
--date_field_name date \
--symbol_field_name symbol \
--file_suffix .csv \
--exclude_fields symbol
```

### 3. Preprocess ETF Data for Qlib Applications

This script loads the Qlib binary data, performs feature engineering, and preprocesses the data into Qlib's `DatasetH` object, ready for model training or backtesting.

-   **Script**: `qlib_etf_preprocessing.py`
-   **Purpose**:
    *   Initializes Qlib to point to the `cn_etf_data` directory.
    *   Defines a custom `ETFDataHandler` to load raw data, calculate derived features (e.g., lagged prices, volume means, volatility) and a next-day return label.
    *   Applies preprocessing steps: dropping missing features, dropping rows with missing labels, and cross-sectional Z-score normalization.
    *   Splits the data into training, validation, and testing segments (`DatasetH`).
    *   Demonstrates how to access prepared features and labels from the `DatasetH`.

**To run this step:**
```bash
python qlib_etf_preprocessing.py
```

## Accessing Prepared Data

Once the data is preprocessed, it can be accessed in other Qlib-based applications. The `qlib.init` call in `qlib_etf_preprocessing.py` (and potentially `access_qlib_etf_data.py` if used for generic access) ensures that Qlib points to the correct ETF data directory (`~/.qlib/qlib_data/cn_etf_data`). You can then use `qlib.data.D.features()` or the `DatasetH` object directly to retrieve historical data for the specified ETFs.