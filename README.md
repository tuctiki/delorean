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

This step fetches daily historical data for a predefined list of Chinese ETFs using the AkShare library. The `ETF_LIST` is defined in `constants.py`.

-   **Script**: `download_etf_data_to_csv.py`
-   **Purpose**: Fetches data for ETFs listed in `constants.py`.
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

### 3. Run Strategy Analysis (End-to-End)

The project has been refactored into a modular architecture for better maintainability and scalability. The main entry point `run_etf_analysis.py` orchestrates the entire pipeline:

-   **`data.py`**: Handles data loading and feature engineering (calculating factors like Momentum, Volatility, RSI, MACD).
-   **`model.py`**: Manages LightGBM model training and prediction.
-   **`backtest.py`**: Executes the `TopkDropoutStrategy` backtest using Qlib's engine.
-   **`analysis.py`**: Calculates metrics (Alpha, Sharpe, Drawdown) and generates visualization plots.

**To run the complete analysis:**
```bash
python run_etf_analysis.py
```

**What happens when you run this:**
1.  **Initialization**: Qlib is initialized with local provider settings.
2.  **Data Loading**: Historical ETF data is loaded, and 13 technical factors (including RSI, MACD, Volatility) are computed.
3.  **Model Training**: A LightGBM model trains on data from 2018-2020 to predict future returns.
4.  **Backtest**: The `TopkDropoutStrategy` (Top 3, Drop 1) is simulated on the test period (2021-2022).
5.  **Analysis**:
    -   Metrics are printed to the console (Annualized Return, Max Drawdown, Turnover).
    -   **Plots Generated**: `cumulative_return.png` and `excess_return.png`.
    -   **Reports Saved**: `backtest_report.pkl`, `backtest_positions.pkl`, `backtest_analysis.pkl`.

### 4. Strategy Details

-   **Model**: LightGBM (Gradient Boosting)
-   **Target**: Next day return (`Ref($close, -1) / $close - 1`)
-   **Features**: Market Cap, Momentum (20/60/120 days), Volatility, RSI, MACD, Volume Ratios.
-   **Strategy**: `TopkDropoutStrategy`
    -   **Top K**: Holds the top 3 ETFs with highest predicted scores.
    -   **N Drop**: Replaces 1 ETF per rebalancing period to ensure portfolio freshness.
    -   **Rebalancing**: Daily.