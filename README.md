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

-   **`data.py`**: Handles data loading and feature engineering (calculating optimized factors like Volatility, Momentum).
-   **`model.py`**: Manages LightGBM model training and prediction.
-   **`backtest.py`**: Executes the `SimpleTopkStrategy` with turnover control using Qlib's engine.
-   **`analysis.py`**: Calculates metrics (Alpha, Sharpe, Drawdown) and generates visualization plots.

**To run the complete analysis:**
```bash
python run_etf_analysis.py --topk 4
```

**CLI Options:**
-   `--topk`: Number of stocks to hold (default: 3).

**What happens when you run this:**
1.  **Initialization**: Qlib is initialized with local provider settings.
2.  **Data Loading**: Historical ETF data (2015-2025) is loaded.
3.  **Feature Engineering**: Computes 6 optimized factors (Volatility, Momentum, Liquidity).
4.  **Model Training**: A LightGBM model trains on data (2015-2023) to predict future returns.
5.  **Signal Processing**: Applies **10-day EWMA Smoothing** to stabilize predictions.
6.  **Backtest**: The `SimpleTopkStrategy` (Top 4, 96% Retention) is simulated on the test period (2024-2025).
7.  **Analysis**:
    -   Prints metrics (Annualized Return, Drawdown, Turnover).
    -   Generates plots (`cumulative_return.png`).
    -   Saves reports to `artifacts/`.

### 4. Strategy Details (Aggressive Turnover Reduction)

-   **Model**: LightGBM (Gradient Boosting) with optimized hyperparameters.
-   **Target**: Next day return (`Ref($close, -1) / $close - 1`)
-   **Features**:
    -   **Predictive**: 60-day & 20-day Volatility (Positive Correlation).
    -   **Trend**: 60-day & 120-day Momentum.
    -   **Reversal**: 5-day Reversal (`REV5`).
    -   **Liquidity**: Log Market Cap (`SIZE`).
-   **Strategy**: `SimpleTopkStrategy` (Robust Version)
    -   **Top K**: Holds the top 4 ETFs.
    -   **Turnover Control**:
        -   **Probabilistic Retention**: 96% probability of **skipping** trades daily to hold positions longer.
        -   **Smoothing**: 10-day Exponential Weighted Moving Average (EWMA) to filter signal noise.
        -   **Swap Limit**: Max 1 stock replacement per trading opportunity.
    -   **Performance (2024-2025)**:
        -   **Return**: ~14%
        -   **Turnover**: < 500% (Low Cost)
        -   **Sharpe**: ~0.76