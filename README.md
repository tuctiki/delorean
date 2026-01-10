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

## Strategy Details
- **Universe**: High-liquidity ETFs (HS300, Sector ETFs, Red Dividend).
- **Model**: LightGBM (Gradient Boosting).
- **Horizon**: 1-Day Forward Return.
- **Signal Smoothing**: Raw scores + 10-day EWMA (preserves magnitude signal).
- **Execution Strategy**: Top 4 ETFs, 100% reallocation check.
- **Turnover Control**: 96% Probabilistic Retention + 1 Swap Limit (n_drop=1).
- **Risk Management**: Mandatory **MA60 Regime Filter** on HS300 (Cash rule).

## Performance (2024-2025 Verification)
| Metric | Value | Comment |
| :--- | :--- | :--- |
| **Annualized Return** | **18.76%** | Beat benchmark (~9%) significantly. |
| **Sharpe Ratio** | **0.85** | High risk-adjusted return. |
| **Max Drawdown** | **-16.10%** | Controlled using MA60 Filter. |
| **Turnover** | ~450% | Moderate/Tactical. |

> [!IMPORTANT]
> For long-term survival, ALWAYS heed the `BEAR MARKET DETECTED` warning in the daily signals. The MA60 filter turned a potential -51% crash (2023) into a manageable -18% correction.