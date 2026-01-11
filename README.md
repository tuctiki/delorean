# Qlib ETF Data Analysis Project

This project focuses on utilizing [Qlib](https://github.com/microsoft/qlib), an AI-driven quantitative investment platform, for the analysis and preprocessing of Chinese ETF data. The goal is to set up an environment where ETF historical data can be efficiently acquired, transformed into Qlib's native binary format, and then preprocessed for quantitative research, model training, and backtesting.

## Project Setup

The project relies on a specific Conda environment (`quant`) and leverages `direnv` for automatic environment activation. Refer to the detailed setup instructions in `GEMINI.md` for installing `direnv`, creating the `quant` environment, and setting up Qlib.

## Data Acquisition and Preprocessing Workflow

### 1. Download ETF Historical Data from AkShare

Fetch daily historical data for a predefined list of Chinese ETFs (defined in `constants.py`) using AkShare.

-   **Script**: `download_etf_data_to_csv.py`
-   **Output**: CSV files in `~/.qlib/csv_data/akshare_etf_data/`.

```bash
python download_etf_data_to_csv.py
```

### 2. Convert Data to Qlib Binary Format

Transform CSV data into Qlib's efficient binary format.

-   **Tool**: `vendors/qlib/scripts/dump_bin.py`

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

The project uses a modular architecture. The main entry point `run_etf_analysis.py` orchestrates the pipeline:

-   **`data.py`**: Handles data loading and feature engineering. Supports Custom, Alpha158, and Hybrid modes.
-   **`feature_selection.py`**: Implements correlation-based feature filtering.
-   **`model.py`**: Manages LightGBM model training (Stage 1 Importance -> Stage 2 Refinement).
-   **`experiment_manager.py`**: Logs experiments and ensures reproducibility.

**Basic Usage (Original Custom Strategy):**
```bash
python run_etf_analysis.py --topk 4
```

**Advanced Usage:**
-   **Alpha158 Factors**: `python run_etf_analysis.py --use_alpha158 --topk 4`
-   **Hybrid Model**: `python run_etf_analysis.py --use_hybrid --topk 4`

## Factor Models & Modes

The project supports three distinct factor modeling approaches:

1.  **Custom Strategy (Default/Baseline)**:
    -   Uses a curated set of low-dimensional factors (Momentum, Volatility, Reversal).
    -   **Pros**: Robust, interpretable, historically best performance on this ETF set.
    
2.  **Alpha158 (Qlib Embedded)**:
    -   Uses Qlib's standard 158 factors (price/volume patterns).
    -   **Pros**: comprehensive feature coverage.
    -   **Optimization**: Includes an auto-selection step (Top 20 features) and stricter regularization to prevent overfitting.

3.  **Hybrid Model**:
    -   Combines Custom Factors + Alpha158.
    -   Uses **Correlation Filtering** `(threshold=0.95)` to remove redundant signals.

## Performance Comparison (Latest Verification)

| Metric | Original Custom Strategy | Optimized Alpha158 (Top 20) | Hybrid (Custom + Alpha158) |
| :--- | :--- | :--- | :--- |
| **Annualized Return** | **8.56%** | 7.67% | 3.96% |
| **Max Drawdown** | **-12.28%** | -15.03% | -14.19% |
| **Sharpe Ratio** | **0.77** | 0.57 | 0.35 |

> [!NOTE]
> The **Original Custom Strategy** remains the recommended baseline due to its superior risk-adjusted returns (Sharpe 0.77).

## Experiment Tracking

Experiments are automatically logged using Qlib's Recorder. 
-   **Artifacts**: Saved in `mlruns/<exp_id>/<run_id>/`.
-   **Config**: A full configuration dump is saved as `experiment_config.yaml` for every run.
-   **Reports**: Backtest reports and analysis plots are saved as artifacts.