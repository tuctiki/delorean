# Qlib ETF Data Analysis Project

This project focuses on utilizing [Qlib](https://github.com/microsoft/qlib), an AI-driven quantitative investment platform, for the analysis, preprocessing, and trading signal generation of Chinese ETF data.

## Project Structure

The project has been refactored into a modular structure:

-   **`delorean/`**: Core Python package containing logic for data handling, modeling, backtesting, and analysis.
    -   `data.py`: Data loading and feature engineering (Custom, Alpha158, Hybrid).
    -   `model.py`: LightGBM model training and inference.
    -   `backtest.py`: Strategy logic and backtest engine.
    -   `config.py`: Centralized configuration.
-   **`scripts/`**: Executable scripts for various tasks.
    -   `download_etf_data_to_csv.py`: Fetches data from AkShare.
    -   `run_etf_analysis.py`: Runs end-to-end training and backtesting analysis.
    -   `run_live_trading.py`: Generates daily trading signals.
    -   `run_daily_task.py`: Orchestrator for the daily update workflow.
-   **`server/`**: FastAPI backend for the web dashboard.
-   **`frontend/`**: Next.js web application for monitoring and visualization.

## Setup

1.  **Environment**: Ensure you have the `quant` conda environment set up (see `GEMINI.md`).
2.  **Dependencies**: Install Python dependencies.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure `akshare`, `qlib`, `fastapi`, `uvicorn`, `pandas`, `lightgbm` are installed)*

## Usage

### 1. Web Dashboard (Recommended)

The easiest way to interact with the system is via the Web Dashboard.

**Start the Backend:**
```bash
export PYTHONPATH=$PYTHONPATH:.
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

**Start the Frontend:**
```bash
cd frontend
npm run dev
```
Access the dashboard at `http://localhost:3000`.

### 2. Manual Daily Update

To run the full daily workflow (Data Download -> Qlib Dump -> Signal Generation) manually:

```bash
export PYTHONPATH=$PYTHONPATH:.
python scripts/run_daily_task.py
```

### 3. Training & Backtesting Analysis

To run research experiments (e.g., Stress Test 2015-2025):

```bash
export PYTHONPATH=$PYTHONPATH:.
python scripts/run_etf_analysis.py \
  --topk 5 --dynamic_exposure --seed 42 \
  --smooth_window 20 --buffer 2 --label_horizon 5 \
  --start_time 2015-01-01 \
  --end_time 2025-12-31 \
  --train_end_time 2021-12-31 \
  --test_start_time 2022-01-01
```

**Options:**
-   `--label_horizon <int>`: Forecast horizon in days (default: 5).
-   `--smooth_window <int>`: EWMA smoothing for signals (default: 20).
-   `--dynamic_exposure`: Enable Trend-based Dynamic Exposure (Manage Beta).
-   `--risk_parity`: Enable Volatility Targeting (Optional, 1/Vol weighting).
-   `--start_time / --end_time`: Data range overrides.
-   `--train_end_time / --test_start_time`: Split configuration.

## Implementation Details

### Factor Models
-   **Custom Strategy**: Momentum + Volatility + Reversal factors. (Best risk-adjusted return).
-   **Optimization**: Uses **Cross-Sectional Z-Score** Feature Neutralization and **5-Day Forward Return** Label.

### Position Control
The strategy now supports advanced position sizing:
-   **Equal Weight (Default)**: Allocates capital equally among Top 5 holdings.
-   **Dynamic Market Exposure**: Adjusts equity exposure (0% - 99%) based on the Benchmark Trend Strength (Close vs MA60) with **Hysteresis** to prevent whipsaws.

### Experiment Tracking
All runs are logged to `mlruns/`, viewable via the Dashboard.
-   **Experiment Details Page**: Includes Cumulative Return and Excess Return plots.

### Performance (Stress Test: 2022-Present)
| Metric | Result (Train 2015-2021) |
| :--- | :--- |
| **Annualized Return** | **11.15%** |
| **Sharpe Ratio** | **0.85** |
| **Max Drawdown** | **-11.65%** |
| **Test Period** | 2022-01-01 to 2025-12-31 |

*Note: This period covers the significant bear market of 2022-2024, demonstrating the strategy's defensive robustness.*

## Frontend Features
-   **Experiment Visuals**: Detailed performance plots and generation timestamps.
-   **Bear Market Alert**: Visual warning when Benchmark Close < MA60.

## Live Trading & Data Periods

### 1. Live Trading System (`scripts/run_live_trading.py`)
The live trading engine receives constant updates to ensure the highest fidelity signals. It now employs a **Two-Pass Training** strategy:

| Phase | Purpose | Training Period | Testing / Prediction Period |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Validation** (Metrics Check) | Start (`2015`) $\to$ **60 Days Ago** | **60 Days Ago** $\to$ **Today** |
| **Phase 2** | **Production** (Signal Gen) | Start (`2015`) $\to$ **Yesterday** | **Today** (Generation) |

- **Phase 1 Validation**: Calculates honest "Out-of-Sample" IC and Sharpe metrics by holding out the last 60 days. This prevents overconfidence from "in-sample" validation.
- **Phase 2 Production**: Retrains the model on *all* available history to generate the optimal signal for tomorrow.

### 2. Data Period Breakdown
Understanding the data split is crucial for replicating results:

- **Default Configuration**: `2015-01-01` to `2099-12-31`.
- **Backtesting (Research)**:
  - **Train**: `2015-01-01` to `2022-12-31`
  - **Test**: `2023-01-01` to `Present` (Out-of-Sample Stress Test)
- **Live Trading**:
  - **Train**: `2015-01-01` to `Yesterday` (Rolling Window)
  - **Predict**: `Today/Tomorrow`

## Daily Operation Guide

To operate the system in a production-like environment:

### A. Start the Dashboard (Monitoring)
1. **Backend** (API & Task Runner):
   ```bash
   python server/main.py
   # Runs on http://localhost:8000
   ```
2. **Frontend** (UI):
   ```bash
   cd frontend
   npm run dev
   # Runs on http://localhost:3000
   ```

### B. Run Daily Update (Action)
This is the **Master Command** to update data and get new signals. Run this every evening after market close (e.g., 6:00 PM).
```bash
python scripts/run_daily_task.py
```
*What this does:*
1. **Downloads** latest market data from AkShare.
2. **Updates** the Qlib binary database.
3. **Executes** the Live Trading Engine (`run_live_trading.py`) to generate `daily_recommendations.json`.