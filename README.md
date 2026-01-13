# Qlib ETF Data Analysis Project

This project focuses on utilizing [Qlib](https://github.com/microsoft/qlib), an AI-driven quantitative investment platform, for the analysis, preprocessing, and trading signal generation of Chinese ETF data.

## Project Structure

The project has been refactored into a modular structure:

-   **`delorean/`**: Core Python package containing logic for data handling, modeling, backtesting, and analysis.
    -   `data.py`: Data loading and feature engineering (Custom, Alpha158, Hybrid).
    -   `model.py`: LightGBM model training and inference.
    -   `backtest.py`: Backtesting engine and strategy composition.
    -   `config.py`: Centralized configuration.
    -   `pipeline.py`: Daily task orchestration.
    -   `utils.py`: Centralized utilities (smoothing, IC calculation).
    -   **`strategy/`**: Modular strategy components.
        -   `portfolio.py`: Weight calculation (Equal Weight / Risk Parity).
        -   `execution.py`: Order generation and turnover control.
-   **`tests/`**: Unit tests for stability and regression prevention.
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
    *(Note: Ensure `akshare`, `qlib`, `fastapi`, `uvicorn`, `pandas`, `lightgbm`, `pytest` are installed)*

## Testing

The project includes a `pytest` suite for unit testing key components.

```bash
conda run -n quant python -m pytest tests/
```

Key tests cover:
- Data Handlers (`tests/test_data_handler.py`)
- Strategy Composition (`tests/test_strategy.py`)
- Utilities (`tests/test_utils.py`)

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
-   `--smooth_window <int>`: EWMA smoothing for signals (default: 15).
-   `--dynamic_exposure`: Enable Trend-based Dynamic Exposure (Manage Beta).
-   `--risk_parity`: Enable Volatility Targeting (Optional, 1/Vol weighting).
-   `--start_time / --end_time`: Data range overrides.
-   `--train_end_time / --test_start_time`: Split configuration.

## Implementation Details

### Factor Models
The strategy utilizes a **Custom Hybrid Factor Model** composed of 12 robust factors optimized for the Chinese ETF market:

1.  **MarketCap_Liquidity**: `Log(Mean($volume * $close, 20))` - Captures flow and capacity.
2.  **MOM60**: `$close / Ref($close, 60) - 1` - Medium-term Momentum.
3.  **MOM120**: `$close / Ref($close, 120) - 1` - Long-term Momentum.
4.  **REV5**: `($close / Ref($close, 5) - 1) * -1` - Short-term Mean Reversion.
5.  **VOL20**: `Std($close / Ref($close, 1) - 1, 20)` - Short-term Volatility.
6.  **VOL60**: `Std($close / Ref($close, 1) - 1, 60)` - Medium-term Volatility.
7.  **VOL120**: `Std($close / Ref($close, 1) - 1, 120)` - Long-term Volatility.
8.  **BB_Width_Norm**: Normalized Bollinger Band Width (Volatility).
9.  **KC_Width_Norm**: Normalized Keltner Channel Width (True Range Volatility).
10. **Squeeze_Ratio**: Ratio of BB Width to KC Width (Detects volatility squeeze/expansion regimes).
11. **Trend_Momentum**: `Mean($close, 10) * Slope($close, 10)` - Trend strength indicator.
12. **Vol_Breakout**: Volatility breakout proxy combining log(MAHIGH10)² with upper band.

**Optimization Technique**:
-   **Preprocessing**: Cross-Sectional Z-Score Feature Neutralization.
-   **Labeling**: **5-Day Forward Return** (`Ref($close, -5) / $close - 1`).
-   **Modeling**: LightGBM Regressor.

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
-   **Bear Market Alert**: Visual warning and "LIQUIDATE" recommendation when **Benchmark (510300.SH) Close < 60-Day Moving Average**.

## Live Trading & Data Periods

### 1. Live Trading System (`scripts/run_live_trading.py`)
The live trading engine receives constant updates to ensure the highest fidelity signals. It now employs a **Two-Pass Training** strategy:

| Phase | Purpose | Training Period | Testing / Prediction Period |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Validation** (Metrics Check) | Start (`2015`) $\to$ **60 Days Ago** | **60 Days Ago** $\to$ **Today** |
| **Phase 2** | **Production** (Signal Gen) | Start (`2015`) $\to$ **Yesterday** | **Today** (Generation) |

- **Phase 1 Validation**: Calculates honest "Out-of-Sample" Rank IC and Sharpe metrics by holding out the last **60 days** from training. This prevents overconfidence from "in-sample" validation.
- **Phase 2 Production**: Retrains the model on *all* available history to generate the optimal signal for tomorrow.
- **Regime Filter**: The system checks the Benchmark (510300.SH). If **Price < MA60**, it triggers a **Bear Market Warning** and recommends holding **CASH** (Liquidate All).

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