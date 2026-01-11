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

To run research experiments:

```bash
export PYTHONPATH=$PYTHONPATH:.
python scripts/run_etf_analysis.py --topk 4 --risk_parity --dynamic_exposure --seed 42
```

**Options:**
-   `--risk_parity`: Enable Volatility Targeting (1/Vol weighting).
-   `--dynamic_exposure`: Enable Trend-based Dynamic Exposure (Manage Beta).
-   `--seed <int>`: Fix random seed for reproducibility (default: 42).
-   `--use_alpha158`: Use Qlib's Alpha158 factors.
-   `--use_hybrid`: Use Hybrid factors.

## Implementation Details

### Factor Models
-   **Custom Strategy**: Momentum + Volatility + Reversal factors. (Best risk-adjusted return).
-   **Alpha158**: Standard Qlib 158 factors.
-   **Hybrid**: Combination with correlation filtering.

### Position Control (New)
The strategy now supports advanced position sizing:
-   **Volatility Targeting (Risk Parity)**: Allocates capital inversely proportional to risk (`1/VOL20`). Includes **Buffer Rebalancing** to minimize noise trading.
-   **Dynamic Market Exposure**: Adjusts equity exposure (0% - 99%) based on the Benchmark Trend Strength (Close vs MA60) with **Hysteresis** to prevent whipsaws.

### Experiment Tracking
All runs are logged to `mlruns/`, viewable via the Dashboard or standard MLflow tools.

### Performance (Verification: 2023-Present)
| Metric | Original Custom Strategy | **Final Optimized (Top 5 + 5-Day Horizon)** |
| :--- | :--- | :--- |
| **Annualized Return** | 8.56% | **12.64%** |
| **Sharpe Ratio** | 0.77 | **1.01** |
| **Annualized Turnover** | ~400% | **1158%** |

*Note: The strategy now targets a **5-Day Forward Return** (Long-Term Alpha) with Top-K (5) diversification and Z-Score feature neutralization.*

## Frontend Features
-   **Bear Market Alert**: Visual warning when Benchmark Close < MA60.
-   **Volatility Scores**: Real-time volatility monitoring for each ETF.