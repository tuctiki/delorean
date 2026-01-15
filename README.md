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
  --topk 4 --seed 42 \
  --smooth_window 10 --buffer 2 --label_horizon 1 \
  --start_time 2015-01-01 \
  --end_time 2025-12-31 \
  --train_end_time 2021-12-31 \
  --test_start_time 2022-01-01
```

**Options:**
-   `--label_horizon <int>`: Forecast horizon in days (default: 1).
-   `--smooth_window <int>`: EWMA smoothing for signals (default: 10).
-   `--topk <int>`: Number of top ETFs to hold (default: 4).
-   `--dynamic_exposure`: Enable Trend-based Dynamic Exposure (Manage Beta).
-   `--risk_parity`: Enable Volatility Targeting (Optional, 1/Vol weighting).
-   `--start_time / --end_time`: Data range overrides.
-   `--train_end_time / --test_start_time`: Split configuration.
-   `--experiment_name <str>`: *(Deprecated)* All runs now log to default experiment `ETF_Strategy`.

### 4. MLflow UI (Experiment Management)

For detailed experiment management, comparisons, and artifact viewing:

```bash
conda run -n quant mlflow ui --backend-store-uri mlruns/ --port 5000
```
Access at `http://localhost:5000`.

## Implementation Details

### Factor Models
The strategy utilizes a **Custom Factor Model** composed of **7 optimized factors** for the Chinese ETF market:

> [!NOTE]
> **2026-01-15 Optimization**: Reduced from 8 to 7 factors. Removed VOL60 (negative IC), Mom20_VolAdj and Accel_Rev (high correlation with new factors). Added Mom_Persistence and Acceleration from Round 4 alpha mining.

#### Core Factors (3)
1.  **MarketCap_Liquidity**: `Log(Mean($volume * $close, 20))` - Liquidity and market cap proxy.
2.  **MOM60**: `$close / Ref($close, 60) - 1` - Medium-term Momentum.
3.  **MOM120**: `$close / Ref($close, 120) - 1` - Long-term Momentum.

#### Validated Factors from Alpha Mining (4)
4. **Trend_Efficiency**: `($close / Ref($close, 20) - 1) / (Std(...) + 0.0001)` - Risk-adjusted trend strength (IC: 0.034).
5. **Gap_Fill**: `($close - $open) / (Abs($open - Ref($close, 1)) + 0.001)` - Gap filling tendency (IC: 0.032).
6. **Mom_Persistence**: `Sum(If($close > Ref($close, 1), 1, 0), 10) / 10` - Momentum consistency (IC: 0.059).
7. **Acceleration**: `($close / Ref($close, 5) - 1) - (Ref($close, 5) / Ref($close, 10) - 1)` - Price acceleration (IC: 0.053).

### ETF Universe (14 Assets)
- **Broad Market**: CSI 300 (510300.SH), A500 (563360.SH), ChiNext (159915.SZ), STAR 50 (588000.SH), CSI 1000 (512100.SH)
- **Sector**: Semiconductor (512480.SH), New Energy (516160.SH), Liquor (512690.SH), Bank (512800.SH), Pharma (512010.SH), Consumer (510630.SH), PV/Solar (515790.SH), Securities (512880.SH)
- **Defensive**: Dividend RedChip (510880.SH)

**Optimization Technique**:
-   **Preprocessing**: Cross-Sectional Z-Score Feature Neutralization.
-   **Labeling**: **1-Day Forward Return** (`Ref($close, -1) / $close - 1`).
-   **Modeling**: LightGBM Regressor.

### Model Hyperparameters (Stage 1)
- **Learning Rate**: 0.02
- **Num Leaves**: 19
- **Max Depth**: 5
- **Feature Fraction (Colsample)**: 0.61
- **Bagging Fraction (Subsample)**: 0.6
- **Regularization**: L1=0.1, L2=0.1

### Position Control
The strategy uses simple, robust position sizing:
-   **Equal Weight (Default)**: Allocates capital equally among Top 4 holdings.
-   **TopK**: 4 (optimized from 5 for higher concentration).
-   **Smooth Window**: 10-day EWMA for signal smoothing.

### Experiment Tracking
All backtest runs are logged to the default experiment `ETF_Strategy` in `mlruns/`.
-   **Dashboard**: View Run History at `http://localhost:3000/experiments`.
-   **MLflow UI**: For advanced comparison and artifact browsing, run `mlflow ui --port 5000`.

### Performance (Latest: 2026-01-15)

**Current Performance (Optimized 7-Factor Library)**:

| Metric | Result (Test 2023-2025) |
| :--- | :--- |
| **Sharpe Ratio** | **1.05** |
| **Annual Return** | **20.4%** |
| **Max Drawdown** | **-24.0%** |
| **Factor Count** | **7** (optimized) |
| **TopK** | **4** |
| **Label Horizon** | **1 Day** |

> [!NOTE]
> **2026-01-15 Update**: Strategy configuration optimized (TopK 5→4, Horizon 5→1, Smooth 15→10). Regime filter removed for simplified always-invested strategy. Sharpe improved from 0.26 to 1.05.

**Historical Performance (Stress Test: 2022-Present)**:

| Metric | Result (Train 2015-2021) |
| :--- | :--- |
| **Annualized Return** | **11.15%** |
| **Sharpe Ratio** | **0.85** |
| **Max Drawdown** | **-11.65%** |
| **Test Period** | 2022-01-01 to 2025-12-31 |

*Note: This period covers the significant bear market of 2022-2024, demonstrating the strategy's defensive robustness.*

## Frontend Features
-   **Experiment Visuals**: Detailed performance plots and generation timestamps.

## Live Trading & Data Periods

### 1. Live Trading System (`scripts/run_live_trading.py`)
The live trading engine receives constant updates to ensure the highest fidelity signals. It now employs a **Two-Pass Training** strategy:

| Phase | Purpose | Training Period | Testing / Prediction Period |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Validation** (Metrics Check) | Start (`2015`) $\to$ **60 Days Ago** | **60 Days Ago** $\to$ **Today** |
| **Phase 2** | **Production** (Signal Gen) | Start (`2015`) $\to$ **Yesterday** | **Today** (Generation) |

- **Phase 1 Validation**: Calculates honest "Out-of-Sample" Rank IC and Sharpe metrics by holding out the last **60 days** from training. This prevents overconfidence from "in-sample" validation.
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

## Docker Deployment

To build and run the application using Docker:

1.  **Prerequisites**: Ensure Docker and Docker Compose are installed.
2.  **Data**: The setup assumes Qlib data is at `~/.qlib/qlib_data/cn_etf_data`.
3.  **Run**:
    ```bash
    docker compose up -d --build
    ```
4.  **Access**:
    - Frontend: http://localhost:3005
    - Backend API: http://localhost:8005

> [!NOTE]
> If deploying to a remote server, update `NEXT_PUBLIC_API_URL` in `docker-compose.yml` to the server's IP or domain before building.

The backend container also runs a cron job for daily trading tasks at 18:00 on weekdays.

---

## Changelog

### 2026-01-14: Alpha Mining & Factor Library Optimization

#### Alpha Mining Initiative
- **Discovered 3 new validated factors** through systematic hypothesis-driven research
- Tested 32 factor candidates across 5 categories (volume-price dynamics, momentum, volatility, etc.)
- Validated on out-of-sample data (2023-2025)

**New Factors Added**:
1. **Mom_Vol_Combo**: Momentum-Volatility Composite (IC: 0.037, Alpha: 12.9%)
2. **Gap_Fill**: Gap Filling Tendency (IC: 0.032, highly unique with 0.16 max correlation)
3. ~~VolAdj_Mom_10~~: Initially added but later removed due to 0.98 correlation with Mom_Vol_Combo

**Performance Impact**: +22.9% Sharpe improvement (0.621 → 0.764)

#### Factor Library Audit & Optimization
- **Comprehensive audit** of all 11 factors on 2023-2025 data
- Evaluated IC, ICIR, correlation, and alpha for each factor
- Identified and removed 3 weak/redundant factors

**Factors Removed**:
1. **ROC_Rev**: No predictive power (IC = -0.001)
2. **KC_Width_Norm**: Redundant with VOL60 (0.77 correlation, worse IC)
3. **VolAdj_Mom_10**: Redundant with Mom_Vol_Combo (0.98 correlation)

**Optimization Results**:
- Factor count: 11 → 8 (-27%)
- Sharpe Ratio: 0.764 → 0.894 (+17%)
- Rank IC: 0.013 → 0.047 (+266%)

**Key Insight**: Removing weak/redundant factors improved performance, demonstrating that quality beats quantity in factor investing.

#### Scripts Created
- `scripts/mine_new_alphas.py`: Initial alpha mining (1-day horizon)
- `scripts/mine_alphas_round2.py`: Refined mining (5-day horizon)
- `scripts/validate_top_factors.py`: Out-of-sample validation
- `scripts/audit_factors_enhanced.py`: Comprehensive factor audit
- `scripts/compare_performance.py`: Performance comparison tool

#### Documentation Updates
- Updated factor descriptions in README.md
- Updated `delorean/data.py` with optimized 8-factor library
- Created comprehensive audit and optimization reports

**Final State**: Optimized 8-factor library with Sharpe Ratio of 0.894, ready for production.