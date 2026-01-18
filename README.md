# Delorean - ETF Trading Strategy

A quantitative trading strategy for Chinese ETFs using machine learning and technical factors with optimized turnover control.

## Overview

Delorean is a momentum-based ETF rotation strategy that:
- Predicts next-day returns using 7 proprietary alpha factors
- Holds top 4 ETFs with dynamic risk management
- Uses walk-forward validation for robust out-of-sample testing
- Achieves **Sharpe ratio of 0.87** with **22.5% annualized turnover**

## Key Features

### Alpha Factors (Ultra 6: 1-Day Optimized)
- **Momentum & Skew**: 
  - `Vol_Skew_20` (Anti-lottery, flipped skew)
  - `Selection_Trend` (GP-mined momentum multiplier)
- **Volume / Flow**: 
  - `Vol_Price_Div_Rev` (Mean-reversion confirmation)
  - `Smart_Flow_Rev` (Liquidity exhaustion)
- **Reversion**: 
  - `Gap_Fill_Rev` (Short-term gap reversal)
  - `Alpha_Gen_8` (GP-mined price range signal)

### ML Architecture
- **Model**: DoubleEnsemble Architecture (based on Qlib)
- **Features**: Regime-aware shuffling and sample reweighting
- **Benefit**: Significantly reduces drawdown and "sign-flip" errors in choppy markets

### Portfolio Management
- **Risk Parity Weighting**: Inverse volatility weights
- **Target Volatility**: 20% annualized with dynamic scaling
- **Asymmetric Volatility Scaling**: 
  - Bull markets: 30% target vol (risk on)
  - Bear markets: 12% target vol (capital protection)
- **Market Regime Filter**: CSI 300 Price/MA60 ratio

### Turnover Control
- **Signal Smoothing**: 3-day EMA + 10-day EWMA for noise reduction
- **Wide Buffer**: Rank hysteresis (TopK + 3 buffer zone)
- **Rebalancing Threshold**: 5% portfolio value threshold

## Performance Metrics

**Walk-Forward Validation (2022-01-01 to 2025-12-31)**:
- **Sharpe Ratio**: 0.60
- **Annual Return**: 6.4%
- **Max Drawdown**: -16.6%
- **Annualized Turnover**: 19.1x
- **Rank IC**: 0.016
- **Trading Frequency**: Daily Rebalancing

## Project Structure

```
delorean/
├── delorean/               # Core strategy package
│   ├── alphas/            # [NEW] Alpha factor registry
│   ├── conf/              # [NEW] Modular configuration
│   ├── data/              # [Refactored] Data handlers & loaders
│   ├── strategy/          # Portfolio & execution logic
│   ├── runner.py          # [NEW] Unified execution engine
│   ├── pipeline.py        # Daily task orchestrator
│   ├── model.py           # LightGBM model trainer
│   ├── backtest.py        # Backtest engine with signal smoothing
│   └── analysis.py        # Performance analytics & plots
├── scripts/
│   ├── ops/               # Operations (Backtest, Live Trading)
│   ├── research/          # Factor mining & Validation
│   └── analysis/          # Ad-hoc analysis
├── server/                # FastAPI backend
└── frontend/              # Next.js dashboard
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/tuctiki/delorean.git
cd delorean

# Create conda environment
conda create -n quant python=3.12
conda activate quant

# Install dependencies
pip install -r requirements.txt

# Install Qlib data
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_etf_data --region cn
```

### Run Backtest

```bash
# Walk-forward validation (default)
conda run -n quant python scripts/run_etf_analysis.py

# Standard backtest (faster, less robust)
conda run -n quant python scripts/run_etf_analysis.py --no_walk_forward

# Custom parameters
conda run -n quant python scripts/run_etf_analysis.py \
  --signal_halflife 3 \
  --buffer 3 \
  --target_vol 0.20
```

### Launch Dashboard

```bash
# Start backend (port 8000)
conda run -n quant python server/main.py

# Start frontend (port 3000) - separate terminal
cd frontend && npm run dev
```

Access at: http://localhost:3000

### Generate Live Signals

```bash
conda run -n quant python scripts/run_live_trading.py
# Output: daily_recommendations.json
```

## Configuration

Default parameters in `delorean/config.py`:

```python
DEFAULT_BACKTEST_PARAMS = {
    "topk": 4,                    # Number of ETFs to hold
    "buffer": 3,                  # Rank hysteresis buffer
    "signal_halflife": 3,         # Signal smoothing (days)
    "rebalance_threshold": 0.05,  # 5% rebalancing threshold
    "target_vol": 0.20,           # 20% target volatility
    "smooth_window": 10,          # Model output smoothing
}
```

## ETF Universe (38 ETFs)

**Broad Market**: CSI 300, A500, ChiNext, STAR 50, CSI 1000, SSE 50, H-Share  
**Commodities**: Gold, Non-Ferrous Metals, Coal, Steel, Energy, Chemical, Rare Earth  
**Sectors**:  
- **Healthcare**: Pharma, Medical Device, Innovative Pharma, TCM, Bio-Pharma  
- **Consumer**: Liquor, Food & Beverage, Home Appliances, Breeding/Livestock  
- **Tech/Other**: Semiconductor, New Energy, PV/Solar, Banks, Securities, Defense, Dividend

## Technical Stack

- **ML Framework**: Qlib (Microsoft), LightGBM
- **Data**: Daily OHLCV from Qlib provider
- **Backtest**: Custom engine with walk-forward validation
- **Backend**: FastAPI + MLflow
- **Frontend**: Next.js + Recharts
- **Deployment**: Docker (optional)

## Development

### Run Tests
```bash
conda run -n quant python -m pytest tests/ -v
```

### Code Structure
- Task tracking: `.agent/workflows/`
- Artifacts: `artifacts/` (plots, results)
- Experiments: `mlruns/` (MLflow tracking)

## License

MIT

## Contact

For questions or collaboration: [GitHub Issues](https://github.com/tuctiki/delorean/issues)
