# Delorean - ETF Trading Strategy

A quantitative trading strategy for Chinese ETFs using machine learning and technical factors with optimized turnover control.

## Overview

Delorean is a momentum-based ETF rotation strategy that:
- Predicts next-day returns using 7 proprietary alpha factors
- Holds top 4 ETFs with dynamic risk management
- Uses walk-forward validation for robust out-of-sample testing
- Achieves **Sharpe ratio of 0.87** with **22.5% annualized turnover**

## Key Features

### Alpha Factors (7 Total)
- **Momentum**: MOM60, MOM120
- **Structural**: Gap_Fill (mean reversion)
- **Trend Quality**: Mom_Persistence, Acceleration
- **Divergence**: Vol_Price_Div (volume-price correlation)
- **Defensive**: RSI_Divergence

### Portfolio Management
- **Risk Parity Weighting**: Inverse volatility weights
- **Target Volatility**: 20% annualized with dynamic scaling
- **Asymmetric Volatility Scaling**: 
  - Bull markets: 30% target vol (risk on)
  - Bear markets: 12% target vol (capital protection)
- **Market Regime Filter**: CSI 300 Price/MA60 ratio

### Turnover Control (NEW)
- **Signal Smoothing**: 3-day EMA + 10-day EWMA for noise reduction
- **Wide Buffer**: Rank hysteresis (TopK + 3 buffer zone)
- **Rebalancing Threshold**: 5% portfolio value threshold
- **Result**: 48% turnover reduction while improving Sharpe ratio

## Performance Metrics

**Walk-Forward Validation (2023-01-01 to 2026-01-16)**:
- **Sharpe Ratio**: 0.87
- **Annual Return**: 11.8%
- **Max Drawdown**: -18.3%
- **Annualized Turnover**: 22.5%
- **Rank IC**: 0.0194
- **Trading Frequency**: 36% (vs 65% baseline)

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

## ETF Universe (18 ETFs)

**Broad Market**: CSI 300, A500, ChiNext, STAR 50, CSI 1000, SSE 50, H-Share  
**Commodities**: Gold, Non-Ferrous Metals  
**Sectors**: Semiconductors, New Energy, Liquor, Banks, Pharma, Consumer, Solar, Securities, Defense, Dividends

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
