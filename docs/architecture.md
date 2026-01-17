# Delorean Architecture

## Directory Structure

```
delorean_project/
├── delorean/            # Core Package
│   ├── alphas/          # Factor Definitions (Registry)
│   ├── conf/            # Configuration Modules (Assets, System, Model, Strategy)
│   ├── data/            # Data Layer (Handlers, Loaders)
│   ├── strategy/        # Execution & Portfolio logic
│   ├── runner.py        # Unified Strategy Execution Entrypoint
│   └── pipeline.py      # Daily Task Orchestrator
├── scripts/
│   ├── ops/             # Operational Runners (Live Trading, Backtest)
│   ├── research/        # Mining & Experiments
│   ├── analysis/        # Ad-hoc Analysis
│   └── data/            # Data Utilities
├── tests/               # Test Suite
└── docs/                # Documentation
```

## Key Components

### 1. Alpha Registry (`delorean.alphas`)
Centralized source of truth for all alpha factors. All factors used in production must be defined in `delorean/alphas/factors.py`.

### 2. Data Layer (`delorean.data`)
- **Loaders**: `ETFDataLoader` manages Qlib `DatasetH` creation.
- **Handlers**: `ETFDataHandler` (and variants) define how data is processed and labelled.

### 3. Configuration (`delorean.conf`)
Split into logical modules:
- `assets.py`: Universe definitions.
- `system.py`: Environment paths and time ranges.
- `model.py`: Hyperparameters.
- `strategy.py`: Live trading and backtest defaults.

### 4. Runner (`delorean.runner`)
The `StrategyRunner` class encapsulates the standard Qlib workflow:
1.  Initialize Qlib.
2.  Load Data.
3.  Train Model (Stage 1 + Optional Stage 2 Optimization).
4.  Generate Predictions included Smoothing.

## Workflows

- **Live Trading**: Executed via `scripts/ops/run_daily_task.py`.
- **Backtesting**: Executed via `scripts/ops/run_etf_analysis.py`.
- **Research**: New factors are mined in `scripts/research/` and manually promoted to `delorean/alphas/factors.py`.
