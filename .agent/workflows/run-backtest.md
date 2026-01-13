---
description: Run full backtest with latest data and latest code
---

# Run Backtest Workflow

This workflow downloads the latest ETF data, dumps it to Qlib format, and runs a full history backtest.

## Prerequisites
- Ensure the `quant` conda environment is active
- Internet connection for data download

## Steps

// turbo-all

1. Download latest ETF data from AkShare:
```bash
cd /Users/jinjing/workspace/delorean && conda run -n quant python scripts/download_etf_data_to_csv.py
```

2. Run the full history backtest (Train 2015-2021, Test 2022-Present):
```bash
cd /Users/jinjing/workspace/delorean && conda run -n quant python scripts/run_etf_analysis.py --start_time 2015-01-01 --end_time 2025-12-31 --train_end_time 2021-12-31 --test_start_time 2022-01-01 --topk 5 --smooth_window 15 --label_horizon 5
```

## Output
- Backtest results saved to `artifacts/experiment_results.json`
- Cumulative return plot saved to `artifacts/cumulative_return.png`
- MLflow experiment logged to `mlruns/`

## Notes
- The backtest uses the current strategy configuration from `delorean/config.py`
- Default parameters: TopK=5, Smooth Window=15, Label Horizon=5
- Results will be visible in the Experiments page of the dashboard

## Alternative: Quick Validation Run
For a faster validation (Train 2019-2021, Test 2022-2023):
```bash
cd /Users/jinjing/workspace/delorean && conda run -n quant python scripts/run_etf_analysis.py --start_time 2019-01-01 --end_time 2023-12-31 --train_end_time 2021-12-31 --test_start_time 2022-01-01
```
