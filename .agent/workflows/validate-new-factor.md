---
description: How to add and validate a new alpha factor in the strategy
---
# Validate New Factor Workflow

This workflow guides you through the process of taking a newly discovered factor (e.g., from `mine_factors.py`), adding it to the codebase, and validating its performance impact.

## Prerequisites
- A valid Qlib expression for the new factor (e.g., discovered by miner)
- `quant` conda environment active

## Steps

1. **Add Factor to Codebase**:
   - Open `delorean/data.py`.
   - Locate the `ETFDataHandler.get_custom_factors` static method.
   - Add your new factor expression to the `custom_exprs` list.
   - Add a descriptive name to the `custom_names` list (order must match!).

   ```python
   # Example in delorean/data.py
   custom_exprs = [
       # ...
       "((Mean($close, 5) - $close) / Std($close, 20))", # <--- Add this
   ]
   custom_names = [
       # ...
       "New_Factor_Name",                 # <--- Add this
   ]
   ```

2. **Verify Syntax**:
   Run the data handler tests to ensure the expression is valid Qlib syntax.
   ```bash
   conda run -n quant python -m pytest tests/test_data_handler.py
   ```

3. **Run Validation Backtest**:
   Run a backtest on the validation period (e.g., 2021-2023) to check for improvement.
   
   **Option A: Standard (Regime Filter ON)**
   ```bash
   cd /Users/jinjing/workspace/delorean && conda run -n quant python scripts/run_etf_analysis.py \
     --start_time 2015-01-01 \
     --end_time 2023-12-31 \
     --train_end_time 2020-12-31 \
     --test_start_time 2021-01-01
   ```

   **Option B: Raw Alpha Power (Regime Filter OFF)**
   ```bash
   cd /Users/jinjing/workspace/delorean && conda run -n quant python scripts/run_etf_analysis.py \
     --no_regime \
     --start_time 2015-01-01 \
     --end_time 2023-12-31 \
     --train_end_time 2020-12-31 \
     --test_start_time 2021-01-01
   ```

4. **Compare Results**:
   Check `experiment_results.json` or the output logs.
   - **Rank IC**: Did it increase? (Target: > 0.05 combined)
   - **Sharpe Ratio**: Did it improve?
   - **Annual Return**: Is it better than -7% (baseline)?

## Decision
- **Keep**: If metrics improve significantly.
- **Discard**: If metrics degrade or are neutral.
