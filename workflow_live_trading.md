# Live Trading Workflow Guide

This guide explains how to use the ETF Strategy model for daily trading operations.

## Prerequisites
-   Active Conda Environment (`quant`)
-   Internet connection for data updates (AkShare)

## Daily Workflow

### 1. Update Data (After Market Close 15:00)
Fetch the latest daily K-line data to ensure the model sees today's market movements.

```bash
# 1. Download latest data from AkShare
python download_etf_data_to_csv.py

# 2. Convert to Qlib binary format (Update database)
# Note: Ensure you run this from the project root
python vendors/qlib/scripts/dump_bin.py dump_all \
    --data_path ~/.qlib/csv_data/akshare_etf_data \
    --qlib_dir ~/.qlib/qlib_data/cn_etf_data \
    --freq day \
    --date_field_name date \
    --symbol_field_name symbol \
    --file_suffix .csv \
    --exclude_fields symbol
```

### 2. Generate Trading Signals
Run the inference script to get the target portfolio for the **Next Trading Day**.

```bash
python run_live_trading.py
```

### 3. Execution (Manual)
The script will output the **Top 4 ETFs** and their scores.

**Trading Rules (Aggressive Turnover Reduction):**
1.  **Check Holdings**: Compare the "Top 4" recommendations with your current holdings.
2.  **Turnover Control**:
    -   If your current holdings are still in the Top 6-8 rankings? **HOLD** (Do not trade).
    -   If a holding drops to the bottom of the list? **SELL**.
    -   **Max Swaps**: Limit yourself to **1 swap per day** even if multiple signals appear.
    -   **96% Retention**: In case of doubt, **HOLD**. The strategy is designed to be lazy.

### 4. Example Output
```text
[5/5] Latest Signal Date: 2026-01-10

------------------------------
  Top 4 Recommendations
------------------------------
  #1  512480.SH  (Score: 0.0052)
  #2  516160.SH  (Score: 0.0048)
  #3  512010.SH  (Score: 0.0041)
  #4  510300.SH  (Score: 0.0035)
------------------------------
```
*Action*: Ensure these 4 assets are in your portfolio.

## Dashboard Backend & Frontend

To view the dashboard, you need to start both the Backend (API) and the Frontend (UI).

### 1. Start Backend Server
```bash
conda run -n quant python server/main.py
```
*Runs on http://localhost:8000*

### 2. Start Frontend Server
```bash
cd frontend
npm run dev
```
*Runs on http://localhost:3000*
