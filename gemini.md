# Gemini Project Setup

## Environment

This project requires the `quant` conda environment. It is recommended to use `direnv` to automatically activate the environment when you enter the project directory.

### Using direnv (Recommended)

1.  Install `direnv`. You can find installation instructions for your OS here: [https://direnv.net/docs/installation.html](https://direnv.net/docs/installation.html)
2.  Hook it into your shell. For bash, add `eval "$(direnv hook bash)"` to your `~/.bashrc`. For other shells, see the `direnv` documentation.
3.  In the root of this project, create a file named `.envrc` with the following content:

    ```
    layout conda quant
    ```

4.  Run `direnv allow`.

Now, whenever you `cd` into the project directory, the `quant` environment will be activated automatically.

### Manual Activation

If you prefer not to use `direnv`, you must manually activate the environment each time you work on the project:

```bash
conda activate quant
```

### Important Notes
- Never install qlib, stick to pyqlib 0.9.7
- The qlib source code package is available under `./vendors/qlib`

## Data Acquisition

This section outlines the process for acquiring and preparing ETF data for use with Qlib. The default Qlib data (`~/.qlib/qlib_data/cn_data`) primarily contains individual stock data and does not include specific ETF instruments in a readily accessible format for direct querying.

### 1. Downloading ETF Data from AkShare

Historical ETF data is downloaded using the `download_etf_data_to_csv.py` script, which has been adapted to fetch data from AkShare's ETF historical interface.

-   **Script:** `download_etf_data_to_csv.py`
-   **AkShare Function:** `ak.fund_etf_hist_em` is utilized to retrieve daily historical data for the specified ETF symbols.
-   **ETF List:** The script uses the `ETF_LIST` (defined within the script for self-containment, mirrored from `data_acquisition.py`) which includes various Chinese ETFs.
-   **Date Range:** Data is fetched from `2015-01-01` to `2026-01-01` (or the latest available date if earlier than 2026).
-   **Output:** The downloaded and formatted data for each ETF is saved as a CSV file in the `~/.qlib/csv_data/akshare_etf_data/` directory.

### 2. Converting Data to Qlib Binary Format

The downloaded CSV files are then converted into Qlib's efficient binary format using the `dump_bin.py` utility provided by Qlib.

-   **Tool:** `vendors/qlib/scripts/dump_bin.py`
-   **Command Used:**
    ```bash
    python vendors/qlib/scripts/dump_bin.py dump_all \
    --data_path ~/.qlib/csv_data/akshare_etf_data \
    --qlib_dir ~/.qlib/qlib_data/cn_etf_data \
    --freq day \
    --date_field_name date \
    --symbol_field_name symbol \
    --file_suffix .csv \
    --exclude_fields symbol
    ```
    -   `--data_path`: Specifies the input directory containing the ETF CSVs.
    -   `--qlib_dir`: Defines the target directory (`~/.qlib/qlib_data/cn_etf_data`) where the Qlib binary data will be stored. This ensures separation from the default stock data.
    -   `--exclude_fields symbol`: Crucially, the `symbol` column, which is a string identifier, is excluded from conversion to avoid `ValueError` during the numerical binary dump process.
-   **Result:** Qlib-formatted binary data (features, calendars, instruments) for the ETFs are generated in `~/.qlib/qlib_data/cn_etf_data`.

### 3. Using the Acquired Data in Qlib Applications

To access the newly prepared ETF data within Qlib-based applications, the `provider_uri` in the Qlib initialization must be updated.

-   **Script Modification:** The `initialize_qlib` function in `access_qlib_etf_data.py` was updated to point to the new ETF data directory:
    ```python
def initialize_qlib(provider_uri="~/.qlib/qlib_data/cn_etf_data", region=REG_CN):
    # ...
    ```
-   **Data Access:** With this configuration, functions like `qlib.data.D.features()` can now correctly retrieve historical data for the ETFs using their respective symbols (e.g., "510300.SH").
