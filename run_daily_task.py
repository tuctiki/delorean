import subprocess
import sys
import os

def run_step(command, step_name):
    """
    Runs a shell command and prints status.
    """
    print(f"\n[{step_name}] Starting...")
    print(f"Command: {command}")
    
    try:
        # Run command and wait for completion
        # shell=True allows using complex arguments easily, but we should be careful.
        # Here inputs are fixed strings, so it's safe.
        result = subprocess.run(command, shell=True, check=True, text=True)
        print(f"[{step_name}] Success!")
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] {step_name} failed with exit code {e.returncode}.")
        sys.exit(1)

def main():
    print("="*50)
    print("  ETF Strategy - Daily Update Task")
    print("="*50)
    
    # 1. Download Latest Data
    step1_cmd = "python download_etf_data_to_csv.py"
    run_step(step1_cmd, "Step 1/3: Download Data from AkShare")
    
    # 2. Update Qlib Database (Binary Dump)
    # Using absolute paths or relative to project root? 
    # The previous guide assumed project root.
    # Note: We assume the user has the 'quant' env active or python refers to it.
    
    # Constructing the complex dumping command
    qlib_dump_cmd = (
        "python vendors/qlib/scripts/dump_bin.py dump_all "
        "--data_path ~/.qlib/csv_data/akshare_etf_data "
        "--qlib_dir ~/.qlib/qlib_data/cn_etf_data "
        "--freq day "
        "--date_field_name date "
        "--symbol_field_name symbol "
        "--file_suffix .csv "
        "--exclude_fields symbol"
    )
    run_step(qlib_dump_cmd, "Step 2/3: Update Qlib Database")
    
    # 3. Generate Signals
    step3_cmd = "python run_live_trading.py"
    run_step(step3_cmd, "Step 3/3: Generate Signal Recommendations")
    
    print("\n" + "="*50)
    print("  Daily Update Complete!")
    print("  Check the 'Top 4 Recommendations' above for action items.")
    print("="*50)

if __name__ == "__main__":
    main()
