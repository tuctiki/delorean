import os
import sys
import subprocess
import logging
from typing import List

# Setup Logger
logger = logging.getLogger("delorean.pipeline")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class DailyPipeline:
    """
    Orchestrates the daily update workflow:
    1. Download Data
    2. Format/Dump to Qlib
    3. Generate Trading Signals
    """
    def __init__(self, project_root: str = None):
        if project_root:
            self.root_dir = project_root
        else:
            # Default to parent of delorean package
            self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
    def run(self):
        logger.info("="*50)
        logger.info("  ETF Strategy - Daily Update Pipeline")
        logger.info("="*50)
        
        try:
            self.step_download()
            self.step_update_qlib()
            self.step_generate_signals()
            
            logger.info("\n" + "="*50)
            logger.info("  Pipeline Completed Successfully!")
            logger.info("="*50)
        except Exception as e:
            logger.error(f"Pipeline Failed: {e}")
            raise e

    def step_download(self):
        self._run_script("scripts/data/download_etf_data_to_csv.py", "Step 1/3: Download Data")

    def step_update_qlib(self):
        # Complex command for Qlib
        # We need to run this relative to project root
        script_path = "vendors/qlib/scripts/dump_bin.py"
        
        args = [
            "dump_all",
            "--data_path", "~/.qlib/csv_data/akshare_etf_data",
            "--qlib_dir", "~/.qlib/qlib_data/cn_etf_data",
            "--freq", "day",
            "--date_field_name", "date",
            "--symbol_field_name", "symbol",
            "--file_suffix", ".csv",
            "--exclude_fields", "symbol"
        ]
        
        self._run_script(script_path, "Step 2/3: Update Qlib Database", args=args)

    def step_generate_signals(self):
        self._run_script("scripts/ops/run_live_trading.py", "Step 3/3: Generate Signals")

    def _run_script(self, rel_path: str, step_name: str, args: List[str] = None):
        logger.info(f"\n[{step_name}] Starting...")
        
        script_path = os.path.join(self.root_dir, rel_path)
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")
            
        cmd = [sys.executable, rel_path]
        if args:
            cmd.extend(args)
            
        logger.info(f"Command: {' '.join(cmd)}")
        
        # We use cwd=self.root_dir to ensure relative imports in scripts work
        try:
            result = subprocess.run(
                cmd, 
                cwd=self.root_dir,
                check=True,
                text=True,
                # We want to see output in real-time or capture it?
                # If we are running in a service, we might want to capture.
                # But typically we want standard out.
            )
            logger.info(f"[{step_name}] Success!")
        except subprocess.CalledProcessError as e:
            logger.error(f"[{step_name}] Failed with exit code {e.returncode}.")
            raise e
