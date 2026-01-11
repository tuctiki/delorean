import subprocess
import re
import numpy as np
import sys
import os

# Ensure we are in the project root or can find scripts
# Assuming run from project root

SEEDS = [42, 12345, 2024, 888, 1]
# SEEDS = [42, 100] # Short test

def run_backtest(seed):
    print(f"\n{'='*40}")
    print(f"Running Backtest with Seed: {seed}")
    print(f"{'='*40}")
    
    cmd = [
        "python", "scripts/run_etf_analysis.py",
        "--topk", "4",
        "--risk_parity",
        "--dynamic_exposure",
        "--seed", str(seed)
    ]
    
    try:
        # Capture stdout to parse metrics
        # Using export PYTHONPATH=. inside subprocess via env if needed, 
        # but usually we run this script with PYTHONPATH set.
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:."
        
        result = subprocess.run(
            cmd, 
            cwd=os.getcwd(),
            env=env,
            capture_output=True, 
            text=True,
            check=True
        )
        
        output = result.stdout
        
        # Regex to find metrics
        # Pattern: "Annualized Return: 12.70%"
        # Pattern: "Sharpe Ratio:      0.84"
        
        ret_match = re.search(r"Annualized Return:\s+([\d.-]+)%", output)
        sharpe_match = re.search(r"Sharpe Ratio:\s+([\d.-]+)", output)
        
        if ret_match and sharpe_match:
            ret = float(ret_match.group(1))
            sharpe = float(sharpe_match.group(1))
            print(f"-> [Success] Seed {seed}: Return = {ret}%, Sharpe = {sharpe}")
            return ret, sharpe
        else:
            print(f"-> [Error] Could not parse metrics for Seed {seed}")
            # print(output[-500:]) # Print last bit of output for debug
            return None, None
            
    except subprocess.CalledProcessError as e:
        print(f"-> [Fail] Script failed for Seed {seed}")
        print(e.stderr)
        return None, None

def main():
    results_ret = []
    results_sharpe = []
    
    for seed in SEEDS:
        ret, sharpe = run_backtest(seed)
        if ret is not None:
            results_ret.append(ret)
            results_sharpe.append(sharpe)
            
    print("\n" + "="*50)
    print("  Stability Analysis Results")
    print("="*50)
    
    if not results_ret:
        print("No successful runs.")
        return
        
    mean_ret = np.mean(results_ret)
    std_ret = np.std(results_ret)
    mean_sharpe = np.mean(results_sharpe)
    std_sharpe = np.std(results_sharpe)
    
    print(f"Seeds Tested: {SEEDS}")
    print(f"Successful Runs: {len(results_ret)}/{len(SEEDS)}")
    print("-" * 30)
    print(f"Annualized Return: Mean = {mean_ret:.2f}%, Std Dev = {std_ret:.2f}%")
    print(f"Sharpe Ratio:      Mean = {mean_sharpe:.2f}, Std Dev = {std_sharpe:.2f}")
    print("-" * 30)
    
    # Interpretation
    print("[Interpretation]")
    baseline_ret = 8.56
    if mean_ret > baseline_ret:
        print(f"-> PASS: Average Return ({mean_ret:.2f}%) > Baseline ({baseline_ret}%).")
    else:
        print(f"-> FAIL: Average Return ({mean_ret:.2f}%) <= Baseline ({baseline_ret}%).")
        
    if std_ret < 3.0:
        print(f"-> PASS: Volatility of Return results ({std_ret:.2f}%) is low (<3.0%).")
    else:
        print(f"-> WARN: Volatility of Return results ({std_ret:.2f}%) is high. Results depend heavily on seed.")

if __name__ == "__main__":
    main()
