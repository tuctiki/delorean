from typing import Dict, Any, Optional
from qlib.contrib.evaluate import risk_analysis
from qlib.data.dataset import DatasetH
import pandas as pd
import matplotlib.pyplot as plt
from .config import OUTPUT_DIR
import os

class ResultAnalyzer:
    """
    Analyzes backtest results, calculates metrics, and generates visualizations.
    """
    def process(self, report: pd.DataFrame, positions: Dict[Any, Any]) -> None:
        """
        Process the backtest report and positions.

        Args:
            report (pd.DataFrame): Daily backtest report containing return, account value, turnover, etc.
            positions (dict): Daily position details.
        """
        # Calculate risk analysis
        analysis = risk_analysis(report["return"], freq="day")

        print("\nBacktest Report Summary:")
        print(report.head())

        # Save results
        report.to_pickle(os.path.join(OUTPUT_DIR, "backtest_report.pkl"))
        pd.to_pickle(positions, os.path.join(OUTPUT_DIR, "backtest_positions.pkl"))
        pd.to_pickle(analysis, os.path.join(OUTPUT_DIR, "backtest_analysis.pkl"))
        print(f"\nBacktest results saved to pkl files in {OUTPUT_DIR}.")

        # Turnover
        # Use the pre-calculated turnover rate from Qlib
        turnover_rate = report["turnover"]
        avg_turnover = turnover_rate.mean() * 252 # Annualized
        print(f"\nAnnualized Turnover (Avg): {avg_turnover:.2%}")
        
        # Debug print to verify
        print("Tail of report columns (turnover check):")
        print(report[["total_turnover", "turnover"]].tail())

        self.analyze_results(report, analysis)

    def analyze_results(self, report: pd.DataFrame, risk_metrics: pd.DataFrame) -> None:
        """
        Print formatted metrics and generate plots.

        Args:
            report (pd.DataFrame): Backtest report.
            risk_metrics (pd.DataFrame): Risk analysis DataFrame from qlib.
        """
        print("\n=== Detailed Backtest Analysis ===")
        
        # 1. Metrics
        # Risk metrics from risk_analysis
        annual_return = risk_metrics.loc["annualized_return", "risk"]
        max_dd = risk_metrics.loc["max_drawdown", "risk"]
        sharpe = risk_metrics.loc["information_ratio", "risk"] # Assuming 0 risk-free roughly
        
        # Calculate Win Rate (Daily)
        win_rate = (report["return"] > 0).mean()
        
        print(f"Annualized Return: {annual_return:.2%}")
        print(f"Max Drawdown:      {max_dd:.2%}")
        print(f"Sharpe Ratio:      {sharpe:.2f}")
        print(f"Daily Win Rate:    {win_rate:.2%}")
        
        # 2. Plots
        try:
            # Cumulative Return
            cum_return = (1 + report["return"]).cumprod()
            bench_cum = (1 + report["bench"]).cumprod()
            
            plt.figure(figsize=(12, 6))
            plt.plot(cum_return.index, cum_return, label="Strategy", color='red')
            plt.plot(bench_cum.index, bench_cum, label="Benchmark (CSI300)", color='gray', alpha=0.7)
            plt.title("Cumulative Return: Strategy vs Benchmark")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "cumulative_return.png"))
            print(f"Saved plot: {os.path.join(OUTPUT_DIR, 'cumulative_return.png')}")
            
            # Excess Return
            excess_ret = (1 + report["return"]).cumprod() - (1 + report["bench"]).cumprod()
            plt.figure(figsize=(12, 6))
            plt.plot(excess_ret.index, excess_ret, label="Excess Return", color='blue')
            plt.title("Cumulative Excess Return")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "excess_return.png"))
            print(f"Saved plot: {os.path.join(OUTPUT_DIR, 'excess_return.png')}")

            # [NEW] Save Metrics JSON for Dashboard
            import json
            import datetime
            import math
            
            def sanitize(val):
                if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                    return None
                return val
            
            results = {
                "annual_return": sanitize(float(annual_return)),
                "max_drawdown": sanitize(float(max_dd)),
                "sharpe_ratio": sanitize(float(sharpe)),
                "win_rate": sanitize(float(win_rate)),
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            results_path = os.path.join(OUTPUT_DIR, "experiment_results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Saved experiment results to: {results_path}")

        except Exception as e:
            print(f"Plotting failed: {e}")

class FactorAnalyzer:
    """
    Analyzes the predictive power of features (IC/RankIC).
    """
    def analyze(self, dataset: DatasetH) -> None:
        """
        Calculate and print Information Coefficient (IC) for each feature.

        Args:
            dataset (DatasetH): The Qlib dataset object.
        """
        print("\n=== Factor Impact Analysis ===")
        
        # 1. Prepare Data
        df = dataset.prepare("train")
        
        # Separate features and label
        label_col = "Ref($close, -1) / $close - 1"
        if label_col not in df.columns:
            # If label is not directly found, it might be separate. 
            # Ideally Qlib handles this, but here we assume standard handler config.
            pass

        # Calculate IC and RankIC
        # If MultiIndex columns (feature, label)
        if isinstance(df.columns, pd.MultiIndex):
            feature_df = df["feature"]
            label_df = df["label"]
            label = label_df.iloc[:, 0]
        else:
            # Handle flat columns (fallback)
            if label_col in df.columns:
                label = df[label_col]
                # Filter out label to get features
                feature_df = df.drop(columns=[label_col])
            else:
                print(f"Warning: Label column '{label_col}' not found in dataframe.")
                return

        results = []
        
        print(f"Analyzing {len(feature_df.columns)} features...")
        
        for feature_name in feature_df.columns:
            feature_val = feature_df[feature_name]
            
            # Align indices
            msg_df = pd.DataFrame({"feature": feature_val, "label": label}).dropna()
            
            if len(msg_df) < 10:
                continue

            # IC: Correlation
            ic = msg_df["feature"].corr(msg_df["label"])
            
            # RankIC: Spearman Correlation
            rank_ic = msg_df["feature"].corr(msg_df["label"], method="spearman")
            
            results.append({
                "Feature": feature_name,
                "IC": ic,
                "RankIC": rank_ic,
                "Direction": "Positive" if ic > 0 else "Negative"
            })
            
        if not results:
            print("No valid features found for analysis.")
            return

        results_df = pd.DataFrame(results).sort_values("IC", ascending=False)
        
        # Sort by Abs(IC) for impact
        results_df["Abs_IC"] = results_df["IC"].abs()
        results_df = results_df.sort_values("Abs_IC", ascending=False)
        
        print("\nFactor Analysis Results (Sorted by Impact):")
        print(results_df[["Feature", "IC", "RankIC", "Direction"]].to_string(index=False))
        
        self.plot_analysis(results_df, feature_df)
        
    def plot_analysis(self, results_df: pd.DataFrame, feature_df: pd.DataFrame) -> None:
        """
        Generate plots for factor analysis (IC bar chart, Correlation matrix).
        """
        try:
            # 1. IC Bar Plot
            plt.figure(figsize=(12, 8))
            # Sort for plotting
            plot_df = results_df.sort_values("IC", ascending=True)
            colors = ['red' if x < 0 else 'blue' for x in plot_df["IC"]]
            plt.barh(plot_df["Feature"], plot_df["IC"], color=colors)
            plt.title("Information Coefficient (IC) by Factor")
            plt.xlabel("IC Value")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "factor_ic.png"))
            print(f"Saved plot: {os.path.join(OUTPUT_DIR, 'factor_ic.png')}")
            
            # 2. Correlation Matrix
            plt.figure(figsize=(10, 10))
            corr = feature_df.corr()
            plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar()
            plt.xticks(range(len(corr)), corr.columns, rotation=90)
            plt.yticks(range(len(corr)), corr.columns)
            plt.title("Feature Correlation Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "feature_correlation.png"))
            print(f"Saved plot: {os.path.join(OUTPUT_DIR, 'feature_correlation.png')}")
            
        except Exception as e:
            print(f"Factor plotting failed: {e}")
