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

        # Turnover Analysis
        turnover_rate = report["turnover"]
        avg_daily_turnover = turnover_rate.mean()
        ann_turnover = avg_daily_turnover * 252  # Annualized
        trading_days = (turnover_rate > 0).sum()
        total_days = len(turnover_rate)
        
        print(f"\n--- Turnover Statistics ---")
        print(f"Daily Mean Turnover:   {avg_daily_turnover:.2%}")
        print(f"Annualized Turnover:   {ann_turnover:.2%}")
        print(f"Trading Days:          {trading_days} / {total_days} ({trading_days/total_days*100:.1f}%)")

        # Calculate Position Percentage (Exposure)
        # value = total market value of held positions
        # account = total account value (cash + positions)
        report["position_pct"] = report["value"] / report["account"]
        avg_pos_pct = report["position_pct"].mean()
        max_pos_pct = report["position_pct"].max()
        min_pos_pct = report["position_pct"].min()
        
        print(f"\n--- Exposure Statistics ---")
        print(f"Avg Position Exposure: {avg_pos_pct:.2%}")
        print(f"Max Position Exposure: {max_pos_pct:.2%}")
        print(f"Min Position Exposure: {min_pos_pct:.2%}")

        self.analyze_results(report, analysis, avg_daily_turnover, trading_days, total_days, 
                             avg_pos_pct, max_pos_pct, min_pos_pct)

    def analyze_results(self, report: pd.DataFrame, risk_metrics: pd.DataFrame, 
                        avg_daily_turnover: float = 0.0, trading_days: int = 0, total_days: int = 0,
                        avg_pos_pct: float = 0.0, max_pos_pct: float = 0.0, min_pos_pct: float = 0.0) -> None:
        """
        Print formatted metrics and generate plots.

        Args:
            report (pd.DataFrame): Backtest report.
            risk_metrics (pd.DataFrame): Risk analysis DataFrame from qlib.
            avg_daily_turnover (float): Average daily turnover rate.
            trading_days (int): Number of days with trading activity.
            total_days (int): Total number of trading days.
            avg_pos_pct (float): Average position percentage.
            max_pos_pct (float): Maximum position percentage.
            min_pos_pct (float): Minimum position percentage.
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
            
            plt.figure(figsize=(12, 10)) # Increased height for subplots if we combined, but here we keep separate or use subplots
            
            # Plot 1: Cumulative Return (Top)
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(cum_return.index, cum_return, label="Strategy", color='red')
            ax1.plot(bench_cum.index, bench_cum, label="Benchmark (CSI300)", color='gray', alpha=0.7)
            ax1.set_title("Cumulative Return: Strategy vs Benchmark")
            ax1.set_ylabel("Cumulative Return")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Position Exposure (Bottom)
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            ax2.plot(report.index, report["position_pct"], label="Position Exposure", color='blue', alpha=0.6)
            ax2.fill_between(report.index, report["position_pct"], alpha=0.2, color='blue')
            ax2.set_title("Daily Position Exposure (%)")
            ax2.set_ylabel("Exposure")
            ax2.set_ylim(0, 1.05) # Assume 0-100% mostly
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "cumulative_return.png"))
            print(f"Saved plot: {os.path.join(OUTPUT_DIR, 'cumulative_return.png')}")
            pass # Suppress output
            
            # Independent Position Plot (Optional but good for clarity)
            plt.figure(figsize=(12, 4))
            plt.plot(report.index, report["position_pct"], label="Position Exposure", color='purple')
            plt.fill_between(report.index, report["position_pct"], alpha=0.2, color='purple')
            plt.title("Daily Portfolio Exposure")
            plt.ylabel("Exposure %")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "position_exposure.png"))
            print(f"Saved plot: {os.path.join(OUTPUT_DIR, 'position_exposure.png')}")
            
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
                "daily_turnover": sanitize(float(avg_daily_turnover)),
                "annualized_turnover": sanitize(float(avg_daily_turnover * 252)),
                "avg_position_pct": sanitize(float(avg_pos_pct)), # [NEW]
                "trading_days": int(trading_days),
                "total_days": int(total_days),
                "trading_frequency": sanitize(float(trading_days / total_days)) if total_days > 0 else 0.0,
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
