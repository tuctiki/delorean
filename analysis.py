from qlib.contrib.evaluate import risk_analysis
import pandas as pd
import matplotlib.pyplot as plt

from constants import OUTPUT_DIR
import os

class ResultAnalyzer:
    def process(self, report, positions):
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
        # total_turnover is likely cumulative or just value, but 'turnover' is the rate
        turnover_rate = report["turnover"]
        avg_turnover = turnover_rate.mean() * 252 # Annualized
        print(f"\nAnnualized Turnover (Avg): {avg_turnover:.2%}")
        
        # Debug print to verify
        print("Tail of report columns (turnover check):")
        print(report[["total_turnover", "turnover"]].tail())

        self.analyze_results(report, analysis)

    def analyze_results(self, report, risk_metrics):
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


        except Exception as e:
            print(f"Plotting failed: {e}")

class FactorAnalyzer:
    def analyze(self, dataset):
        print("\n=== Factor Impact Analysis ===")
        
        # 1. Prepare Data
        # We need the dataframe with features and label
        # segments["train"] is typically used for analysis
        df = dataset.prepare("train")
        
        # Separate features and label
        label_col = "Ref($close, -1) / $close - 1"
        if label_col not in df.columns:
            # Try to find the label column (it might be named differently or is the last column)
            # In Qlib, labels are often handled separately or named 'label' in some contexts,
            # but here we defined it explicitly. Let's check columns.
            # Usually dataset.prepare returns features. The label might be in a separate handler?
            # Actually, DataHandlerLP prepares both.
            # Let's inspect columns effectively.
            # "feature" columns are under 'feature' multiindex level if applicable, or flat.
            pass



        # Calculate IC and RankIC
        ic_data = []
        
        # If MultiIndex columns (feature, label)
        if isinstance(df.columns, pd.MultiIndex):
            feature_df = df["feature"]
            label_df = df["label"]
            # Assuming single label
            label = label_df.iloc[:, 0]
        else:
            # Handle flat columns
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
            
            # Algin indices
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
            
        results_df = pd.DataFrame(results).sort_values("IC", ascending=False)
        
        # Sort by Abs(IC) for impact
        results_df["Abs_IC"] = results_df["IC"].abs()
        results_df = results_df.sort_values("Abs_IC", ascending=False)
        
        print("\nFactor Analysis Results (Sorted by Impact):")
        print(results_df[["Feature", "IC", "RankIC", "Direction"]].to_string(index=False))
        
        self.plot_analysis(results_df, feature_df)
        
    def plot_analysis(self, results_df, feature_df):
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
