from qlib.contrib.evaluate import risk_analysis
import pandas as pd
import matplotlib.pyplot as plt

class ResultAnalyzer:
    def process(self, report, positions):
        # Calculate risk analysis
        analysis = risk_analysis(report["return"], freq="day")

        print("\nBacktest Report Summary:")
        print(report.head())

        # Save results
        report.to_pickle("backtest_report.pkl")
        pd.to_pickle(positions, "backtest_positions.pkl")
        pd.to_pickle(analysis, "backtest_analysis.pkl")
        print("\nBacktest results saved to pkl files.")

        # Turnover
        # Calculate turnover rate: total_turnover / value
        turnover_rate = report["total_turnover"] / report["value"]
        avg_turnover = turnover_rate.mean() * 252 # Annualized
        print(f"\nAnnualized Turnover (Avg): {avg_turnover:.2%}")

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
            plt.savefig("cumulative_return.png")
            print("Saved plot: cumulative_return.png")
            
            # Excess Return
            excess_ret = (1 + report["return"]).cumprod() - (1 + report["bench"]).cumprod()
            plt.figure(figsize=(12, 6))
            plt.plot(excess_ret.index, excess_ret, label="Excess Return", color='blue')
            plt.title("Cumulative Excess Return")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("excess_return.png")
            print("Saved plot: excess_return.png")

        except Exception as e:
            print(f"Plotting failed: {e}")
