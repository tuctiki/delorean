---
name: alpha-evaluation
description: Audit existing alpha factors, filter out poor performers or redundant signals, and provide strategic recommendations for future research.
---

# Alpha Evaluation & Curation

This skill enables the agent to act as a **Senior Quantitative Researcher**. It is responsible for maintaining the quality of the factor library by auditing existing alphas, removing decayed or correlated signals, and suggesting improvements.

## 1. Capabilities

- **Performance Audit**: Re-evaluate existing factors on recent data to detect "alpha decay" (performance drop-off).
- **Redundancy Check**: Calculate the correlation matrix between a target factor and the existing pool to identify duplicates.
- **Triage**: Classify factors into **KEEP**, **REMOVE**, or **REWORK**.
- **Advisory**: Analyze *why* a factor failed or succeeded and suggest specific modifications (e.g., "reduce turnover," "add volatility filter").

## 2. Evaluation Metrics & Thresholds

The agent uses the following rubric to evaluate factors. Adjust these thresholds based on the `delorean` configuration:

| Metric | Threshold for "KEEP" | Description |
| :--- | :--- | :--- |
| **Information Coefficient (IC)** | > 0.03 | Predictive power of the raw signal. |
| **Sharpe Ratio** | > 1.5 | Risk-adjusted return (annualized). |
| **Turnover** | < 60% (Daily) | Trading cost proxy. |
| **Correlation** | < 0.7 | Similarity to existing approved factors. |
| **Fitness Score** | Top 20% | Custom combined score (Returns * IC / Volatility). |

## 3. Workflow

When asked to "review factors," "audit the portfolio," or "suggest improvements":

1.  **Load Factor Pool**: Retrieve the list of currently active formulas/expressions.
2.  **Backtest (Out-of-Sample)**: Run simulations specifically on the most recent data period (e.g., last 6 months) to check for decay.
3.  **Correlation Clustering**:
    - Group factors by similarity.
    - If two factors have Correlation > 0.7, keep the one with the higher Sharpe Ratio; mark the other for **REMOVAL**.
4.  **Diagnostics**:
    - *High Turnover?* -> Recommendation: "Apply a smoothing function like `Decay` or `Ma`."
    - *Low Volatility?* -> Recommendation: "Multiply by a volatility regime filter."
5.  **Reporting**: Output a table of actions and a summary of future research directions.

## 4. Tools & Libraries

- `delorean.analysis.PerformanceReport`: Generate tearsheets (cumulative returns, drawdown).
- `scipy.stats.pearsonr`: For correlation analysis.
- `seaborn.heatmap`: (Optional) To visualize the correlation matrix.
- `pandas`: For filtering and ranking the factor dataframe.

## 5. Example Prompts

- "Evaluate the 'momentum_v3' factor. Should we keep it?"
- "Audit the current `alpha_pool.json`. Remove highly correlated factors and output the clean list."
- "Why is factor #42 failing? Suggest a fix."
- "Analyze our current best factors and tell me which market anomalies we are missing (e.g., are we too heavy on Mean Reversion?)."

## 6. Output Template (Research Direction)

When providing **Future Research Directions**, use this format:

> **Status**: [KEEP / REMOVE / REWORK]
> **Reasoning**: Factor has strong IC (0.05) but excessive turnover (80%). Correlation with 'Trend_A' is low.
> **Recommendation**: Attempt to lower turnover by adding a transaction cost penalty or using `TsRank(x, 10)` instead of `Rank(x)`.
> **New Direction**: Investigate adding volume-weighted filters to this signal to improve stability.
