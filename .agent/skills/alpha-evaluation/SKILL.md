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
| **Information Coefficient (IC)** | > 0.02 | Mean Predictive power across all periods. |
| **Regime Stability** | > 70% | % of audited periods with positive Rank IC. |
| **Sharpe Ratio** | > 1.0 | Risk-adjusted return (annualized). |
| **Turnover** | < 60% (Daily) | Trading cost proxy. |
| **Correlation** | < 0.7 | Similarity to existing approved factors. |

## 3. Workflow

When asked to "review factors," "audit the portfolio," or "suggest improvements":

1.  **Multi-Period Regime Audit**: 
    - Retrieve the list of currently active formulas.
    - Run Rank IC audits across defined regimes (e.g., Bull Market 19-20, Transition 21-22, Bear/Choppy 23-25, Recent 2025).
    - Identify "Flipped" factors where Rank IC has reversed sign.

2.  **Comparative Backtesting**: 
    - Compare the **Full Library** against a **Refined Library** (pruned of noisy/decayed signals).
    - If the Refined Library has a higher Sharpe/Return with lower Drawdown, suggest pruning.

3.  **Triage**:
    - **KEEP**: Positive IC across most regimes and low correlation.
    - **REMOVE**: Negative IC in modern regimes or High Correlation (> 0.7) with a better factor.
    - **REWORK**: Sign-flipped factors or high-turnover signals.

4.  **Diagnostics & Rework**:
    - **Sign Flip**: If Rank IC is consistently negative in the modern regime but was positive historically, suggest a Sign Flip (multiply formula by -1).
    - **High Turnover (> 60%)?** -> **Mandatory**: "Apply `Mean(x, 5)` or `Decay(x, 5)` to reduce noise."
    - **Refinement**: Suggest moving to a smaller "Refined Ensemble" if it yields better out-of-sample performance.

5.  **Reporting**: Output a table comparing metrics by Period and provide a "Refined Ensemble" recommendation.


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
