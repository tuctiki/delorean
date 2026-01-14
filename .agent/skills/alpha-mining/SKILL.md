---
name: alpha-mining
description: Discover, backtest, and validate formulaic alpha factors to predict future asset returns using historical data.
---

# Alpha Mining

This skill enables the agent to act as a quantitative researcher ("Quant") to mine alpha factors. It leverages the "DeLorean" time-series capabilities to analyze historical data and predict future outcomes.

## 1. Capabilities
The agent can perform the following actions under this skill:
- **Factor Generation**: Construct mathematical formulas (alphas) from raw data (Open, High, Low, Close, Volume).
- **Backtesting**: Simulate the performance of these factors over a historical timeline.
- **Performance Evaluation**: Calculate key metrics: Information Ratio (IR), Sharpe Ratio, Max Drawdown, and Turnover.
- **Correlation Analysis**: Check for correlation with existing factors to ensure uniqueness.

## 2. Workflow
When asked to "mine alpha" or "find signals":

1.  **Data Loading**: 
    - Identify the target universe (e.g., Crypto, Equities) and time range.
    - Load the relevant OHLCV dataset.
    
2.  **Hypothesis Generation**:
    - Propose a hypothesis (e.g., "High volume on a down day predicts a reversal").
    - Convert this hypothesis into a formula expression (e.g., `-1 * Rank(TsArgMax(SignedPower(If(Returns < 0, Volume, -Volume), 2), 5))`).

3.  **Simulation (The DeLorean Loop)**:
    - Run a backtest using the generated formula.
    - *Constraint*: Ensure no look-ahead bias (do not use future data for current predictions).

4.  **Validation**:
    - **Pass Criteria**: 
        - Information Ratio (IR) > 0.05
        - Sharpe Ratio > 1.0
        - Turnover < 50%
    - If a factor fails, refine the formula and repeat.

## 3. Tools & Libraries
Use the available Python environment to execute these tasks. Preferred libraries:
- `pandas` / `polars`: For high-performance data manipulation.
- `numpy`: For numerical operations.
- `scipy.stats`: For statistical validation.
- *(Optional)* `qlib` or `alphalens`: For standardized backtesting if available in the environment.

## 4. Example Prompts
- "Mine a mean-reversion alpha factor for the hourly ETH/USDT pair."
- "Test the momentum factor `Rank(Close / Delay(Close, 10))` and report the Sharpe ratio."
- "Optimize this alpha to reduce turnover while maintaining profitability."

## 5. Safety & Constraints
- **Overfitting**: Avoid complex formulas with too many parameters.
- **Robustness**: Factors must perform well across different market regimes (Bull, Bear, Sideways).
