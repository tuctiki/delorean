---
name: strategy-development
description: Compose validated alpha factors into a cohesive trading strategy, incorporating portfolio optimization, risk management, and execution logic.
---

# Strategy Development & Portfolio Construction

This skill enables the agent to act as a **Portfolio Manager**. While "Alpha Mining" finds the signals, this skill decides how to combine them, how much capital to allocate, and how to protect the portfolio from downside risk.

## 1. Capabilities

- **Factor Combination**: Merge multiple alpha factors into a single composite score using techniques like:
    - *Equal Weighting* (Simple Average)
    - *Inverse Volatility Weighting*
    - *Machine Learning Ensemble* (e.g., Linear Regression, XGBoost)
- **Portfolio Optimization**: Calculate optimal position sizes (weights) to maximize Sharpe Ratio or minimize Volatility (e.g., Mean-Variance Optimization).
- **Risk Management Implementation**:
    - Apply constraints (e.g., "Max 5% allocation per asset").
    - Implement Stop-Loss / Take-Profit logic.
    - Neutralize exposure (Dollar Neutral, Sector Neutral).
- **Regime Detection**: Adjust strategy logic based on market conditions (e.g., "Only trade Momentum factors during high volatility").

## 2. Strategy Components (The "Recipe")

When defining a strategy, the agent must specify these four pillars:

1.  **Signal Engine**: Which factors are included? (Output from `alpha-evaluation`).
2.  **Allocation Engine**: How do we map signals to target weights? 
    - *Example*: `TargetWeight = SignalStrength * VolatilityTarget`
3.  **Constraint Engine**: Hard rules that cannot be broken.
    - *Example*: `Leverage <= 1.5x`, `Long exposure == Short exposure`.
4.  **Execution Engine**: How to enter/exit?
    - *Example*: "Rebalance daily at Open," "TWAP over 1 hour."

## 3. Workflow

When asked to "build a strategy," "optimize the portfolio," or "create a trading bot":

1.  **Selection**: Import the "KEEP" list from the `alpha-evaluation` skill.
2.  **Correlation Check**: Ensure selected factors are uncorrelated (Diversification).
3.  **Combination**:
    - Normalize factors (z-score) to ensure they are on the same scale.
    - Apply weighting scheme.
4.  **Backtest Strategy**: Run the *full* portfolio simulation (accounting for transaction costs and slippage).
    - *Note*: This differs from factor testing; here we care about Net PnL, not just IC.
5.  **Stress Testing**: Simulate extreme market events (e.g., "What happens if correlation goes to 1?").

## 4. Tools & Libraries

- `scipy.optimize`: For finding optimal weights (Mean-Variance).
- `cvxpy`: For convex optimization problems with constraints.
- `sklearn.linear_model`: For learning factor weights dynamically.
- `delorean.portfolio.Portfolio`: The core class for managing holdings.
- `empyrical`: For advanced financial risk metrics (Sortino, Calmar).

## 5. Example Prompts

- "Combine our top 3 momentum factors into a single strategy using equal weights."
- "Create a strategy that trades Mean Reversion but neutralizes overall market beta (Market Neutral)."
- "Optimize the weights of these 5 factors to target 10% annual volatility."
- "Add a risk rule: Close all positions if drawdown exceeds 5% in a single day."

## 6. Output Template (Strategy Definition)

When outputting a strategy, use this structure:

> **Strategy Name**: `Multi_Factor_Neutral_V1`
> **Universe**: Top 500 Equities (Liquid)
> **Alphas Used**: 
>  1. `alpha_001` (Trend, 40% weight)
>  2. `alpha_045` (Reversion, 30% weight)
>  3. `alpha_099` (Volatility, 30% weight)
> **Optimization Goal**: Maximize Sharpe
> **Constraints**: 
>  - Max Single Position: 2%
>  - Sector Exposure: +/- 10%
> **Stop Loss**: Global 10% Trailing Stop
