"""
Historical Value at Risk (VaR) Example

This example demonstrates the usage of HistoricalVaR using synthetic price data.
Historical VaR makes no distributional assumptions and instead uses the empirical
distribution of actual historical returns.

Mathematical Foundation:
    VaR = -Percentile(returns, 1-α) × V

    Where:
        - Percentile: The (1-α)th percentile of historical portfolio returns
        - α: Confidence level (e.g., 0.95 for 95%)
        - V: Portfolio market value
"""

import numpy as np
import pandas as pd
from VaR import ValueAtRisk, HistoricalVaR


def create_sample_portfolio():
    """Create a synthetic portfolio with reproducible price data."""
    np.random.seed(42)
    n_days = 252
    n_assets = 3
    base_prices = np.array([100.0, 200.0, 150.0])

    # Simulate daily returns with slight drift
    daily_returns = np.random.normal(0.0005, 0.015, (n_days - 1, n_assets))
    prices = np.zeros((n_days, n_assets))
    prices[0] = base_prices
    for i in range(1, n_days):
        prices[i] = prices[i - 1] * np.exp(daily_returns[i - 1])

    return prices


def main():
    print("=" * 65)
    print("Historical Value at Risk (VaR) Example")
    print("=" * 65)

    # --- Portfolio setup ---
    prices = create_sample_portfolio()
    weights = np.array([0.4, 0.35, 0.25])
    portfolio_value = 1_000_000  # $1 million portfolio
    confidence_level = 0.95

    print(f"\nPortfolio Configuration:")
    print(f"  Assets      : Stock A, Stock B, Stock C")
    print(f"  Weights     : {weights}")
    print(f"  Market Value: ${portfolio_value:,.0f}")
    print(f"  CI Level    : {confidence_level * 100:.0f}%")
    print(f"  Data Points : {len(prices)} trading days")

    # ------------------------------------------------------------------
    # 1. Historical VaR using all available data
    # ------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("1. Historical VaR — Full History")
    print("-" * 65)

    hist_var = HistoricalVaR(confidence_level, prices, weights)

    var_pct = hist_var.var()
    var_dollar = hist_var.var(marketValue=portfolio_value)

    print(f"\n  VaR (percentage): {var_pct * 100:.4f}%")
    print(f"  VaR (dollar)    : ${var_dollar:,.2f}")
    print(
        f"\n  Interpretation: With {confidence_level * 100:.0f}% confidence, the portfolio"
    )
    print(f"  will not lose more than ${var_dollar:,.0f} on any given day.")

    # ------------------------------------------------------------------
    # 2. Historical VaR with a rolling window
    # ------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("2. Historical VaR — Rolling Windows")
    print("-" * 65)

    windows = [30, 60, 100, 252]
    print(f"\n  {'Window (days)':<20} {'VaR (%)':<15} {'VaR ($)':<15}")
    print(f"  {'-'*18:<20} {'-'*13:<15} {'-'*13:<15}")

    for window in windows:
        w_pct = hist_var.var(window=window)
        w_dollar = hist_var.var(marketValue=portfolio_value, window=window)
        print(f"  {window:<20} {w_pct * 100:<15.4f} ${w_dollar:<14,.2f}")

    # ------------------------------------------------------------------
    # 3. VaR at different confidence levels
    # ------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("3. Historical VaR — Different Confidence Levels (100-day window)")
    print("-" * 65)

    confidence_levels = [0.90, 0.95, 0.99]
    print(f"\n  {'Confidence':<15} {'VaR (%)':<15} {'VaR ($)':<15}")
    print(f"  {'-'*13:<15} {'-'*13:<15} {'-'*13:<15}")

    for ci in confidence_levels:
        hist_var.setCI(ci)
        ci_pct = hist_var.var(window=100)
        ci_dollar = hist_var.var(marketValue=portfolio_value, window=100)
        print(f"  {ci * 100:.0f}%{'':<12} {ci_pct * 100:<15.4f} ${ci_dollar:<14,.2f}")

    # ------------------------------------------------------------------
    # 4. Comparison: Parametric VaR vs Historical VaR
    # ------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("4. Parametric VaR vs Historical VaR (Daily, 95% CI)")
    print("-" * 65)

    # Reset confidence interval
    hist_var.setCI(0.95)

    param_var = ValueAtRisk(0.95, prices, weights)
    daily_param_pct = param_var.var(window=1)
    daily_param_dollar = param_var.var(marketValue=portfolio_value, window=1)

    hist_full_pct = hist_var.var()
    hist_full_dollar = hist_var.var(marketValue=portfolio_value)

    print(f"\n  Parametric VaR (daily, 95%): {daily_param_pct * 100:.4f}%  /  ${daily_param_dollar:,.2f}")
    print(f"  Historical VaR (full hist.): {hist_full_pct * 100:.4f}%  /  ${hist_full_dollar:,.2f}")

    diff = abs(daily_param_dollar - hist_full_dollar)
    print(f"\n  Absolute difference: ${diff:,.2f}")
    print(
        "\n  Note: Parametric VaR assumes normal returns; Historical VaR uses"
    )
    print("        the actual empirical distribution (no distribution assumption).")

    # ------------------------------------------------------------------
    # 5. Impact of portfolio weights
    # ------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("5. Impact of Portfolio Weights on Historical VaR (95% CI, full history)")
    print("-" * 65)

    weight_scenarios = {
        "Equal weight        ": np.array([1/3, 1/3, 1/3]),
        "Stock A heavy (80%) ": np.array([0.80, 0.10, 0.10]),
        "Stock B heavy (80%) ": np.array([0.10, 0.80, 0.10]),
        "Stock C heavy (80%) ": np.array([0.10, 0.10, 0.80]),
    }

    print(f"\n  {'Scenario':<28} {'VaR (%)':<15} {'VaR ($)':<15}")
    print(f"  {'-'*26:<28} {'-'*13:<15} {'-'*13:<15}")

    for label, w in weight_scenarios.items():
        hist_var.setWeights(w)
        w_pct = hist_var.var()
        w_dollar = hist_var.var(marketValue=portfolio_value)
        print(f"  {label:<28} {w_pct * 100:<15.4f} ${w_dollar:<14,.2f}")

    # ------------------------------------------------------------------
    # 6. DataFrame input support
    # ------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("6. Using a pandas DataFrame as Input")
    print("-" * 65)

    df_prices = pd.DataFrame(prices, columns=["Stock_A", "Stock_B", "Stock_C"])
    hist_var_df = HistoricalVaR(0.95, df_prices, np.array([0.4, 0.35, 0.25]))
    df_var = hist_var_df.var(marketValue=portfolio_value)

    print(f"\n  Input type : pandas DataFrame with shape {df_prices.shape}")
    print(f"  VaR (95%)  : ${df_var:,.2f}")

    print("\n" + "=" * 65)
    print("Historical VaR Example Complete")
    print("=" * 65)


if __name__ == "__main__":
    main()
