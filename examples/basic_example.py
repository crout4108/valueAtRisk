"""
Basic Value at Risk (VaR) Calculation Example

This example demonstrates the fundamental usage of the Value at Risk calculator
for a simple portfolio of stocks.

Author: Farshad Noravesh
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from VaR import ValueAtRisk, HistoricalVaR

def main():
    print("=" * 60)
    print("Basic Value at Risk (VaR) Calculation Example")
    print("=" * 60)

    # Define portfolio
    stocks = ['AAPL', 'MSFT', 'GOOGL']
    weights = np.array([0.5, 0.3, 0.2])  # Portfolio allocation
    portfolio_value = 1000000  # $1 million portfolio
    confidence_level = 0.95  # 95% confidence

    print(f"\nPortfolio Configuration:")
    print(f"  Stocks: {stocks}")
    print(f"  Weights: {weights}")
    print(f"  Total Value: ${portfolio_value:,.0f}")
    print(f"  Confidence Level: {confidence_level*100}%")

    # Fetch historical data
    print("\nFetching historical data from Yahoo Finance...")
    try:
        data = web.DataReader(
            stocks,
            data_source="yahoo",
            start='2020-01-01',
            end='2023-12-31'
        )['Adj Close']

        print(f"  Retrieved {len(data)} trading days of data")
        print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Calculate Parametric VaR
    print("\n" + "-" * 60)
    print("Parametric VaR (Variance-Covariance Method)")
    print("-" * 60)

    var_calc = ValueAtRisk(confidence_level, data, weights)

    # Daily VaR
    daily_var_pct = var_calc.var(window=1) * 100
    daily_var_dollar = var_calc.var(marketValue=portfolio_value, window=1)

    print(f"\nDaily VaR:")
    print(f"  Percentage: {daily_var_pct:.2f}%")
    print(f"  Dollar Amount: ${daily_var_dollar:,.0f}")
    print(f"  Interpretation: With {confidence_level*100}% confidence, the portfolio")
    print(f"                  will not lose more than ${daily_var_dollar:,.0f} in one day")

    # Annual VaR
    annual_var_pct = var_calc.var() * 100
    annual_var_dollar = var_calc.var(marketValue=portfolio_value)

    print(f"\nAnnual VaR:")
    print(f"  Percentage: {annual_var_pct:.2f}%")
    print(f"  Dollar Amount: ${annual_var_dollar:,.0f}")
    print(f"  Interpretation: With {confidence_level*100}% confidence, the portfolio")
    print(f"                  will not lose more than ${annual_var_dollar:,.0f} in one year")

    # Calculate Historical VaR
    print("\n" + "-" * 60)
    print("Historical VaR (Historical Simulation Method)")
    print("-" * 60)

    hist_var = HistoricalVaR(confidence_level, data.values, weights)

    # VaR using all data
    hist_var_pct = hist_var.var() * 100
    hist_var_dollar = hist_var.var(marketValue=portfolio_value)

    print(f"\nHistorical VaR (all data):")
    print(f"  Percentage: {hist_var_pct:.2f}%")
    print(f"  Dollar Amount: ${hist_var_dollar:,.0f}")

    # VaR using 100-day window
    hist_var_100d_pct = hist_var.var(window=100) * 100
    hist_var_100d_dollar = hist_var.var(marketValue=portfolio_value, window=100)

    print(f"\nHistorical VaR (100-day window):")
    print(f"  Percentage: {hist_var_100d_pct:.2f}%")
    print(f"  Dollar Amount: ${hist_var_100d_dollar:,.0f}")

    # Compare methods
    print("\n" + "-" * 60)
    print("Method Comparison")
    print("-" * 60)
    print(f"\nParametric Daily VaR:  ${daily_var_dollar:,.0f}")
    print(f"Historical VaR (100d): ${hist_var_100d_dollar:,.0f}")
    print(f"\nDifference: ${abs(daily_var_dollar - hist_var_100d_dollar):,.0f}")

    # Portfolio statistics
    print("\n" + "-" * 60)
    print("Portfolio Statistics")
    print("-" * 60)

    cov_matrix = var_calc.covMatrix()
    variance = var_calc.calculateVariance()
    volatility = np.sqrt(variance)

    print(f"\nPortfolio Volatility (daily): {volatility:.4f} ({volatility*100:.2f}%)")
    print(f"Portfolio Volatility (annual): {volatility*np.sqrt(252):.4f} ({volatility*np.sqrt(252)*100:.2f}%)")

    print("\nCovariance Matrix:")
    print(pd.DataFrame(cov_matrix, index=stocks, columns=stocks).round(6))

    print("\n" + "=" * 60)
    print("Example Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
