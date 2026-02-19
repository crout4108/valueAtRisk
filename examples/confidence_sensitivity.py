"""
Confidence Level Sensitivity Analysis

This example demonstrates how VaR changes with different confidence levels
and helps understand the relationship between confidence and risk estimates.

Author: Farshad Noravesh
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from VaR import ValueAtRisk, HistoricalVaR

def main():
    print("=" * 70)
    print("Confidence Level Sensitivity Analysis")
    print("=" * 70)

    # Define portfolio
    stocks = ['AAPL', 'GOOGL', 'JPM', 'XOM']
    weights = np.array([0.3, 0.3, 0.2, 0.2])
    portfolio_value = 1000000

    print(f"\nPortfolio Configuration:")
    print(f"  Stocks: {stocks}")
    print(f"  Weights: {weights}")
    print(f"  Portfolio Value: ${portfolio_value:,.0f}")

    # Fetch historical data
    print("\nFetching historical data...")
    try:
        data = web.DataReader(
            stocks,
            data_source="yahoo",
            start='2020-01-01',
            end='2023-12-31'
        )['Adj Close']
        print(f"  Retrieved {len(data)} trading days")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Test different confidence levels
    confidence_levels = [0.90, 0.95, 0.99, 0.999]

    print("\n" + "=" * 70)
    print("Parametric VaR Sensitivity Analysis")
    print("=" * 70)

    parametric_results = []
    for conf_level in confidence_levels:
        var_calc = ValueAtRisk(conf_level, data, weights)

        daily_var = var_calc.var(marketValue=portfolio_value, window=1)
        annual_var = var_calc.var(marketValue=portfolio_value)

        parametric_results.append({
            'Confidence': f"{conf_level*100}%",
            'Daily VaR': f"${daily_var:,.0f}",
            'Annual VaR': f"${annual_var:,.0f}",
            'Daily %': f"{daily_var/portfolio_value*100:.2f}%",
            'Annual %': f"{annual_var/portfolio_value*100:.2f}%"
        })

    param_df = pd.DataFrame(parametric_results)
    print("\n" + param_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("Historical VaR Sensitivity Analysis")
    print("=" * 70)

    historical_results = []
    for conf_level in confidence_levels:
        hist_var = HistoricalVaR(conf_level, data.values, weights)

        var_full = hist_var.var(marketValue=portfolio_value)
        var_100d = hist_var.var(marketValue=portfolio_value, window=100)

        historical_results.append({
            'Confidence': f"{conf_level*100}%",
            'Full History': f"${var_full:,.0f}",
            '100-Day': f"${var_100d:,.0f}",
            'Full %': f"{var_full/portfolio_value*100:.2f}%",
            '100d %': f"{var_100d/portfolio_value*100:.2f}%"
        })

    hist_df = pd.DataFrame(historical_results)
    print("\n" + hist_df.to_string(index=False))

    # Interpretation
    print("\n" + "=" * 70)
    print("Interpretation Guide")
    print("=" * 70)

    print("""
Confidence Level Interpretation:

90% Confidence (α = 0.10):
  - VaR is exceeded on 10% of days (roughly 1 in 10 days)
  - Suitable for routine risk monitoring
  - Less conservative estimate

95% Confidence (α = 0.05):
  - VaR is exceeded on 5% of days (roughly 1 in 20 days)
  - Standard confidence level for risk management
  - Balanced between conservatism and practicality

99% Confidence (α = 0.01):
  - VaR is exceeded on 1% of days (roughly 1 in 100 days)
  - Regulatory requirement for many financial institutions
  - More conservative estimate

99.9% Confidence (α = 0.001):
  - VaR is exceeded on 0.1% of days (roughly 1 in 1000 days)
  - Very conservative, used for stress testing
  - Captures extreme tail events

Key Observations:
  - Higher confidence levels result in higher VaR estimates
  - The relationship is non-linear (doubling confidence doesn't double VaR)
  - Parametric VaR assumes normal distribution (may underestimate at high confidence)
  - Historical VaR uses actual data (better for capturing extreme events)
    """)

    # Compare methods at 99% confidence
    print("=" * 70)
    print("Method Comparison at 99% Confidence")
    print("=" * 70)

    var_99 = ValueAtRisk(0.99, data, weights)
    param_daily_99 = var_99.var(marketValue=portfolio_value, window=1)

    hist_99 = HistoricalVaR(0.99, data.values, weights)
    hist_daily_99 = hist_99.var(marketValue=portfolio_value, window=100)

    print(f"\nParametric VaR (daily): ${param_daily_99:,.0f}")
    print(f"Historical VaR (100d):  ${hist_daily_99:,.0f}")
    print(f"Difference: ${abs(param_daily_99 - hist_daily_99):,.0f}")

    if param_daily_99 < hist_daily_99:
        print("\nNote: Parametric VaR is lower, possibly due to fat tails in actual returns")
    else:
        print("\nNote: Historical VaR is lower, recent data may be less volatile")

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)

if __name__ == "__main__":
    main()
