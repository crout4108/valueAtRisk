"""
Portfolio Comparison Example

This example demonstrates how to compare VaR across different portfolio allocations
to understand the impact of diversification on portfolio risk.

Author: Farshad Noravesh
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from VaR import ValueAtRisk
import matplotlib.pyplot as plt

def calculate_var_for_allocation(data, weights, confidence_level, portfolio_value):
    """Calculate VaR metrics for a given portfolio allocation."""
    var_calc = ValueAtRisk(confidence_level, data, weights)

    daily_var = var_calc.var(marketValue=portfolio_value, window=1)
    annual_var = var_calc.var(marketValue=portfolio_value)
    volatility = np.sqrt(var_calc.calculateVariance())

    return {
        'daily_var': daily_var,
        'annual_var': annual_var,
        'daily_volatility': volatility,
        'annual_volatility': volatility * np.sqrt(252)
    }

def main():
    print("=" * 70)
    print("Portfolio Comparison: Impact of Diversification on Risk")
    print("=" * 70)

    # Define portfolio parameters
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'JPM']
    portfolio_value = 1000000
    confidence_level = 0.95

    # Define different allocation strategies
    allocations = {
        'Equal Weight': np.array([0.25, 0.25, 0.25, 0.25]),
        'Tech Heavy': np.array([0.40, 0.40, 0.15, 0.05]),
        'Conservative': np.array([0.20, 0.20, 0.20, 0.40]),
        'Single Stock': np.array([1.00, 0.00, 0.00, 0.00]),
    }

    print(f"\nPortfolio Configuration:")
    print(f"  Stocks: {stocks}")
    print(f"  Portfolio Value: ${portfolio_value:,.0f}")
    print(f"  Confidence Level: {confidence_level*100}%")

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

    # Calculate VaR for each allocation
    print("\n" + "=" * 70)
    print("VaR Analysis by Allocation Strategy")
    print("=" * 70)

    results = {}
    for name, weights in allocations.items():
        print(f"\n{name}:")
        print(f"  Allocation: {dict(zip(stocks, weights))}")

        metrics = calculate_var_for_allocation(data, weights, confidence_level, portfolio_value)
        results[name] = metrics

        print(f"  Daily VaR: ${metrics['daily_var']:,.0f} ({metrics['daily_var']/portfolio_value*100:.2f}%)")
        print(f"  Annual VaR: ${metrics['annual_var']:,.0f} ({metrics['annual_var']/portfolio_value*100:.2f}%)")
        print(f"  Daily Volatility: {metrics['daily_volatility']*100:.2f}%")
        print(f"  Annual Volatility: {metrics['annual_volatility']*100:.2f}%")

    # Comparison table
    print("\n" + "=" * 70)
    print("Comparative Summary")
    print("=" * 70)

    comparison_df = pd.DataFrame({
        name: {
            'Daily VaR ($)': f"${metrics['daily_var']:,.0f}",
            'Annual VaR ($)': f"${metrics['annual_var']:,.0f}",
            'Daily Vol (%)': f"{metrics['daily_volatility']*100:.2f}%",
            'Annual Vol (%)': f"{metrics['annual_volatility']*100:.2f}%"
        }
        for name, metrics in results.items()
    })

    print("\n" + comparison_df.to_string())

    # Find best and worst allocations
    daily_vars = {name: metrics['daily_var'] for name, metrics in results.items()}
    best_allocation = min(daily_vars, key=daily_vars.get)
    worst_allocation = max(daily_vars, key=daily_vars.get)

    print("\n" + "=" * 70)
    print("Key Insights")
    print("=" * 70)
    print(f"\nLowest Risk: {best_allocation}")
    print(f"  Daily VaR: ${daily_vars[best_allocation]:,.0f}")
    print(f"\nHighest Risk: {worst_allocation}")
    print(f"  Daily VaR: ${daily_vars[worst_allocation]:,.0f}")
    print(f"\nRisk Reduction through Diversification:")
    print(f"  ${daily_vars[worst_allocation] - daily_vars[best_allocation]:,.0f}")
    print(f"  ({(1 - daily_vars[best_allocation]/daily_vars[worst_allocation])*100:.1f}% reduction)")

    # Visualization
    print("\n" + "=" * 70)
    print("Note: To generate visualization, uncomment the plotting code below")
    print("=" * 70)

    # Uncomment to generate bar chart
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    #
    # names = list(results.keys())
    # daily_vars_plot = [results[name]['daily_var'] for name in names]
    # annual_vols = [results[name]['annual_volatility']*100 for name in names]
    #
    # ax1.bar(names, daily_vars_plot, color=['green', 'blue', 'orange', 'red'])
    # ax1.set_ylabel('Daily VaR ($)', fontsize=12)
    # ax1.set_title('Daily VaR by Allocation Strategy', fontsize=14)
    # ax1.tick_params(axis='x', rotation=45)
    #
    # ax2.bar(names, annual_vols, color=['green', 'blue', 'orange', 'red'])
    # ax2.set_ylabel('Annual Volatility (%)', fontsize=12)
    # ax2.set_title('Annual Volatility by Allocation Strategy', fontsize=14)
    # ax2.tick_params(axis='x', rotation=45)
    #
    # plt.tight_layout()
    # plt.savefig('portfolio_comparison.png', dpi=300, bbox_inches='tight')
    # print("\nVisualization saved as 'portfolio_comparison.png'")

if __name__ == "__main__":
    main()
