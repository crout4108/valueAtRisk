# Examples

This directory contains practical examples demonstrating various use cases of the Value at Risk (VaR) calculator.

## Available Examples

### 1. Basic Example (`basic_example.py`)

Demonstrates fundamental VaR calculations for a simple portfolio.

**Features:**
- Fetching historical data from Yahoo Finance
- Calculating parametric VaR (daily and annual)
- Calculating historical VaR (full data and rolling window)
- Comparing different VaR methods
- Portfolio statistics (covariance matrix, volatility)

**Run:**
```bash
python examples/basic_example.py
```

### 2. Portfolio Comparison (`portfolio_comparison.py`)

Compares VaR across different portfolio allocations to demonstrate the impact of diversification.

**Features:**
- Multiple allocation strategies (equal weight, tech heavy, conservative, single stock)
- Side-by-side comparison of risk metrics
- Quantifying diversification benefits
- Optional visualization of results

**Run:**
```bash
python examples/portfolio_comparison.py
```

### 3. Confidence Level Sensitivity (`confidence_sensitivity.py`)

Analyzes how VaR changes with different confidence levels.

**Features:**
- Testing multiple confidence levels (90%, 95%, 99%, 99.9%)
- Both parametric and historical VaR calculations
- Interpretation guide for confidence levels
- Method comparison at high confidence levels

**Run:**
```bash
python examples/confidence_sensitivity.py
```

### 4. Historical VaR (`historical_var_example.py`)

Dedicated demonstration of the `HistoricalVaR` class using self-contained synthetic data (no internet connection required).

**Features:**
- Historical VaR using the full price history
- Rolling-window VaR at multiple look-back periods (30, 60, 100, 252 days)
- Confidence level sensitivity (90%, 95%, 99%)
- Side-by-side comparison with parametric VaR
- Impact of portfolio weight allocation on historical VaR
- Accepting a pandas DataFrame as input

**Run:**
```bash
python examples/historical_var_example.py
```

## Prerequisites

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- numpy
- scipy
- pandas_datareader
- matplotlib

## Data Source

All examples fetch historical stock price data from Yahoo Finance using `pandas_datareader`. You'll need an active internet connection to run these examples.

## Modifying Examples

Feel free to modify these examples to suit your needs:

- **Change stocks**: Edit the `stocks` list
- **Adjust weights**: Modify the `weights` array
- **Change time period**: Update `start` and `end` dates
- **Adjust portfolio value**: Modify `portfolio_value`
- **Change confidence level**: Update `confidence_level`

Example:
```python
# Your custom portfolio
stocks = ['AAPL', 'TSLA', 'NVDA']
weights = np.array([0.4, 0.3, 0.3])
portfolio_value = 500000
confidence_level = 0.99
```

## Expected Output

Each example produces formatted console output with:
- Portfolio configuration details
- VaR calculations and interpretations
- Comparative analyses
- Key insights and observations

## Troubleshooting

### Data Fetching Issues

If you encounter errors fetching data:
1. Check your internet connection
2. Verify stock tickers are valid
3. Try adjusting the date range
4. Yahoo Finance may temporarily block requests - wait a few minutes and retry

### Import Errors

If you get import errors:
```bash
# Make sure you're in the project root directory
cd /path/to/valueAtRisk

# Run with proper path
python -m examples.basic_example
```

Or add the project root to your Python path:
```python
import sys
sys.path.insert(0, '/path/to/valueAtRisk')
from VaR import ValueAtRisk, HistoricalVaR
```

## Additional Resources

- See [README.md](../README.md) for API documentation
- See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines
- See main [runVaR](../runVaR) script for another example

## Creating Your Own Examples

When creating new examples:

1. Follow the existing structure and style
2. Include clear documentation and comments
3. Add interpretation of results
4. Handle errors gracefully
5. Add your example to this README

Example template:
```python
"""
Your Example Title

Brief description of what this example demonstrates.

Author: Your Name
"""

import numpy as np
from VaR import ValueAtRisk

def main():
    print("=" * 60)
    print("Your Example Title")
    print("=" * 60)

    # Your code here

if __name__ == "__main__":
    main()
```

## Contributing Examples

If you've created a useful example, consider contributing it! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

For questions or issues with examples, please open an issue on GitHub.
