# Value at Risk (VaR) Calculator

![Python](https://img.shields.io/badge/python-3.x-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A Python library for calculating **Value at Risk (VaR)** for financial portfolios using both parametric and historical methods.

## 📊 Overview

Value at Risk (VaR) is a statistical measure used in finance to quantify the level of financial risk within a portfolio over a specific time frame. This library provides two implementations:

1. **Parametric VaR** - Uses variance-covariance matrix and assumes normal distribution
2. **Historical VaR** - Uses historical returns without distributional assumptions

While VaR is important in finance, note that **Conditional Value at Risk (CVaR)** is often preferred as it can be reduced to linear programming and solved with the simplex algorithm.

## ✨ Features

- Calculate VaR for multi-asset portfolios
- Support for both parametric (variance-covariance) and historical simulation methods
- Flexible time windows (daily, annualized, or custom periods)
- Portfolio variance calculation with two approaches:
  - Covariance matrix method (exact)
  - Portfolio return approximation
- Easy integration with Yahoo Finance data via `pandas_datareader`
- Configurable confidence intervals
- Results available in both percentage and dollar amounts

## 📦 Installation

### Prerequisites

```bash
pip install pandas numpy scipy pandas_datareader matplotlib
```

### Clone the Repository

```bash
git clone https://github.com/crout4108/valueAtRisk.git
cd valueAtRisk
```

## 🚀 Quick Start

### Basic Usage

```python
from VaR import ValueAtRisk, HistoricalVaR
import pandas_datareader.data as web
import numpy as np

# Define portfolio
stocks = ['GOOG', 'FB', 'TSLA', 'AMZN']
weights = np.array([0.4, 0.3, 0.2, 0.1])  # Portfolio weights (must sum to 1)

# Fetch historical data
data = web.DataReader(stocks, data_source="yahoo", 
                     start='12/04/2017', end='09/01/2020')['Adj Close']

# Calculate Parametric VaR
var_calc = ValueAtRisk(interval=0.95, matrix=data, weights=weights)

# Daily VaR (1% or $33,155 for $1M portfolio)
daily_var_pct = var_calc.var(window=1) * 100
daily_var_dollar = var_calc.var(marketValue=1000000, window=1)

# Annualized VaR (52.6% or $526,333 for $1M portfolio)
annual_var_pct = var_calc.var() * 100
annual_var_dollar = var_calc.var(marketValue=1000000)

# Calculate Historical VaR
hist_var = HistoricalVaR(interval=0.95, 
                        matrix=data.values, weights=weights)
hist_var_dollar = hist_var.var(marketValue=1000000)
```

### Running the Example

```bash
python runVaR
```

## 📖 API Documentation

### ValueAtRisk Class

The `ValueAtRisk` class implements parametric VaR calculation using the variance-covariance method.

#### Constructor

```python
ValueAtRisk(interval, matrix, weights)
```

**Parameters:**
- `interval` (float): Confidence interval (0 < interval < 1), e.g., 0.95 for 95%
- `matrix` (ndarray or DataFrame): Stock price matrix where rows represent dates/time periods and columns represent different tickers/assets
- `weights` (ndarray): Portfolio weights (must sum to 1)

#### Methods

##### `covMatrix()`
Returns the variance-covariance matrix of portfolio returns.

**Returns:** `ndarray` - Covariance matrix

##### `calculateVariance(Approximation=False)`
Calculates portfolio variance.

**Parameters:**
- `Approximation` (bool): If True, uses portfolio return approximation; if False, uses covariance matrix method

**Returns:** `float` - Portfolio variance

##### `var(marketValue=0, Approximation=False, window=252)`
Calculates parametric Value at Risk.

**Parameters:**
- `marketValue` (float): Portfolio value in dollars. If ≤ 0, returns percentage VaR
- `Approximation` (bool): Variance calculation method
- `window` (int): Time period scaling factor (default: 252 for annualized, 1 for daily)

**Returns:** `float` - VaR in dollars or percentage

##### `setCI(interval)`
Updates the confidence interval.

**Parameters:**
- `interval` (float): New confidence interval (0 < interval < 1)

##### `setPortfolio(matrix)`
Updates the portfolio data.

**Parameters:**
- `matrix` (ndarray or DataFrame): New stock price matrix

##### `setWeights(weights)`
Updates portfolio weights.

**Parameters:**
- `weights` (ndarray): New portfolio weights

### HistoricalVaR Class

The `HistoricalVaR` class extends `ValueAtRisk` and implements historical simulation VaR.

#### Constructor

Inherits from `ValueAtRisk` with the same parameters.

#### Methods

##### `var(marketValue=0, window=0)`
Calculates historical VaR using percentile method.

**Parameters:**
- `marketValue` (float): Portfolio value in dollars. If ≤ 0, returns percentage VaR
- `window` (int): Look-back period. If 0, uses entire price series

**Returns:** `float` - VaR in dollars or percentage

## 🧮 Mathematical Background

### Parametric VaR

The parametric VaR assumes returns follow a normal distribution and is calculated as:

```
VaR = -z_α × σ_p × √t × V
```

Where:
- `z_α` is the critical value at confidence level α (e.g., 1.645 for 95%)
- `σ_p` is the portfolio standard deviation
- `t` is the time horizon
- `V` is the portfolio value

Portfolio variance is calculated as:
```
σ_p² = w^T × Σ × w
```

Where:
- `w` is the weight vector
- `Σ` is the covariance matrix of returns

### Historical VaR

Historical VaR uses empirical distribution of returns:

```
VaR = -Percentile(returns, 1-α) × V
```

This method makes no distributional assumptions and uses actual historical returns.

## 📋 Requirements

- Python 3.x
- pandas
- numpy
- scipy
- pandas_datareader
- matplotlib

## 🎯 Example Output

Running the example script produces:

```
Portfolio Data (4 stocks, 691 trading days):

Symbols        AMZN      FB     GOOG    TSLA
Date                                        
2017-12-04  1133.95  171.47   998.68   61.04
2017-12-05  1141.57  172.83  1005.15   60.74
2017-12-06  1152.35  176.06  1018.38   62.65
2017-12-07  1159.79  180.14  1030.93   62.25
2017-12-08  1162.00  179.00  1037.05   63.03
2017-12-11  1168.92  179.04  1041.10   65.78
2017-12-12  1165.08  176.96  1040.48   68.21
2017-12-13  1164.13  178.30  1040.61   67.81
2017-12-14  1174.26  178.39  1049.15   67.58
2017-12-15  1179.14  180.18  1064.19   68.69
2017-12-18  1190.58  180.82  1077.14   67.77
2017-12-19  1187.38  179.51  1070.68   66.22
2017-12-20  1177.62  177.89  1064.95   65.80
2017-12-21  1174.76  177.45  1063.63   66.33
2017-12-22  1168.36  177.20  1060.12   65.04
2017-12-26  1176.76  175.99  1056.74   63.46
2017-12-27  1182.26  177.62  1049.37   62.33
2017-12-28  1186.10  177.92  1048.14   63.07
2017-12-29  1169.47  176.46  1046.40   62.27
2018-01-02  1189.01  181.42  1065.00   64.11
2018-01-03  1204.20  184.67  1082.48   63.45
2018-01-04  1209.59  184.33  1086.40   62.92
2018-01-05  1229.14  186.85  1102.23   63.32
2018-01-08  1246.87  188.28  1106.94   67.28
2018-01-09  1252.70  187.87  1106.26   66.74
2018-01-10  1254.33  187.84  1102.61   66.96
2018-01-11  1276.68  187.77  1105.52   67.59
2018-01-12  1305.20  179.37  1122.26   67.24
2018-01-16  1304.86  178.39  1121.76   68.01
2018-01-17  1295.00  177.60  1131.98   69.43
...             ...     ...      ...     ...
2020-07-22  3099.91  239.87  1568.49  318.47
2020-07-23  2986.55  232.60  1515.68  302.61
2020-07-24  3008.91  230.71  1511.87  283.40
2020-07-27  3055.21  233.50  1530.20  307.92
2020-07-28  3000.33  230.12  1500.34  295.30
2020-07-29  3033.53  233.29  1522.02  299.82
2020-07-30  3051.88  234.50  1531.45  297.50
2020-07-31  3164.68  253.67  1482.96  286.15
2020-08-03  3111.89  251.96  1474.45  297.00
2020-08-04  3138.83  249.83  1464.97  297.40
2020-08-05  3205.03  249.12  1473.61  297.00
2020-08-06  3225.00  265.28  1500.10  297.92
2020-08-07  3167.46  268.44  1494.49  290.54
2020-08-10  3148.16  263.00  1496.10  283.71
2020-08-11  3080.67  256.13  1480.32  274.88
2020-08-12  3162.24  259.89  1506.62  310.95
2020-08-13  3161.02  261.30  1518.45  324.20
2020-08-14  3148.02  261.24  1507.73  330.14
2020-08-17  3182.41  261.16  1517.98  367.13
2020-08-18  3312.49  262.34  1558.60  377.42
2020-08-19  3260.48  262.59  1547.53  375.71
2020-08-20  3297.37  269.01  1581.75  400.37
2020-08-21  3284.72  267.01  1580.42  410.00
2020-08-24  3307.46  271.39  1588.20  402.84
2020-08-25  3346.49  280.82  1608.22  404.67
2020-08-26  3441.85  303.91  1652.38  430.63
2020-08-27  3400.00  293.22  1634.33  447.75
2020-08-28  3401.80  293.66  1644.41  442.68
2020-08-31  3450.96  293.20  1634.18  498.32
2020-09-01  3499.12  295.44  1660.71  475.05

[691 rows x 4 columns]

Parametric VaR Results:
- Annualized VaR: 52.63% ($526,333 for $1M portfolio)
- Daily VaR: 3.32% ($33,156 for $1M portfolio)

Historical VaR Results:
- Overall VaR: 0.0333% ($33,349 for $1M portfolio)
- 100-day VaR: 3.11% ($31,093 for $1M portfolio)
```

## 🛠️ How It Works

### Data Processing
1. Fetches historical adjusted closing prices from Yahoo Finance
2. Calculates log returns: `r_t = ln(P_t / P_{t-1})`
3. Applies portfolio weights to compute portfolio returns

### Parametric Method
1. Computes covariance matrix of returns
2. Calculates portfolio variance using weights
3. Applies normal distribution inverse CDF with specified confidence level
4. Scales to desired time period using square root of time rule

### Historical Method
1. Computes weighted portfolio returns for each historical period
2. Sorts returns and finds the percentile corresponding to confidence level
3. No distributional assumptions required

## 💡 Use Cases

- **Risk Management**: Quantify maximum expected loss at a given confidence level
- **Portfolio Optimization**: Compare risk across different portfolio allocations
- **Regulatory Compliance**: Meet capital reserve requirements (Basel III)
- **Performance Reporting**: Communicate risk metrics to stakeholders
- **Stress Testing**: Evaluate portfolio resilience under historical scenarios

## 🔬 Limitations

- **Parametric VaR**: Assumes normal distribution (may underestimate tail risk)
- **Historical VaR**: Limited by historical data availability and relevance
- **Fat Tails**: Both methods may underestimate risk during extreme events
- **Non-stationarity**: Assumes future returns behave like historical returns

**Note**: Consider using **Conditional VaR (CVaR)** or **Expected Shortfall** for more robust risk measures, especially for portfolios with non-normal return distributions.

## 👨‍💻 Author

**Farshad Noravesh**
- Email: bmmturbo@icloud.com
- GitHub: [@crout4108](https://github.com/crout4108)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🔮 Future Enhancements

- [ ] Conditional Value at Risk (CVaR) implementation
- [ ] Monte Carlo simulation method
- [ ] Cornish-Fisher VaR for non-normal distributions
- [ ] Portfolio optimization integration
- [ ] Interactive visualization dashboard
- [ ] Support for additional data sources
- [ ] Backtesting framework
- [ ] Risk decomposition by asset

## 📚 References

- J.P. Morgan (1996). "RiskMetrics Technical Document"
- Jorion, P. (2007). "Value at Risk: The New Benchmark for Managing Financial Risk"
- Basel Committee on Banking Supervision. "Basel III: A global regulatory framework"

## 🙏 Acknowledgments

- Yahoo Finance for providing historical market data
- The open-source Python community for excellent libraries

---

⚠️ **Disclaimer**: This library is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Always consult with qualified financial professionals before making investment decisions.
