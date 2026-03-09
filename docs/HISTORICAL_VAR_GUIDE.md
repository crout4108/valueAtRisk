# Historical VaR: Comprehensive Guide

This guide provides in-depth information about Historical Value at Risk (VaR) implementation, methodology, and best practices.

## Table of Contents

- [Introduction](#introduction)
- [Methodology](#methodology)
- [Implementation Details](#implementation-details)
- [When to Use Historical VaR](#when-to-use-historical-var)
- [Practical Examples](#practical-examples)
- [Common Pitfalls](#common-pitfalls)
- [Advanced Topics](#advanced-topics)
- [References](#references)

---

## Introduction

Historical VaR is a non-parametric method for estimating Value at Risk that uses the actual empirical distribution of historical returns. Unlike parametric VaR (which assumes normally distributed returns), Historical VaR makes no assumptions about the underlying distribution of returns.

### Key Characteristics

- **Distribution-free**: No assumption about return distribution shape
- **Data-driven**: Based entirely on observed historical returns
- **Intuitive**: Easy to explain to non-technical stakeholders
- **Captures tail events**: Includes actual extreme events from history

### Basic Formula

```
VaR = -Percentile(R_p, 100×(1-α)) × V
```

Where:
- `R_p` = Historical portfolio returns
- `α` = Confidence level (e.g., 0.95)
- `V` = Portfolio value

---

## Methodology

### Step-by-Step Process

#### 1. Calculate Portfolio Returns

For each time period `t`, compute the weighted portfolio return:

```
R_p(t) = Σ(w_i × r_i(t))
```

Where:
- `w_i` = Weight of asset i
- `r_i(t)` = Log return of asset i at time t
- Log returns: `r_i(t) = ln(P_i(t) / P_i(t-1))`

#### 2. Sort Returns

Arrange portfolio returns in ascending order (from most negative to most positive).

#### 3. Find Percentile

Identify the (1-α) percentile of the sorted returns:
- For 95% confidence: Find 5th percentile
- For 99% confidence: Find 1st percentile

#### 4. Calculate VaR

Take the absolute value of the percentile and multiply by portfolio value:

```
VaR = |Percentile value| × Portfolio Value
```

### Mathematical Properties

**Log Returns vs Simple Returns**

This implementation uses log returns for several advantages:
- Time additivity: `r(t1→t3) = r(t1→t2) + r(t2→t3)`
- Symmetry: A 10% gain followed by 10% loss yields zero log return
- Better statistical properties for aggregation
- Standard practice in financial modeling

**Percentile Method**

The implementation uses numpy's percentile function with 'nearest' interpolation method, which selects the actual return value closest to the desired percentile rather than interpolating between values.

---

## Implementation Details

### Class Structure

```python
class HistoricalVaR(ValueAtRisk):
    """
    Extends ValueAtRisk base class to provide historical simulation method.
    Inherits data processing and portfolio management capabilities.
    """
```

### Key Parameters

#### Confidence Interval (`interval`)
- **Type**: float (0 < interval < 1)
- **Common values**: 0.90, 0.95, 0.99
- **Interpretation**: Probability that losses will not exceed VaR
- **Example**: 0.95 means 95% confidence, examining worst 5% of outcomes

#### Price Matrix (`matrix`)
- **Type**: numpy array or pandas DataFrame
- **Shape**: (n_periods, n_assets)
- **Content**: Historical prices (typically adjusted closing prices)
- **Data frequency**: Usually daily, but can be any consistent frequency

#### Portfolio Weights (`weights`)
- **Type**: numpy array
- **Shape**: (n_assets,)
- **Constraint**: Should sum to 1.0 (though not enforced)
- **Example**: `[0.4, 0.3, 0.2, 0.1]` for 4-asset portfolio

#### Window (`window` parameter in var method)
- **Type**: int
- **Default**: 0 (uses all data)
- **Purpose**: Limit lookback period to recent history
- **Example**: 252 for trailing one year of daily data

---

## When to Use Historical VaR

### Ideal Use Cases

#### 1. Non-Normal Return Distributions

Historical VaR excels when returns exhibit:
- **Fat tails**: More extreme events than normal distribution predicts
- **Skewness**: Asymmetric return distributions
- **Kurtosis**: Excess kurtosis (leptokurtic distributions)

```python
# Example: Comparing distributions
from scipy import stats

returns = calculate_returns(prices, weights)

# Check for normality
_, p_value = stats.jarque_bera(returns)

if p_value < 0.05:
    print("Returns are not normally distributed")
    print("Historical VaR may be more appropriate than parametric VaR")
```

#### 2. Volatile Market Periods

During market stress or high volatility:
- Captures actual extreme movements
- No reliance on stability assumptions
- Reflects real correlation breakdowns

#### 3. Regulatory Compliance

Many regulatory frameworks allow or prefer historical simulation:
- Basel III permits historical VaR for market risk
- More conservative than parametric in crisis periods
- Easier to explain to regulators

#### 4. Stakeholder Communication

Non-technical audiences find it intuitive:
- "Based on past 2 years, worst 5% of days"
- No complex mathematical assumptions to explain
- Clear connection to actual events

### When to Avoid Historical VaR

#### 1. Limited Historical Data

With fewer than 100 observations:
- Percentile estimates become unreliable
- Few data points in the tail
- High sampling error

**Alternative**: Use parametric VaR or obtain more data

#### 2. Structural Market Changes

After major regime shifts:
- Pre-2008 data may not be relevant post-crisis
- COVID-19 changed many market dynamics
- Regulatory changes alter market behavior

**Solution**: Use windowed approach focusing on recent regime

#### 3. New Assets or Portfolios

For newly listed securities:
- Insufficient trading history
- No data on how they behave in various market conditions

**Alternative**: Use parametric methods with peer data

#### 4. Very High Confidence Levels

At 99.9% confidence with 1000 observations:
- Only 1 observation in the tail
- Extremely high sampling uncertainty

**Alternative**: Consider Extreme Value Theory (EVT) methods

---

## Practical Examples

### Example 1: Basic Historical VaR Calculation

```python
import numpy as np
from VaR import HistoricalVaR
import pandas_datareader.data as web

# Fetch data
stocks = ['AAPL', 'MSFT', 'GOOGL']
data = web.DataReader(stocks, 'yahoo', '2020-01-01', '2023-12-31')['Adj Close']

# Define portfolio
weights = np.array([0.5, 0.3, 0.2])
portfolio_value = 1000000

# Calculate Historical VaR
hist_var = HistoricalVaR(0.95, data.values, weights)

# Daily VaR
daily_var = hist_var.var(marketValue=portfolio_value)
print(f"Daily 95% VaR: ${daily_var:,.0f}")

# Interpretation
print(f"\nInterpretation: With 95% confidence, daily losses")
print(f"will not exceed ${daily_var:,.0f}")
```

### Example 2: Rolling Window Analysis

```python
# Compare different lookback periods
windows = [100, 252, 504, 0]  # ~4mo, 1yr, 2yr, all data
window_names = ['4 months', '1 year', '2 years', 'All data']

print("VaR by Lookback Period:")
print("-" * 50)

for window, name in zip(windows, window_names):
    var = hist_var.var(marketValue=portfolio_value, window=window)
    print(f"{name:12s}: ${var:>10,.0f}")
```

### Example 3: Comparing Methods

```python
from VaR import ValueAtRisk, HistoricalVaR

# Setup
weights = np.array([0.5, 0.5])
portfolio_value = 1000000

# Parametric VaR
param_var = ValueAtRisk(0.95, data, weights)
param_result = param_var.var(marketValue=portfolio_value, window=1)

# Historical VaR
hist_var = HistoricalVaR(0.95, data.values, weights)
hist_result = hist_var.var(marketValue=portfolio_value)

# Compare
print(f"Parametric VaR: ${param_result:,.0f}")
print(f"Historical VaR: ${hist_result:,.0f}")
print(f"Difference: {abs(param_result - hist_result) / param_result * 100:.1f}%")

if hist_result > param_result * 1.2:
    print("\nHistorical VaR is significantly higher than parametric VaR.")
    print("This suggests the presence of fat tails in the return distribution.")
```

### Example 4: Backtesting VaR Accuracy

```python
# Calculate VaR
hist_var = HistoricalVaR(0.95, prices, weights)
var_estimate = hist_var.var()

# Get portfolio returns
portfolio_returns = np.dot(hist_var.returnMatrix, weights)

# Identify exceedances (losses exceeding VaR)
exceedances = portfolio_returns < -var_estimate
num_exceedances = np.sum(exceedances)
total_observations = len(portfolio_returns)

# Expected vs Actual
expected_rate = 1 - 0.95  # 5% for 95% confidence
actual_rate = num_exceedances / total_observations

print(f"Backtesting Results:")
print(f"Expected exceedances: {expected_rate*100:.1f}% ({int(expected_rate*total_observations)} out of {total_observations})")
print(f"Actual exceedances: {actual_rate*100:.1f}% ({num_exceedances} out of {total_observations})")

# Statistical test (rough approximation)
if abs(actual_rate - expected_rate) < 0.02:
    print("✓ VaR model appears well-calibrated")
else:
    print("⚠ VaR model may need adjustment")
```

### Example 5: Stress Testing

```python
# Identify worst historical scenarios
hist_var = HistoricalVaR(0.95, prices, weights)
portfolio_returns = np.dot(hist_var.returnMatrix, weights)

# Find worst 10 days
worst_indices = np.argsort(portfolio_returns)[:10]
worst_returns = portfolio_returns[worst_indices]

print("10 Worst Historical Days:")
print("-" * 60)
for i, (idx, ret) in enumerate(zip(worst_indices, worst_returns), 1):
    loss_pct = ret * 100
    loss_dollar = ret * portfolio_value
    print(f"{i:2d}. Day {idx:3d}: {loss_pct:6.2f}% (${loss_dollar:>12,.0f})")

# Compare worst day to VaR
var_95 = hist_var.var()
worst_loss = abs(worst_returns[0])

print(f"\n95% VaR: {var_95*100:.2f}%")
print(f"Worst day: {worst_loss*100:.2f}%")
print(f"Ratio: {worst_loss/var_95:.2f}x VaR")
```

---

## Common Pitfalls

### 1. Insufficient Data

**Problem**: Using too few observations leads to unreliable estimates.

```python
# Bad: Only 50 observations
data_short = prices[-50:]
hist_var = HistoricalVaR(0.95, data_short, weights)
var = hist_var.var()  # Based on only 2-3 tail observations!
```

**Solution**: Ensure adequate sample size
```python
# Good: Check data length first
if len(prices) < 100:
    print("Warning: Insufficient data for reliable Historical VaR")
    print("Consider using parametric VaR or obtaining more data")
else:
    hist_var = HistoricalVaR(0.95, prices, weights)
```

### 2. Including Irrelevant Historical Periods

**Problem**: Using data from different market regimes

```python
# Bad: Including pre-crisis data when post-crisis regime is different
data_all = get_prices('2005-01-01', '2023-12-31')  # Includes 2008 crisis
hist_var = HistoricalVaR(0.95, data_all, weights)
```

**Solution**: Focus on relevant period
```python
# Good: Use recent data only
data_recent = get_prices('2020-01-01', '2023-12-31')
hist_var = HistoricalVaR(0.95, data_recent, weights)
```

### 3. Ignoring Non-Stationarity

**Problem**: Assuming volatility is constant over time

**Solution**: Use rolling windows and monitor VaR over time
```python
# Calculate rolling VaR to detect changes
window_size = 252
rolling_vars = []

for i in range(window_size, len(prices)):
    window_data = prices[i-window_size:i]
    hist_var = HistoricalVaR(0.95, window_data, weights)
    rolling_vars.append(hist_var.var(marketValue=1000000))

# Plot or analyze trend
import matplotlib.pyplot as plt
plt.plot(rolling_vars)
plt.title('Rolling Historical VaR')
plt.ylabel('VaR ($)')
plt.show()
```

### 4. Wrong Interpretation of Confidence Level

**Problem**: Misunderstanding what 95% confidence means

**Incorrect**: "We are 95% sure the loss will be exactly this amount"

**Correct**: "With 95% confidence, losses will not exceed this amount. On 5% of days, losses may be larger."

### 5. Overlooking Interpolation Methods

**Problem**: Different percentile calculation methods give different results

This implementation uses 'nearest' interpolation. Other methods (linear, lower, higher) may give slightly different results. Be consistent in methodology.

---

## Advanced Topics

### 1. Combining Historical and Parametric Approaches

**Hybrid VaR**: Use both methods and take a weighted average or the more conservative estimate

```python
# Calculate both
param_var = ValueAtRisk(0.95, prices, weights)
hist_var = HistoricalVaR(0.95, prices, weights)

param_result = param_var.var(marketValue=1000000, window=1)
hist_result = hist_var.var(marketValue=1000000)

# Conservative approach: take the larger
conservative_var = max(param_result, hist_result)
print(f"Conservative VaR: ${conservative_var:,.0f}")

# Weighted average approach
hybrid_var = 0.6 * hist_result + 0.4 * param_result
print(f"Hybrid VaR: ${hybrid_var:,.0f}")
```

### 2. Weighted Historical Simulation

Give more weight to recent observations (not implemented in this library, but conceptually):

```python
# Conceptual (not actual implementation)
# Apply exponential weights to recent data
lambda_factor = 0.94  # RiskMetrics standard
weights_time = np.array([lambda_factor**i for i in range(len(returns))])
weights_time = weights_time / weights_time.sum()

# Use weighted percentile (would require custom implementation)
```

### 3. Conditional VaR (CVaR) Extension

Calculate expected loss beyond VaR:

```python
# Get VaR threshold
hist_var = HistoricalVaR(0.95, prices, weights)
var_threshold = hist_var.var()

# Calculate portfolio returns
portfolio_returns = np.dot(hist_var.returnMatrix, weights)

# CVaR: average of losses exceeding VaR
losses_beyond_var = portfolio_returns[portfolio_returns < -var_threshold]
cvar = abs(np.mean(losses_beyond_var))

print(f"95% VaR: {var_threshold*100:.2f}%")
print(f"95% CVaR: {cvar*100:.2f}%")
print(f"\nIf losses exceed VaR, average loss is {cvar*100:.2f}%")
```

### 4. Multi-Period VaR

Estimating VaR for horizons longer than 1 day:

```python
# Naive approach: square root of time rule (assumes i.i.d. returns)
daily_var = hist_var.var()
weekly_var = daily_var * np.sqrt(5)
monthly_var = daily_var * np.sqrt(21)

print(f"Daily VaR: {daily_var*100:.2f}%")
print(f"Weekly VaR: {weekly_var*100:.2f}%")
print(f"Monthly VaR: {monthly_var*100:.2f}%")

# Note: This assumes returns are independent, which may not hold
```

### 5. Bootstrap Confidence Intervals

Quantify uncertainty in VaR estimates:

```python
# Bootstrap VaR estimates
n_bootstrap = 1000
bootstrap_vars = []

portfolio_returns = np.dot(hist_var.returnMatrix, weights)

for _ in range(n_bootstrap):
    # Resample with replacement
    bootstrap_sample = np.random.choice(portfolio_returns,
                                       size=len(portfolio_returns),
                                       replace=True)
    # Calculate VaR for this bootstrap sample
    bootstrap_var = abs(np.percentile(bootstrap_sample, 5, method='nearest'))
    bootstrap_vars.append(bootstrap_var)

# Confidence interval for VaR estimate
ci_lower = np.percentile(bootstrap_vars, 2.5)
ci_upper = np.percentile(bootstrap_vars, 97.5)

print(f"VaR estimate: {hist_var.var()*100:.2f}%")
print(f"95% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
```

---

## References

### Academic Papers

1. **Jorion, P. (2007)**. "Value at Risk: The New Benchmark for Managing Financial Risk" (3rd ed.). McGraw-Hill.
   - Comprehensive treatment of VaR methodologies including historical simulation

2. **Dowd, K. (2005)**. "Measuring Market Risk" (2nd ed.). Wiley.
   - Detailed comparison of parametric vs. historical VaR

3. **Hull, J. C. (2018)**. "Risk Management and Financial Institutions" (5th ed.). Wiley.
   - Practical implementation of historical simulation methods

### Regulatory Guidelines

4. **Basel Committee on Banking Supervision (2019)**. "Minimum capital requirements for market risk"
   - Standards for VaR in regulatory capital calculations

5. **European Banking Authority (2020)**. "Guidelines on market risk"
   - Regulatory perspective on historical simulation

### Industry Standards

6. **RiskMetrics (1996)**. "RiskMetrics Technical Document" (4th ed.). J.P. Morgan/Reuters.
   - Industry standard methodologies for VaR calculation

### Online Resources

7. **GARP (Global Association of Risk Professionals)**: www.garp.org
   - Professional resources and certifications

8. **Quantitative Finance Stack Exchange**: quant.stackexchange.com
   - Community discussions on VaR implementation

---

## See Also

- [README.md](../README.md) - Main project documentation
- [API_REFERENCE.md](API_REFERENCE.md) - Complete API reference
- [examples/](../examples/) - Practical code examples
- [CONTRIBUTING.md](../CONTRIBUTING.md) - How to contribute

---

**Last Updated**: 2026-02-24

For questions, issues, or contributions related to Historical VaR, please open an issue on GitHub or consult the main documentation.
