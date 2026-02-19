# API Reference Documentation

Complete API reference for the Value at Risk (VaR) Calculator library.

## Table of Contents

- [ValueAtRisk Class](#valueatrisk-class)
- [HistoricalVaR Class](#historicalvar-class)
- [Mathematical Background](#mathematical-background)
- [Usage Patterns](#usage-patterns)
- [Error Handling](#error-handling)

---

## ValueAtRisk Class

The `ValueAtRisk` class implements parametric VaR calculation using the variance-covariance method.

### Class Definition

```python
class ValueAtRisk:
    """
    Parametric Value at Risk calculator using the variance-covariance method.
    """
```

### Constructor

```python
ValueAtRisk(interval, matrix, weights)
```

Creates a new ValueAtRisk calculator instance.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `interval` | float | Confidence interval (0 < interval < 1). Common values: 0.90, 0.95, 0.99 |
| `matrix` | ndarray or DataFrame | Stock price matrix where rows = time periods, columns = assets |
| `weights` | ndarray or array-like | Portfolio weights (must sum to 1.0) |

**Raises:**

- `Exception`: If interval is not between 0 and 1
- `Exception`: If matrix is not 2-dimensional
- `Exception`: If weights length doesn't match number of assets

**Example:**

```python
import numpy as np
from VaR import ValueAtRisk

prices = np.array([[100, 200], [102, 198], [101, 201]])
weights = np.array([0.6, 0.4])
var_calc = ValueAtRisk(0.95, prices, weights)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `ci` | float | Confidence interval |
| `input` | ndarray | Original price matrix |
| `returnMatrix` | ndarray | Log returns calculated from prices |
| `weights` | ndarray | Portfolio weights |
| `variance` | float | Portfolio variance (computed on demand) |

### Methods

#### `covMatrix()`

Calculate the variance-covariance matrix of portfolio returns.

**Returns:**
- `ndarray`: Covariance matrix of shape (n_assets, n_assets)

**Example:**
```python
cov_matrix = var_calc.covMatrix()
print(cov_matrix)
```

**Notes:**
- Returns are already log returns
- Matrix is symmetric
- Diagonal elements are variances
- Off-diagonal elements are covariances

---

#### `calculateVariance(Approximation=False)`

Calculate portfolio variance using exact or approximation method.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Approximation` | bool | False | If True, uses portfolio return approximation; if False, uses covariance matrix |

**Returns:**
- `float`: Portfolio variance

**Example:**
```python
# Exact method (recommended)
variance_exact = var_calc.calculateVariance(Approximation=False)

# Approximation method (faster for large portfolios)
variance_approx = var_calc.calculateVariance(Approximation=True)
```

**Method Details:**

- **Exact method**: σ_p² = w^T × Σ × w
  - More accurate
  - Captures correlation effects fully
  - Recommended for final calculations

- **Approximation method**: σ_p² = Var(Σ(w_i × r_i))
  - Faster for very large portfolios
  - May differ slightly from exact method
  - Useful for quick estimates

---

#### `var(marketValue=0, Approximation=False, window=252)`

Calculate parametric Value at Risk for the portfolio.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `marketValue` | float | 0 | Portfolio value in dollars. If ≤ 0, returns percentage VaR |
| `Approximation` | bool | False | Variance calculation method |
| `window` | int | 252 | Time period scaling factor (252 = annual, 1 = daily) |

**Returns:**
- `float`: VaR in dollars or percentage

**Raises:**
- `Exception`: If weights and portfolio dimensions don't match

**Example:**
```python
# Daily VaR as percentage
daily_var_pct = var_calc.var(window=1)  # Returns e.g., 0.033 (3.3%)

# Daily VaR in dollars
daily_var_dollar = var_calc.var(marketValue=1000000, window=1)  # Returns e.g., 33000

# Annual VaR (default)
annual_var = var_calc.var(marketValue=1000000)  # Returns e.g., 526333

# 30-day VaR
monthly_var = var_calc.var(marketValue=1000000, window=30)
```

**Mathematical Formula:**
```
VaR = |z_α × σ_p × √t| × V
```

Where:
- z_α: Critical value at confidence level (e.g., 1.645 for 95%)
- σ_p: Portfolio standard deviation
- t: Time horizon (window parameter)
- V: Market value

**Window Parameter Guide:**

| Window | Period | Use Case |
|--------|--------|----------|
| 1 | Daily | Day-to-day risk management |
| 5 | Weekly | Short-term planning |
| 21 | Monthly | Monthly risk reports |
| 63 | Quarterly | Quarterly assessments |
| 252 | Annual | Annual risk budgets |

---

#### `setCI(interval)`

Update the confidence interval for VaR calculation.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `interval` | float | New confidence interval (0 < interval < 1) |

**Raises:**
- `Exception`: If interval is not between 0 and 1

**Example:**
```python
var_calc.setCI(0.99)  # Change to 99% confidence
new_var = var_calc.var(marketValue=1000000)
```

---

#### `setPortfolio(matrix)`

Update the portfolio price data.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix` | ndarray or DataFrame | New stock price matrix |

**Raises:**
- `Exception`: If matrix is not 2-dimensional

**Example:**
```python
# Update with new data
new_prices = fetch_latest_prices()
var_calc.setPortfolio(new_prices)
updated_var = var_calc.var(marketValue=1000000)
```

**Notes:**
- Weights remain unchanged
- Returns are automatically recalculated
- Useful for rolling calculations

---

#### `setWeights(weights)`

Update the portfolio weights.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `weights` | ndarray or array-like | New portfolio weights |

**Example:**
```python
# Rebalance portfolio
new_weights = np.array([0.3, 0.7])
var_calc.setWeights(new_weights)
rebalanced_var = var_calc.var(marketValue=1000000)
```

**Notes:**
- Price data remains unchanged
- Weights should sum to 1.0 (not enforced but recommended)
- Allows testing different allocations on same data

---

## HistoricalVaR Class

The `HistoricalVaR` class extends `ValueAtRisk` to implement historical simulation VaR.

### Class Definition

```python
class HistoricalVaR(ValueAtRisk):
    """
    Historical Value at Risk calculator using historical simulation method.
    """
```

### Constructor

```python
HistoricalVaR(interval, matrix, weights)
```

Same as `ValueAtRisk` constructor. Inherits all initialization parameters and validation.

### Inherited Methods

All methods from `ValueAtRisk` are available:
- `covMatrix()`
- `calculateVariance()`
- `setCI()`
- `setPortfolio()`
- `setWeights()`

### Overridden Methods

#### `var(marketValue=0, window=0)`

Calculate historical VaR using percentile method.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `marketValue` | float | 0 | Portfolio value in dollars. If ≤ 0, returns percentage VaR |
| `window` | int | 0 | Look-back period. If 0, uses entire history |

**Returns:**
- `float`: VaR in dollars or percentage

**Raises:**
- `Exception`: If window exceeds available data length

**Example:**
```python
from VaR import HistoricalVaR

hist_var = HistoricalVaR(0.95, prices, weights)

# VaR using all historical data
var_full = hist_var.var(marketValue=1000000)

# VaR using last 100 days
var_100d = hist_var.var(marketValue=1000000, window=100)

# VaR as percentage
var_pct = hist_var.var() * 100
```

**Method Details:**

1. Calculates weighted portfolio returns: R_p = Σ(w_i × r_i)
2. Finds (1-α) percentile of returns
3. Takes absolute value (represents loss)
4. Scales by market value if provided

**Window Parameter Guide:**

| Window | Use Case |
|--------|----------|
| 0 | All available history (default) |
| 100 | Recent market conditions |
| 252 | Trailing 1-year window |
| 504 | Trailing 2-year window |

**Comparison with Parametric VaR:**

| Aspect | Parametric | Historical |
|--------|-----------|------------|
| Distribution | Assumes normal | Uses actual data |
| Tail risk | May underestimate | Captures actual tails |
| Data requirements | Less sensitive | Needs sufficient history |
| Computation | Fast | Fast |
| Interpretation | Theoretical | Empirical |

---

## Mathematical Background

### Log Returns

Both classes use log returns rather than simple returns:

```
r_t = ln(P_t / P_{t-1})
```

**Advantages:**
- Time-additive
- Symmetric
- Better statistical properties
- Standard in financial modeling

### Parametric VaR Formula

```
VaR = |z_α × σ_p × √t| × V
```

Where:
- z_α = norm.ppf(confidence_level)
- σ_p = √(w^T × Σ × w)
- t = time horizon
- V = portfolio value

### Historical VaR Formula

```
VaR = -Percentile(R_p, 100×(1-α)) × V
```

Where:
- R_p = portfolio returns
- α = confidence level
- Percentile uses nearest interpolation

### Time Scaling

VaR scales with square root of time:

```
VaR_t = VaR_1 × √t
```

Example:
- Daily VaR = $10,000
- Weekly VaR ≈ $10,000 × √5 ≈ $22,361
- Annual VaR ≈ $10,000 × √252 ≈ $158,745

---

## Usage Patterns

### Pattern 1: Basic VaR Calculation

```python
from VaR import ValueAtRisk
import numpy as np

# Setup
prices = fetch_prices(['AAPL', 'GOOGL'])
weights = np.array([0.6, 0.4])
var_calc = ValueAtRisk(0.95, prices, weights)

# Calculate
daily_var = var_calc.var(marketValue=1000000, window=1)
print(f"Daily VaR: ${daily_var:,.0f}")
```

### Pattern 2: Comparing Methods

```python
from VaR import ValueAtRisk, HistoricalVaR

# Parametric
param_var = ValueAtRisk(0.95, prices, weights)
param_result = param_var.var(marketValue=1000000, window=1)

# Historical
hist_var = HistoricalVaR(0.95, prices, weights)
hist_result = hist_var.var(marketValue=1000000, window=100)

print(f"Parametric: ${param_result:,.0f}")
print(f"Historical: ${hist_result:,.0f}")
```

### Pattern 3: Portfolio Optimization

```python
# Test different allocations
allocations = [
    np.array([0.5, 0.5]),
    np.array([0.3, 0.7]),
    np.array([0.7, 0.3])
]

results = {}
for weights in allocations:
    var_calc = ValueAtRisk(0.95, prices, weights)
    results[tuple(weights)] = var_calc.var(marketValue=1000000)

# Find minimum VaR allocation
best_weights = min(results, key=results.get)
```

### Pattern 4: Rolling VaR

```python
# Calculate VaR over time
window_size = 252
rolling_vars = []

for i in range(window_size, len(prices)):
    window_prices = prices[i-window_size:i]
    var_calc = ValueAtRisk(0.95, window_prices, weights)
    rolling_vars.append(var_calc.var(marketValue=1000000, window=1))
```

---

## Error Handling

### Common Errors and Solutions

#### Invalid Confidence Interval

```python
# Error
var_calc = ValueAtRisk(1.5, prices, weights)
# Exception: Invalid confidence interval

# Solution
var_calc = ValueAtRisk(0.95, prices, weights)  # Use 0 < interval < 1
```

#### Dimension Mismatch

```python
# Error
prices = np.array([[100, 200], [102, 198]])  # 2 assets
weights = np.array([0.5, 0.3, 0.2])  # 3 weights
# Exception: Weights Length doesn't match

# Solution
weights = np.array([0.6, 0.4])  # Match number of assets
```

#### Invalid Matrix

```python
# Error
prices = np.array([100, 102, 101])  # 1D array
# Exception: Only accept 2 dimensions matrix

# Solution
prices = np.array([[100], [102], [101]])  # 2D array
```

#### Window Too Large

```python
# Error
hist_var = HistoricalVaR(0.95, prices, weights)
result = hist_var.var(window=1000)  # More than available data
# Exception: invalid Window, cannot excess

# Solution
result = hist_var.var(window=100)  # Use reasonable window
```

### Best Practices

1. **Validate inputs before creating objects**
   ```python
   assert 0 < confidence < 1, "Confidence must be between 0 and 1"
   assert prices.ndim == 2, "Prices must be 2D array"
   assert len(weights) == prices.shape[1], "Weights must match assets"
   ```

2. **Handle data fetching errors**
   ```python
   try:
       data = web.DataReader(stocks, 'yahoo', start, end)
   except Exception as e:
       print(f"Error fetching data: {e}")
       return None
   ```

3. **Check for sufficient data**
   ```python
   min_periods = 100
   if len(prices) < min_periods:
       print(f"Warning: Only {len(prices)} periods available")
   ```

4. **Validate weights sum to 1**
   ```python
   if not np.isclose(weights.sum(), 1.0):
       print(f"Warning: Weights sum to {weights.sum()}, should be 1.0")
   ```

---

## See Also

- [README.md](../README.md) - General documentation
- [examples/](../examples/) - Usage examples
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
