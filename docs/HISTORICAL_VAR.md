# Historical Value at Risk (VaR)

A comprehensive guide to the historical simulation method for calculating Value at Risk.

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Mathematical Foundation](#mathematical-foundation)
- [Choosing a Window Size](#choosing-a-window-size)
- [Advantages and Limitations](#advantages-and-limitations)
- [Usage Guide](#usage-guide)
- [Comparison with Parametric VaR](#comparison-with-parametric-var)

---

## Overview

Historical Value at Risk (VaR) estimates the maximum expected loss of a portfolio over a given time horizon at a specified confidence level, using only the **empirical distribution of past returns**. Unlike the parametric (variance-covariance) method, historical VaR makes **no assumptions** about the shape of the return distribution.

**Core idea:** rank all historical daily portfolio returns from worst to best and read off the loss at the desired percentile.

For a 95% confidence level, historical VaR answers:

> *"Based on past returns, what daily loss will be exceeded only 5% of the time?"*

---

## How It Works

The calculation follows three steps:

1. **Compute log returns** for each asset and each day:

   ```
   r_t = ln(P_t / P_{t-1})
   ```

2. **Compute portfolio returns** by applying the portfolio weights:

   ```
   R_p,t = Σ (w_i × r_i,t)
   ```

3. **Find the percentile** that corresponds to the loss tail:

   ```
   VaR = |Percentile(R_p, 100 × (1 − α))|
   ```

   Multiply by the portfolio market value to obtain a dollar figure:

   ```
   VaR ($) = VaR (%) × V
   ```

Where:
- `α` is the confidence level (e.g. 0.95)
- `V` is the portfolio market value
- `w_i` are the portfolio weights

---

## Mathematical Foundation

### Log Returns vs Simple Returns

The library uses **log returns** rather than simple returns because they are:

- **Time-additive**: multi-period returns are the sum of single-period log returns
- **Symmetric**: percentage gains and losses are treated symmetrically
- **Better behaved statistically**: more compatible with normal-distribution approximations when needed

### Percentile Calculation

The (1 − α) percentile of the portfolio return series is computed with NumPy's `nearest` interpolation, meaning the result is always an observed return from the historical data set—no extrapolation occurs.

### No Time-Scaling

Historical VaR does **not** apply the square-root-of-time rule. The result already reflects the actual single-period (daily) risk embedded in the historical data. If you want a multi-period estimate, combine longer data windows rather than scaling.

---

## Choosing a Window Size

The `window` parameter of `HistoricalVaR.var()` controls how many of the most recent trading days are used. Setting `window=0` (the default) uses the entire available price history.

| Window | Approximate Period | Typical Use Case |
|--------|--------------------|-----------------|
| 0 | All available data | Baseline / regulatory capital |
| 60 | ~3 months | Recent market stress |
| 100 | ~5 months | Short-to-medium term risk |
| 252 | ~1 year | Standard trading-year window |
| 504 | ~2 years | Captures a full market cycle |

**Guidelines:**

- **Shorter windows** react quickly to changing market conditions but are more sensitive to outliers and may miss rare events.
- **Longer windows** produce more stable estimates and capture extreme historical scenarios, but may include market regimes that are no longer relevant.
- A window must be smaller than the number of available return observations; otherwise an exception is raised.

---

## Advantages and Limitations

### Advantages

| Advantage | Description |
|-----------|-------------|
| **No distribution assumption** | Works for fat-tailed, skewed, or multimodal return distributions |
| **Captures historical extremes** | Automatically reflects real crashes, crises, and volatility clusters |
| **Simple interpretation** | VaR is a directly observed historical loss, not a model output |
| **Incorporates correlations** | Correlation between assets is implicitly included in portfolio returns |

### Limitations

| Limitation | Description |
|------------|-------------|
| **History dependence** | Quality of the estimate is limited by the length and relevance of the historical data |
| **Backward-looking** | Cannot capture risks from events that have not occurred yet |
| **Window sensitivity** | Results can change materially depending on the chosen look-back period |
| **Ghost effects** | A single extreme event dominates the tail until it rolls out of the window |

---

## Usage Guide

### Basic Usage

```python
import numpy as np
from VaR import HistoricalVaR

# Price matrix: rows = time periods, columns = assets
prices  = np.array([
    [100.0, 200.0, 150.0],
    [101.5, 198.0, 151.2],
    [102.0, 201.0, 149.8],
    # ... more rows
])
weights = np.array([0.4, 0.35, 0.25])  # Must sum to 1

hist_var = HistoricalVaR(interval=0.95, matrix=prices, weights=weights)

# VaR as a decimal (e.g. 0.033 means 3.3%)
var_pct = hist_var.var()

# VaR in dollars for a $1 million portfolio
var_dollar = hist_var.var(marketValue=1_000_000)

print(f"Daily VaR (95%): {var_pct * 100:.2f}%  /  ${var_dollar:,.0f}")
```

### Rolling-Window VaR

```python
# Use only the most recent 100 trading days
var_100d = hist_var.var(marketValue=1_000_000, window=100)

# Compare against the full-history estimate
var_full = hist_var.var(marketValue=1_000_000)

print(f"Full history VaR : ${var_full:,.0f}")
print(f"100-day VaR      : ${var_100d:,.0f}")
```

### Changing the Confidence Level

```python
for ci in [0.90, 0.95, 0.99]:
    hist_var.setCI(ci)
    print(f"  {ci*100:.0f}% VaR: ${hist_var.var(marketValue=1_000_000):,.0f}")
```

### Pandas DataFrame Input

```python
import pandas as pd
from VaR import HistoricalVaR

df = pd.DataFrame(prices, columns=["AAPL", "MSFT", "GOOGL"])
hist_var = HistoricalVaR(0.95, df, np.array([0.5, 0.3, 0.2]))
print(hist_var.var(marketValue=1_000_000))
```

### Fetching Real Market Data

```python
import pandas_datareader.data as web
import numpy as np
from VaR import HistoricalVaR

stocks  = ["GOOG", "AMZN", "TSLA"]
data    = web.DataReader(stocks, "yahoo", start="2020-01-01", end="2023-12-31")["Adj Close"]
weights = np.array([0.4, 0.35, 0.25])

hist_var = HistoricalVaR(0.95, data.values, weights)
print(f"Historical VaR (95%, daily): ${hist_var.var(marketValue=1_000_000):,.0f}")
```

---

## Comparison with Parametric VaR

| Aspect | Parametric (Variance-Covariance) | Historical Simulation |
|--------|----------------------------------|-----------------------|
| **Distribution** | Assumes normal returns | Uses actual empirical distribution |
| **Tail risk** | May underestimate fat tails | Directly reflects historical extremes |
| **Data requirement** | Covariance matrix (can work with limited data) | Needs a sufficiently long history |
| **Computation** | Closed-form, very fast | Simple percentile lookup, also fast |
| **Time scaling** | Uses √t rule | Not applicable (period is fixed by window) |
| **Sensitivity to outliers** | Low (smoothed by covariance estimate) | High (one bad day directly moves the percentile) |
| **Best for** | Stable markets, normally distributed assets | Portfolios with non-normal or fat-tailed returns |

```python
from VaR import ValueAtRisk, HistoricalVaR

param_var  = ValueAtRisk(0.95, prices, weights)
hist_var   = HistoricalVaR(0.95, prices, weights)

print(f"Parametric VaR (daily): ${param_var.var(marketValue=1_000_000, window=1):,.0f}")
print(f"Historical VaR (daily): ${hist_var.var(marketValue=1_000_000):,.0f}")
```

---

## See Also

- [API Reference](API_REFERENCE.md) — Full method signatures and parameters
- [Examples](../examples/historical_var_example.py) — Runnable historical VaR example
- [README](../README.md) — Project overview and quick start
