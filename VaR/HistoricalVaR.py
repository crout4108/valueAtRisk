"""
Historical Value at Risk (VaR) Calculation Module

This module implements historical VaR calculation using the historical simulation method.
Unlike parametric VaR, historical VaR makes no assumptions about the distribution of returns
and instead uses the empirical distribution of actual historical returns.

Mathematical Foundation:
    VaR = -Percentile(returns, 1-α) × V

    Where:
        - Percentile: The (1-α)th percentile of historical portfolio returns
        - α: Confidence level (e.g., 0.95 for 95%)
        - V: Portfolio market value

Key Advantages:
    - No distributional assumptions required
    - Captures actual historical market behavior
    - Accounts for fat tails and skewness in returns
    - Simple and intuitive interpretation

Key Limitations:
    - Limited by availability and relevance of historical data
    - Assumes future returns will behave like historical returns
    - May not capture unprecedented events

Created on Wed Sep 2 18:01:00 2020
@author: Farshad Noravesh
"""
from VaR import ValueAtRisk
import numpy as np

class HistoricalVaR(ValueAtRisk):
    """
    Historical Value at Risk calculator using the historical simulation method.

    This class extends ValueAtRisk and overrides the var() method to calculate VaR
    using actual historical returns rather than assuming a normal distribution.
    The calculation uses percentile-based approach on weighted portfolio returns.

    Inherits all attributes and methods from ValueAtRisk, including:
        - ci: Confidence interval
        - input: Price matrix
        - returnMatrix: Log returns
        - weights: Portfolio weights
        - covMatrix(), calculateVariance(), setCI(), setPortfolio(), setWeights()
    """

    def var(self, marketValue=0, window=0):
        """
        Calculate historical Value at Risk (VaR) for the portfolio.

        This method computes VaR by finding the appropriate percentile of the
        empirical distribution of historical portfolio returns. Unlike parametric VaR,
        this approach makes no assumptions about the distribution of returns.

        The calculation process:
        1. Compute weighted portfolio returns for each historical period
        2. Sort returns to find the (1-confidence_level) percentile
        3. Scale by portfolio value if provided

        Args:
            marketValue (float, optional): Portfolio value in dollars. If <= 0, returns
                                         percentage VaR as a decimal. Defaults to 0.
            window (int, optional): Look-back period for VaR calculation. If 0, uses
                                   entire price series. If > 0, uses only the most
                                   recent 'window' periods. Defaults to 0.

        Returns:
            float: Value at Risk in dollars (if marketValue > 0) or as a percentage/decimal
                  (if marketValue <= 0). The value represents the maximum expected loss
                  at the specified confidence level.

        Raises:
            Exception: If window exceeds the length of available return data

        Example:
            >>> hist_var = HistoricalVaR(0.95, prices, weights)
            >>> # VaR using all historical data
            >>> var_full = hist_var.var(marketValue=1000000)
            >>> # VaR using only last 100 days
            >>> var_100d = hist_var.var(marketValue=1000000, window=100)
            >>> # VaR as percentage
            >>> var_pct = hist_var.var() * 100

        Notes:
            - The method uses numpy's 'nearest' interpolation for percentile calculation
            - Returns are already log returns from the parent class initialization
            - VaR is returned as an absolute (positive) value representing potential loss
        """
        # Calculate weighted portfolio returns: R_p = Σ(w_i × r_i)
        self.portfolioReturn = np.dot(self.returnMatrix,self.weights)
        if(window >len(self.portfolioReturn)+1 ):
            raise  Exception("invalid Window, cannot excess", len(self.portfolioReturn))

        if(window > 0 and window < len(self.portfolioReturn)):
            # Use only the most recent 'window' periods
            PercentageVaR = abs(np.percentile(self.portfolioReturn[-window:],100*(1-self.ci),method = 'nearest'))
        else:
            # Use all available historical data
            PercentageVaR = abs(np.percentile(self.portfolioReturn,100*(1-self.ci),method = 'nearest'))

        if(marketValue <= 0):
            # Return VaR as a decimal (e.g., 0.033 for 3.3% loss)
            return PercentageVaR
        else:
            # Return VaR in dollar terms
            return PercentageVaR * marketValue
