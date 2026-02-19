"""
Parametric Value at Risk (VaR) Calculation Module

This module implements parametric VaR calculation using the variance-covariance method,
which assumes that portfolio returns follow a normal distribution. The VaR measure quantifies
the maximum expected loss over a given time horizon at a specified confidence level.

Mathematical Foundation:
    VaR = -z_α × σ_p × √t × V

    Where:
        - z_α: Critical value at confidence level α (e.g., 1.645 for 95%)
        - σ_p: Portfolio standard deviation (volatility)
        - t: Time horizon (e.g., 252 trading days for annual)
        - V: Portfolio market value

Portfolio Variance:
    σ_p² = w^T × Σ × w

    Where:
        - w: Portfolio weight vector
        - Σ: Covariance matrix of asset returns

Created on Wed Sep 2 18:01:00 2020
@author: Farshad Noravesh
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
import math

class ValueAtRisk:
    """
    Parametric Value at Risk calculator using the variance-covariance method.

    This class calculates VaR by assuming portfolio returns follow a normal distribution.
    It uses the variance-covariance matrix approach to compute portfolio risk metrics.

    Attributes:
        ci (float): Confidence interval for VaR calculation (e.g., 0.95 for 95%)
        input (ndarray): Original price matrix for the portfolio
        returnMatrix (ndarray): Log returns calculated from price matrix
        weights (ndarray): Portfolio weights for each asset
        variance (float): Portfolio variance (computed on demand)
    """

    def __init__(self, interval, matrix, weights):
        """
        Initialize the ValueAtRisk calculator.

        Args:
            interval (float): Confidence interval for VaR calculation. Must be between 0 and 1
                            (e.g., 0.95 for 95% confidence, 0.99 for 99% confidence).
            matrix (ndarray or DataFrame): Stock price matrix where:
                                          - Rows represent time periods (dates)
                                          - Columns represent different assets/tickers
                                          Must be a 2-dimensional array.
            weights (ndarray or array-like): Portfolio weights for each asset.
                                            Must sum to 1.0 and have length equal to
                                            number of columns in matrix.

        Raises:
            Exception: If confidence interval is not between 0 and 1
            Exception: If matrix is not 2-dimensional
            Exception: If weights length doesn't match number of assets

        Example:
            >>> import numpy as np
            >>> prices = np.array([[100, 200], [102, 198], [101, 201]])
            >>> weights = np.array([0.6, 0.4])
            >>> var_calc = ValueAtRisk(0.95, prices, weights)
        """
        if(interval > 0 and interval < 1):
            self.ci = interval
        else:
            raise Exception("Invalid confidence interval", interval)

        if(isinstance(matrix,pd.DataFrame)):
            matrix = matrix.values

        if(matrix.ndim!=2):
            raise Exception("Only accept 2 dimensions matrix", matrix.ndim)

        if(len(weights)!= matrix.shape[1]):
            raise Exception("Weights Length doesn't match")

        self.input = matrix
        # Log return calculation: r_t = ln(P_t / P_{t-1})
        # This is preferred over simple returns for better statistical properties
        self.returnMatrix = np.diff(np.log(self.input),axis=0)
        if (not isinstance(weights, np.ndarray)):
            self.weights = np.array(weights)
        else:
            self.weights = weights

    def covMatrix(self):
        """
        Calculate the variance-covariance matrix of portfolio returns.

        The covariance matrix captures the relationships between different assets
        in the portfolio, including both individual asset variances (diagonal elements)
        and covariances between asset pairs (off-diagonal elements).

        Returns:
            ndarray: Variance-covariance matrix of shape (n_assets, n_assets)
                    where n_assets is the number of assets in the portfolio.

        Example:
            >>> var_calc = ValueAtRisk(0.95, prices, weights)
            >>> cov_matrix = var_calc.covMatrix()
            >>> print(cov_matrix.shape)  # (n_assets, n_assets)
        """
        return np.cov(self.returnMatrix.T)

    def calculateVariance(self, Approximation = False):
        """
        Calculate portfolio variance using either exact or approximation method.

        Portfolio variance can be calculated in two ways:
        1. Exact method (Approximation=False): Uses the formula σ_p² = w^T × Σ × w
           where Σ is the covariance matrix and w is the weight vector.
        2. Approximation method (Approximation=True): Calculates weighted portfolio
           returns first, then computes their variance.

        Args:
            Approximation (bool, optional): If True, uses portfolio return approximation.
                                          If False, uses covariance matrix method.
                                          Defaults to False.

        Returns:
            float: Portfolio variance (annualized if using daily returns)

        Example:
            >>> var_calc = ValueAtRisk(0.95, prices, weights)
            >>> variance_exact = var_calc.calculateVariance(Approximation=False)
            >>> variance_approx = var_calc.calculateVariance(Approximation=True)
        """
        if(Approximation == True):
            # Approximation: Calculate portfolio returns then compute variance
            self.variance = np.var(np.dot(self.returnMatrix,self.weights))
        else:
            # Exact: Use quadratic form w^T × Σ × w
            self.variance = np.dot(np.dot(self.weights,np.cov(self.returnMatrix.T)),self.weights.T)
        return self.variance


    def var(self,marketValue = 0,Approximation = False,window = 252):
        """
        Calculate parametric Value at Risk (VaR) for the portfolio.

        VaR represents the maximum expected loss over a given time horizon at a
        specified confidence level, assuming returns follow a normal distribution.

        The calculation follows the formula:
            VaR = |z_α × σ_p × √t| × V

        Where:
            - z_α: Critical value from standard normal distribution at confidence level
            - σ_p: Portfolio standard deviation (from variance)
            - t: Time horizon (window parameter)
            - V: Portfolio market value

        Args:
            marketValue (float, optional): Portfolio value in dollars. If <= 0, returns
                                         percentage VaR. Defaults to 0.
            Approximation (bool, optional): Variance calculation method. If True, uses
                                          portfolio return approximation. If False, uses
                                          covariance matrix method. Defaults to False.
            window (int, optional): Time period scaling factor. Defaults to 252 (trading days
                                   in a year for annualized VaR). Use 1 for daily VaR.

        Returns:
            float: Value at Risk in dollars (if marketValue > 0) or as a percentage/decimal
                  (if marketValue <= 0)

        Raises:
            Exception: If weights and portfolio dimensions don't match

        Example:
            >>> var_calc = ValueAtRisk(0.95, prices, weights)
            >>> # Daily VaR as percentage
            >>> daily_var_pct = var_calc.var(window=1) * 100
            >>> # Daily VaR in dollars for $1M portfolio
            >>> daily_var_dollar = var_calc.var(marketValue=1000000, window=1)
            >>> # Annualized VaR (default)
            >>> annual_var = var_calc.var(marketValue=1000000)
        """
        if(self.returnMatrix.shape[1] != len(self.weights)):
            raise Exception("The weights and portfolio doesn't match")
        self.calculateVariance(Approximation)
        if(marketValue <= 0):
            # Return VaR as a decimal percentage (e.g., 0.03 for 3%)
            return abs(norm.ppf(self.ci)*np.sqrt(self.variance))*math.sqrt(window)
        else:
            # Return VaR in dollar amount
            return abs(norm.ppf(self.ci)*np.sqrt(self.variance))*marketValue*math.sqrt(window)

    def setCI(self, interval):
        """
        Update the confidence interval for VaR calculation.

        Args:
            interval (float): New confidence interval. Must be between 0 and 1
                            (e.g., 0.95 for 95% confidence, 0.99 for 99% confidence).

        Raises:
            Exception: If confidence interval is not between 0 and 1

        Example:
            >>> var_calc = ValueAtRisk(0.95, prices, weights)
            >>> var_calc.setCI(0.99)  # Change to 99% confidence
        """
        if(interval > 0 and interval < 1):
            self.ci = interval
        else:
            raise Exception("Invalid confidence interval", interval)

    def setPortfolio(self, matrix):
        """
        Update the portfolio price data.

        This method allows you to change the underlying price data for the portfolio
        while maintaining the same weights and confidence interval. Returns are
        automatically recalculated from the new price data.

        Args:
            matrix (ndarray or DataFrame): New stock price matrix where:
                                          - Rows represent time periods (dates)
                                          - Columns represent different assets/tickers
                                          Must be a 2-dimensional array with the same
                                          number of columns as the current portfolio.

        Raises:
            Exception: If matrix is not 2-dimensional

        Example:
            >>> var_calc = ValueAtRisk(0.95, old_prices, weights)
            >>> var_calc.setPortfolio(new_prices)  # Update with new data
        """
        if (isinstance(matrix, pd.DataFrame)):
            matrix = matrix.values

        if (matrix.ndim != 2):
            raise Exception("Only accept 2 dimensions matrix", matrix.ndim)

        self.input = matrix
        # Recalculate log returns from new price data
        self.returnMatrix = np.diff(np.log(self.input), axis=0)

    def setWeights(self, weights):
        """
        Update the portfolio weights.

        This method allows you to change the allocation of assets in your portfolio
        while maintaining the same price data and confidence interval.

        Args:
            weights (ndarray or array-like): New portfolio weights for each asset.
                                            Should sum to 1.0 and have length equal to
                                            the number of assets in the portfolio.

        Example:
            >>> var_calc = ValueAtRisk(0.95, prices, [0.5, 0.5])
            >>> var_calc.setWeights([0.6, 0.4])  # Change allocation
            >>> new_var = var_calc.var(marketValue=1000000)
        """
        if (not isinstance(weights, np.ndarray)):
            self.weights = np.array(weights)
        else:
            self.weights = weights
