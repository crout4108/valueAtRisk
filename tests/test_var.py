"""
Unit tests for ValueAtRisk class
"""
import unittest
import numpy as np
import pandas as pd
from VaR import ValueAtRisk


class TestValueAtRisk(unittest.TestCase):
    """Test cases for ValueAtRisk class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create sample price data (10 days, 3 stocks)
        self.sample_prices = np.array([
            [100, 200, 150],
            [102, 198, 152],
            [101, 201, 151],
            [103, 199, 153],
            [105, 202, 155],
            [104, 200, 154],
            [106, 203, 156],
            [107, 201, 157],
            [108, 204, 158],
            [109, 205, 159]
        ])
        self.weights = np.array([0.4, 0.3, 0.3])
        self.confidence_interval = 0.95

    def test_initialization_valid(self):
        """Test ValueAtRisk initialization with valid inputs"""
        var = ValueAtRisk(self.confidence_interval, self.sample_prices, self.weights)
        self.assertEqual(var.ci, self.confidence_interval)
        self.assertEqual(var.weights.shape[0], 3)
        self.assertEqual(var.returnMatrix.shape[0], 9)  # One less than price matrix

    def test_initialization_with_dataframe(self):
        """Test ValueAtRisk initialization with pandas DataFrame"""
        df = pd.DataFrame(self.sample_prices, columns=['A', 'B', 'C'])
        var = ValueAtRisk(self.confidence_interval, df, self.weights)
        self.assertEqual(var.returnMatrix.shape[0], 9)

    def test_initialization_invalid_confidence_interval(self):
        """Test that invalid confidence intervals raise exceptions"""
        with self.assertRaises(Exception) as context:
            ValueAtRisk(1.5, self.sample_prices, self.weights)
        self.assertIn("Invalid confidence interval", str(context.exception))

        with self.assertRaises(Exception) as context:
            ValueAtRisk(0, self.sample_prices, self.weights)
        self.assertIn("Invalid confidence interval", str(context.exception))

        with self.assertRaises(Exception) as context:
            ValueAtRisk(-0.5, self.sample_prices, self.weights)
        self.assertIn("Invalid confidence interval", str(context.exception))

    def test_initialization_invalid_matrix_dimensions(self):
        """Test that invalid matrix dimensions raise exceptions"""
        # 1D array should fail
        invalid_matrix = np.array([100, 200, 150])
        with self.assertRaises(Exception) as context:
            ValueAtRisk(self.confidence_interval, invalid_matrix, self.weights)
        self.assertIn("Only accept 2 dimensions matrix", str(context.exception))

    def test_initialization_mismatched_weights(self):
        """Test that mismatched weights raise exceptions"""
        wrong_weights = np.array([0.5, 0.5])  # Only 2 weights for 3 stocks
        with self.assertRaises(Exception) as context:
            ValueAtRisk(self.confidence_interval, self.sample_prices, wrong_weights)
        self.assertIn("Weights Length doesn't match", str(context.exception))

    def test_cov_matrix(self):
        """Test covariance matrix calculation"""
        var = ValueAtRisk(self.confidence_interval, self.sample_prices, self.weights)
        cov_matrix = var.covMatrix()

        # Check that it returns a square matrix
        self.assertEqual(cov_matrix.shape[0], cov_matrix.shape[1])
        self.assertEqual(cov_matrix.shape[0], 3)  # 3 stocks

        # Check that covariance matrix is symmetric
        np.testing.assert_array_almost_equal(cov_matrix, cov_matrix.T)

        # Check that diagonal elements (variances) are non-negative
        for i in range(cov_matrix.shape[0]):
            self.assertGreaterEqual(cov_matrix[i, i], 0)

    def test_calculate_variance_exact(self):
        """Test variance calculation using covariance matrix method"""
        var = ValueAtRisk(self.confidence_interval, self.sample_prices, self.weights)
        variance = var.calculateVariance(Approximation=False)

        # Variance should be positive
        self.assertGreater(variance, 0)

        # Verify it's stored in the object
        self.assertEqual(var.variance, variance)

    def test_calculate_variance_approximation(self):
        """Test variance calculation using approximation method"""
        var = ValueAtRisk(self.confidence_interval, self.sample_prices, self.weights)
        variance = var.calculateVariance(Approximation=True)

        # Variance should be positive
        self.assertGreater(variance, 0)

    def test_var_percentage(self):
        """Test VaR calculation returning percentage"""
        var = ValueAtRisk(self.confidence_interval, self.sample_prices, self.weights)
        var_pct = var.var(marketValue=0, window=1)

        # VaR should be positive
        self.assertGreater(var_pct, 0)

        # Should be a reasonable percentage (less than 100%)
        self.assertLess(var_pct, 1)

    def test_var_dollar(self):
        """Test VaR calculation returning dollar amount"""
        var = ValueAtRisk(self.confidence_interval, self.sample_prices, self.weights)
        market_value = 1000000
        var_dollar = var.var(marketValue=market_value, window=1)

        # VaR should be positive
        self.assertGreater(var_dollar, 0)

        # Should be less than total market value
        self.assertLess(var_dollar, market_value)

    def test_var_window_scaling(self):
        """Test that VaR scales correctly with time window"""
        var = ValueAtRisk(self.confidence_interval, self.sample_prices, self.weights)

        # Daily VaR
        var_daily = var.var(marketValue=1000000, window=1)

        # 10-day VaR should be sqrt(10) times daily VaR
        var_10day = var.var(marketValue=1000000, window=10)

        # Check approximate scaling (using sqrt(10))
        expected_ratio = np.sqrt(10)
        actual_ratio = var_10day / var_daily

        # Allow for small numerical differences
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=5)

    def test_set_ci_valid(self):
        """Test setting confidence interval with valid value"""
        var = ValueAtRisk(self.confidence_interval, self.sample_prices, self.weights)
        var.setCI(0.99)
        self.assertEqual(var.ci, 0.99)

    def test_set_ci_invalid(self):
        """Test setting confidence interval with invalid value"""
        var = ValueAtRisk(self.confidence_interval, self.sample_prices, self.weights)

        with self.assertRaises(Exception) as context:
            var.setCI(1.5)
        self.assertIn("Invalid confidence interval", str(context.exception))

    def test_set_portfolio(self):
        """Test updating portfolio data"""
        var = ValueAtRisk(self.confidence_interval, self.sample_prices, self.weights)

        # Create new price data
        new_prices = np.array([
            [110, 210, 160],
            [112, 208, 162],
            [111, 211, 161],
            [113, 209, 163],
            [115, 212, 165]
        ])

        var.setPortfolio(new_prices)

        # Check that return matrix has been updated
        self.assertEqual(var.returnMatrix.shape[0], 4)  # One less than new price matrix

    def test_set_portfolio_with_dataframe(self):
        """Test updating portfolio data with DataFrame"""
        var = ValueAtRisk(self.confidence_interval, self.sample_prices, self.weights)

        new_prices = pd.DataFrame([
            [110, 210, 160],
            [112, 208, 162],
            [111, 211, 161]
        ], columns=['A', 'B', 'C'])

        var.setPortfolio(new_prices)

        # Check that return matrix has been updated
        self.assertEqual(var.returnMatrix.shape[0], 2)

    def test_set_weights(self):
        """Test updating portfolio weights"""
        var = ValueAtRisk(self.confidence_interval, self.sample_prices, self.weights)

        new_weights = np.array([0.5, 0.3, 0.2])
        var.setWeights(new_weights)

        np.testing.assert_array_equal(var.weights, new_weights)

    def test_set_weights_from_list(self):
        """Test updating portfolio weights from list"""
        var = ValueAtRisk(self.confidence_interval, self.sample_prices, self.weights)

        new_weights = [0.5, 0.3, 0.2]
        var.setWeights(new_weights)

        # Should convert list to numpy array
        self.assertIsInstance(var.weights, np.ndarray)
        np.testing.assert_array_equal(var.weights, np.array(new_weights))

    def test_return_matrix_calculation(self):
        """Test that return matrix is calculated correctly using log returns"""
        var = ValueAtRisk(self.confidence_interval, self.sample_prices, self.weights)

        # Manually calculate first log return for first stock
        expected_return = np.log(self.sample_prices[1, 0] / self.sample_prices[0, 0])
        actual_return = var.returnMatrix[0, 0]

        self.assertAlmostEqual(actual_return, expected_return, places=10)

    def test_var_different_confidence_levels(self):
        """Test VaR at different confidence levels"""
        var = ValueAtRisk(0.95, self.sample_prices, self.weights)
        var_95 = var.var(marketValue=1000000, window=1)

        var.setCI(0.99)
        var_99 = var.var(marketValue=1000000, window=1)

        # Higher confidence interval should give higher VaR
        self.assertGreater(var_99, var_95)


if __name__ == '__main__':
    unittest.main()
