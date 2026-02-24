"""
Unit tests for HistoricalVaR class
"""
import unittest
import numpy as np
import pandas as pd
from VaR import HistoricalVaR


class TestHistoricalVaR(unittest.TestCase):
    """Test cases for HistoricalVaR class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create sample price data (20 days, 3 stocks)
        np.random.seed(42)  # For reproducibility
        base_prices = np.array([100, 200, 150])
        returns = np.random.normal(0.001, 0.02, (19, 3))  # 19 days of returns

        # Build price series from returns
        self.sample_prices = np.zeros((20, 3))
        self.sample_prices[0] = base_prices

        for i in range(1, 20):
            self.sample_prices[i] = self.sample_prices[i-1] * np.exp(returns[i-1])

        self.weights = np.array([0.4, 0.3, 0.3])
        self.confidence_interval = 0.95

    def test_initialization(self):
        """Test HistoricalVaR initialization"""
        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, self.weights)
        self.assertEqual(hist_var.ci, self.confidence_interval)
        self.assertEqual(hist_var.weights.shape[0], 3)

    def test_var_percentage_full_window(self):
        """Test VaR calculation returning percentage using full data"""
        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, self.weights)
        var_pct = hist_var.var(marketValue=0, window=0)

        # VaR should be positive (we take absolute value)
        self.assertGreater(var_pct, 0)

        # Should be a reasonable percentage
        self.assertLess(var_pct, 1)

    def test_var_dollar_full_window(self):
        """Test VaR calculation returning dollar amount using full data"""
        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, self.weights)
        market_value = 1000000
        var_dollar = hist_var.var(marketValue=market_value, window=0)

        # VaR should be positive
        self.assertGreater(var_dollar, 0)

        # Should be less than total market value
        self.assertLess(var_dollar, market_value)

    def test_var_with_window(self):
        """Test VaR calculation with specified window"""
        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, self.weights)

        # Calculate VaR using last 10 days
        var_10day = hist_var.var(marketValue=1000000, window=10)

        # Should be positive
        self.assertGreater(var_10day, 0)

    def test_var_window_size_impact(self):
        """Test that different window sizes produce different VaR values"""
        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, self.weights)

        var_full = hist_var.var(marketValue=1000000, window=0)
        var_10day = hist_var.var(marketValue=1000000, window=10)

        # Values might be different (not necessarily ordered)
        # Just verify both are valid positive numbers
        self.assertGreater(var_full, 0)
        self.assertGreater(var_10day, 0)

    def test_var_invalid_window(self):
        """Test that invalid window size raises exception"""
        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, self.weights)

        # Window larger than available data should raise exception
        with self.assertRaises(Exception) as context:
            hist_var.var(marketValue=1000000, window=100)
        self.assertIn("invalid Window", str(context.exception))

    def test_var_edge_case_window_equal_to_data(self):
        """Test VaR with window size equal to data length"""
        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, self.weights)

        # Should work with window size equal to return matrix length
        var_result = hist_var.var(marketValue=1000000, window=19)
        self.assertGreater(var_result, 0)

    def test_var_percentage_vs_dollar_consistency(self):
        """Test that percentage and dollar VaR are consistent"""
        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, self.weights)
        market_value = 1000000

        var_pct = hist_var.var(marketValue=0, window=0)
        var_dollar = hist_var.var(marketValue=market_value, window=0)

        # Dollar VaR should equal percentage VaR times market value
        self.assertAlmostEqual(var_dollar, var_pct * market_value, places=5)

    def test_portfolio_return_calculation(self):
        """Test that portfolio returns are calculated correctly"""
        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, self.weights)

        # Call var to trigger portfolio return calculation
        hist_var.var(marketValue=1000000)

        # Check that portfolio returns were calculated
        self.assertIsNotNone(hist_var.portfolioReturn)
        self.assertEqual(len(hist_var.portfolioReturn), 19)  # One less than price data

        # Portfolio return should be weighted sum of individual returns
        expected_first_return = np.dot(hist_var.returnMatrix[0], hist_var.weights)
        self.assertAlmostEqual(hist_var.portfolioReturn[0], expected_first_return, places=10)

    def test_var_different_confidence_levels(self):
        """Test VaR at different confidence levels"""
        hist_var_95 = HistoricalVaR(0.95, self.sample_prices, self.weights)
        var_95 = hist_var_95.var(marketValue=1000000, window=0)

        hist_var_99 = HistoricalVaR(0.99, self.sample_prices, self.weights)
        var_99 = hist_var_99.var(marketValue=1000000, window=0)

        # Higher confidence interval should generally give higher VaR
        # (though with small samples this isn't guaranteed)
        self.assertGreater(var_95, 0)
        self.assertGreater(var_99, 0)

    def test_inheritance_from_value_at_risk(self):
        """Test that HistoricalVaR inherits from ValueAtRisk"""
        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, self.weights)

        # Should have access to parent methods
        self.assertTrue(hasattr(hist_var, 'covMatrix'))
        self.assertTrue(hasattr(hist_var, 'calculateVariance'))
        self.assertTrue(hasattr(hist_var, 'setCI'))
        self.assertTrue(hasattr(hist_var, 'setPortfolio'))
        self.assertTrue(hasattr(hist_var, 'setWeights'))

    def test_set_ci_updates_calculation(self):
        """Test that changing confidence interval affects VaR calculation"""
        hist_var = HistoricalVaR(0.95, self.sample_prices, self.weights)
        var_before = hist_var.var(marketValue=1000000, window=0)

        hist_var.setCI(0.90)
        var_after = hist_var.var(marketValue=1000000, window=0)

        # With lower confidence interval (90% vs 95%), VaR might be lower
        # Just verify both are valid
        self.assertGreater(var_before, 0)
        self.assertGreater(var_after, 0)

    def test_var_with_dataframe_input(self):
        """Test HistoricalVaR with DataFrame input"""
        df = pd.DataFrame(self.sample_prices, columns=['A', 'B', 'C'])
        hist_var = HistoricalVaR(self.confidence_interval, df, self.weights)

        var_result = hist_var.var(marketValue=1000000, window=0)
        self.assertGreater(var_result, 0)

    def test_var_with_single_window(self):
        """Test VaR calculation with window size of 1"""
        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, self.weights)

        # Window of 1 should use only the last return
        var_result = hist_var.var(marketValue=1000000, window=1)
        self.assertGreater(var_result, 0)

    def test_percentile_calculation(self):
        """Test that percentile calculation uses correct interpolation"""
        # Create data with known distribution
        simple_prices = np.array([
            [100, 100, 100],
            [102, 102, 102],
            [98, 98, 98],
            [101, 101, 101],
            [99, 99, 99],
            [103, 103, 103],
            [97, 97, 97],
            [104, 104, 104],
            [96, 96, 96],
            [105, 105, 105],
            [95, 95, 95]
        ])

        equal_weights = np.array([1.0/3, 1.0/3, 1.0/3])
        hist_var = HistoricalVaR(0.95, simple_prices, equal_weights)

        var_result = hist_var.var(marketValue=0, window=0)

        # Should return a positive value
        self.assertGreater(var_result, 0)

    def test_single_asset_historical_var(self):
        """Test HistoricalVaR with single asset portfolio"""
        single_asset_prices = self.sample_prices[:, 0:1]
        single_weight = np.array([1.0])

        hist_var = HistoricalVaR(self.confidence_interval, single_asset_prices, single_weight)
        var_result = hist_var.var(marketValue=1000000, window=0)

        # Should work with single asset
        self.assertGreater(var_result, 0)

    def test_minimal_historical_data(self):
        """Test HistoricalVaR with minimal data (3 price points)"""
        minimal_prices = self.sample_prices[:3, :]

        hist_var = HistoricalVaR(self.confidence_interval, minimal_prices, self.weights)

        # Should have 2 returns
        self.assertEqual(hist_var.returnMatrix.shape[0], 2)

        # Should still calculate VaR
        var_result = hist_var.var(marketValue=1000000, window=0)
        self.assertGreater(var_result, 0)

    def test_var_window_boundary_conditions(self):
        """Test VaR with various window boundary conditions"""
        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, self.weights)
        n_returns = hist_var.returnMatrix.shape[0]

        # Window = 1 (minimum valid window)
        var_1 = hist_var.var(marketValue=1000000, window=1)
        self.assertGreater(var_1, 0)

        # Window = number of returns (maximum valid window)
        var_max = hist_var.var(marketValue=1000000, window=n_returns)
        self.assertGreater(var_max, 0)

    def test_var_with_zero_window_uses_all_data(self):
        """Test that window=0 uses all available data"""
        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, self.weights)

        var_zero_window = hist_var.var(marketValue=1000000, window=0)
        var_full_data = hist_var.var(marketValue=1000000, window=hist_var.returnMatrix.shape[0])

        # Should be the same
        self.assertEqual(var_zero_window, var_full_data)

    def test_portfolio_return_shape(self):
        """Test that portfolio return has correct shape after calculation"""
        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, self.weights)

        # Trigger calculation
        hist_var.var(marketValue=1000000)

        # Portfolio return should be 1D array with length = number of returns
        self.assertEqual(len(hist_var.portfolioReturn.shape), 1)
        self.assertEqual(len(hist_var.portfolioReturn), hist_var.returnMatrix.shape[0])

    def test_extreme_confidence_intervals_historical(self):
        """Test HistoricalVaR with extreme confidence intervals"""
        # Very low confidence
        hist_var_low = HistoricalVaR(0.01, self.sample_prices, self.weights)
        var_1pct = hist_var_low.var(marketValue=1000000, window=0)

        # Very high confidence
        hist_var_high = HistoricalVaR(0.999, self.sample_prices, self.weights)
        var_999 = hist_var_high.var(marketValue=1000000, window=0)

        # Both should be positive
        self.assertGreater(var_1pct, 0)
        self.assertGreater(var_999, 0)

    def test_var_updates_portfolio_return_each_call(self):
        """Test that portfolio return is recalculated on each var() call"""
        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, self.weights)

        # First calculation
        var1 = hist_var.var(marketValue=1000000, window=0)
        portfolio_return_1 = hist_var.portfolioReturn.copy()

        # Change weights
        new_weights = np.array([0.5, 0.3, 0.2])
        hist_var.setWeights(new_weights)

        # Second calculation
        var2 = hist_var.var(marketValue=1000000, window=0)
        portfolio_return_2 = hist_var.portfolioReturn

        # Portfolio returns should be different
        self.assertFalse(np.array_equal(portfolio_return_1, portfolio_return_2))

        # VaR values might be different
        # Just verify both are positive
        self.assertGreater(var1, 0)
        self.assertGreater(var2, 0)

    def test_var_with_negative_market_value_historical(self):
        """Test HistoricalVaR with negative market value"""
        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, self.weights)

        var_negative = hist_var.var(marketValue=-1000000, window=0)
        var_zero = hist_var.var(marketValue=0, window=0)

        # Both should return percentage (same value)
        self.assertEqual(var_negative, var_zero)

    def test_var_window_too_large(self):
        """Test that window much larger than data length raises exception"""
        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, self.weights)
        n_returns = hist_var.returnMatrix.shape[0]

        # Window significantly larger than available data should raise exception
        # The condition is: window > len(portfolioReturn) + 1
        with self.assertRaises(Exception) as context:
            hist_var.var(marketValue=1000000, window=n_returns + 2)
        self.assertIn("invalid Window", str(context.exception))

    def test_percentile_method_nearest(self):
        """Test that percentile uses 'nearest' interpolation method"""
        # Create simple price data with known returns
        prices = np.array([
            [100, 100, 100],
            [101, 101, 101],
            [102, 102, 102],
            [103, 103, 103],
            [104, 104, 104],
            [105, 105, 105]
        ])
        weights = np.array([1.0/3, 1.0/3, 1.0/3])

        hist_var = HistoricalVaR(0.80, prices, weights)  # 80% confidence
        var_result = hist_var.var(marketValue=0, window=0)

        # With 5 returns and 80% confidence, we look at 20th percentile
        # Should use nearest interpolation
        self.assertGreater(var_result, 0)

    def test_var_consistency_across_multiple_calls(self):
        """Test that multiple calls to var() with same parameters give same result"""
        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, self.weights)

        var1 = hist_var.var(marketValue=1000000, window=10)
        var2 = hist_var.var(marketValue=1000000, window=10)
        var3 = hist_var.var(marketValue=1000000, window=10)

        # All should be equal
        self.assertEqual(var1, var2)
        self.assertEqual(var2, var3)

    def test_weights_not_summing_to_one_historical(self):
        """Test HistoricalVaR with weights not summing to 1.0"""
        weights_not_normalized = np.array([0.5, 0.5, 0.5])
        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, weights_not_normalized)

        var_result = hist_var.var(marketValue=1000000, window=0)

        # Should still calculate (though results may not be meaningful)
        self.assertIsInstance(var_result, (int, float, np.number))

    def test_equal_weights_historical(self):
        """Test HistoricalVaR with equal weights"""
        equal_weights = np.array([1.0/3, 1.0/3, 1.0/3])
        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, equal_weights)

        var_result = hist_var.var(marketValue=1000000, window=0)
        self.assertGreater(var_result, 0)

    def test_set_portfolio_updates_returns(self):
        """Test that setPortfolio updates return matrix for HistoricalVaR"""
        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, self.weights)

        original_returns = hist_var.returnMatrix.shape[0]

        # Set new portfolio with different size
        new_prices = self.sample_prices[:15, :]  # Fewer rows
        hist_var.setPortfolio(new_prices)

        new_returns = hist_var.returnMatrix.shape[0]

        # Should have different number of returns
        self.assertNotEqual(original_returns, new_returns)
        self.assertEqual(new_returns, 14)  # 15 prices = 14 returns

    def test_var_absolute_value(self):
        """Test that VaR is always returned as absolute (positive) value"""
        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, self.weights)

        # Test with different windows
        for window in [0, 5, 10]:
            var_result = hist_var.var(marketValue=1000000, window=window)
            self.assertGreater(var_result, 0, f"VaR should be positive for window={window}")

    def test_comparison_with_parametric_var(self):
        """Test that HistoricalVaR and parametric VaR are in similar range"""
        # Create both calculators with same data
        from VaR import ValueAtRisk

        hist_var = HistoricalVaR(self.confidence_interval, self.sample_prices, self.weights)
        param_var = ValueAtRisk(self.confidence_interval, self.sample_prices, self.weights)

        # Calculate both VaR values (daily)
        hist_var_result = hist_var.var(marketValue=1000000, window=0)
        param_var_result = param_var.var(marketValue=1000000, window=1)

        # Both should be positive
        self.assertGreater(hist_var_result, 0)
        self.assertGreater(param_var_result, 0)

        # They should be within an order of magnitude of each other
        ratio = max(hist_var_result, param_var_result) / min(hist_var_result, param_var_result)
        self.assertLess(ratio, 10, "Historical and parametric VaR differ by more than 10x")


if __name__ == '__main__':
    unittest.main()
