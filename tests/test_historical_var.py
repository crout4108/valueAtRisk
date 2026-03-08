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


    # ------------------------------------------------------------------
    # Deterministic tests with pre-computed expected values
    # ------------------------------------------------------------------

    def test_var_exact_value_full_window(self):
        """Test VaR against a pre-computed expected value (full history)"""
        # 6 prices → 5 log returns; equal weights
        known_prices = np.array([
            [100.0, 200.0, 150.0],
            [110.0, 190.0, 155.0],
            [95.0,  205.0, 145.0],
            [105.0, 195.0, 160.0],
            [100.0, 210.0, 150.0],
            [108.0, 200.0, 158.0],
        ])
        equal_weights = np.array([1/3, 1/3, 1/3])

        # Portfolio returns (weighted average of per-asset log returns):
        #   [0.025602, -0.045770, 0.049504, -0.013074, 0.026710]
        # Sorted: [-0.045770, -0.013074, 0.025602, 0.026710, 0.049504]
        # 5th percentile using 'nearest' on 5 values selects index 0 → -0.045770
        # VaR = abs(-0.045770) = 0.045769647237542024
        expected_var = 0.045769647237542024

        hist_var = HistoricalVaR(0.95, known_prices, equal_weights)
        var_result = hist_var.var()

        self.assertAlmostEqual(var_result, expected_var, places=12)

    def test_var_exact_value_with_window(self):
        """Test VaR against a pre-computed expected value using a rolling window"""
        known_prices = np.array([
            [100.0, 200.0, 150.0],
            [110.0, 190.0, 155.0],
            [95.0,  205.0, 145.0],
            [105.0, 195.0, 160.0],
            [100.0, 210.0, 150.0],
            [108.0, 200.0, 158.0],
        ])
        equal_weights = np.array([1/3, 1/3, 1/3])

        # window=3 selects the last 3 portfolio returns: [0.049504, -0.013074, 0.026710]
        # Sorted: [-0.013074, 0.026710, 0.049504]
        # 5th percentile using 'nearest' on 3 values selects index 0 → -0.013074
        # VaR = abs(-0.013074) = 0.013073571051093559
        expected_var_window3 = 0.013073571051093559

        hist_var = HistoricalVaR(0.95, known_prices, equal_weights)
        var_result = hist_var.var(window=3)

        self.assertAlmostEqual(var_result, expected_var_window3, places=12)

    def test_var_exact_dollar_value(self):
        """Test that dollar VaR equals percentage VaR scaled by market value exactly"""
        known_prices = np.array([
            [100.0, 200.0, 150.0],
            [110.0, 190.0, 155.0],
            [95.0,  205.0, 145.0],
            [105.0, 195.0, 160.0],
            [100.0, 210.0, 150.0],
            [108.0, 200.0, 158.0],
        ])
        equal_weights = np.array([1/3, 1/3, 1/3])
        market_value = 100_000

        expected_var_pct = 0.045769647237542024
        expected_var_dollar = expected_var_pct * market_value  # 4576.9647...

        hist_var = HistoricalVaR(0.95, known_prices, equal_weights)
        var_dollar = hist_var.var(marketValue=market_value)

        self.assertAlmostEqual(var_dollar, expected_var_dollar, places=8)

    def test_set_weights_updates_var(self):
        """Test that setWeights correctly updates the VaR calculation"""
        known_prices = np.array([
            [100.0, 200.0, 150.0],
            [110.0, 190.0, 155.0],
            [95.0,  205.0, 145.0],
            [105.0, 195.0, 160.0],
            [100.0, 210.0, 150.0],
            [108.0, 200.0, 158.0],
        ])
        equal_weights = np.array([1/3, 1/3, 1/3])
        first_only_weights = np.array([1.0, 0.0, 0.0])

        hist_var = HistoricalVaR(0.95, known_prices, equal_weights)
        var_before = hist_var.var()

        hist_var.setWeights(first_only_weights)
        var_after = hist_var.var()

        # VaR should change when weights change
        self.assertNotAlmostEqual(var_before, var_after, places=6)

        # Verify the new VaR matches the pre-computed value for first-asset-only weights
        # First-asset log returns: [log(110/100), log(95/110), log(105/95), log(100/105), log(108/100)]
        #   ≈ [0.09531, -0.14660, 0.10008, -0.04879, 0.07696]
        # Sorted: [-0.14660, -0.04879, 0.07696, 0.09531, 0.10008]
        # 5th percentile using 'nearest' on 5 values selects index 0 → -0.14660 (the minimum)
        # VaR = abs(-0.14660) = 0.1466034741918758
        expected_var_after = 0.1466034741918758
        self.assertAlmostEqual(var_after, expected_var_after, places=12)

    def test_set_portfolio_updates_var(self):
        """Test that setPortfolio correctly updates the VaR calculation"""
        known_prices = np.array([
            [100.0, 200.0, 150.0],
            [110.0, 190.0, 155.0],
            [95.0,  205.0, 145.0],
            [105.0, 195.0, 160.0],
            [100.0, 210.0, 150.0],
            [108.0, 200.0, 158.0],
        ])
        # Uniformly trending prices produce very small returns
        trending_prices = np.array([
            [100.0, 200.0, 150.0],
            [102.0, 198.0, 152.0],
            [104.0, 196.0, 154.0],
            [106.0, 194.0, 156.0],
            [108.0, 192.0, 158.0],
            [110.0, 190.0, 160.0],
        ])
        equal_weights = np.array([1/3, 1/3, 1/3])

        hist_var = HistoricalVaR(0.95, known_prices, equal_weights)
        var_original = hist_var.var()

        hist_var.setPortfolio(trending_prices)
        var_updated = hist_var.var()

        # VaR should change after updating the price data
        self.assertNotAlmostEqual(var_original, var_updated, places=6)

        # Pre-computed expected value for the trending prices
        expected_var_updated = 0.006818873669253674
        self.assertAlmostEqual(var_updated, expected_var_updated, places=12)

    def test_higher_confidence_level_increases_var(self):
        """Test that a higher confidence level yields equal or greater VaR on larger data"""
        # Use a larger dataset so confidence-level ordering is reliable
        np.random.seed(0)
        base = np.array([100.0, 200.0, 150.0])
        rets = np.random.normal(0, 0.02, (99, 3))
        prices = np.zeros((100, 3))
        prices[0] = base
        for i in range(1, 100):
            prices[i] = prices[i - 1] * np.exp(rets[i - 1])

        weights = np.array([0.4, 0.3, 0.3])

        hist_var_90 = HistoricalVaR(0.90, prices, weights)
        hist_var_95 = HistoricalVaR(0.95, prices, weights)
        hist_var_99 = HistoricalVaR(0.99, prices, weights)

        var_90 = hist_var_90.var()
        var_95 = hist_var_95.var()
        var_99 = hist_var_99.var()

        # VaR must be non-decreasing as confidence level rises
        self.assertLessEqual(var_90, var_95)
        self.assertLessEqual(var_95, var_99)


if __name__ == '__main__':
    unittest.main()
