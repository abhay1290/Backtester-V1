import unittest
import pandas as pd
from pandas import Timestamp
from src.backtesting.strategies import MomentumStrategy  # Update with the correct import path

class TestMomentumStrategy(unittest.TestCase):
    def setUp(self):
        """Set up test data and parameters for the MomentumStrategy class."""
        data = {
            "Close": [100, 102, 104, 106, 108, 110, 108, 106, 104, 102]
        }
        self.df = pd.DataFrame(data, index=pd.date_range("2023-01-01", periods=10))
        self.threshold = 0.01
        self.period = 3
        self.strategy = MomentumStrategy(self.df, self.threshold, self.period)

    def test_initialization(self):
        """Test proper initialization of the MomentumStrategy class."""
        self.assertEqual(self.strategy.threshold, self.threshold)
        self.assertEqual(self.strategy.period, self.period)
        self.assertTrue("Close" in self.strategy.data.columns)

    def test_calculate_momentum(self):
        """Test the momentum calculation logic."""
        momentum = self.strategy.calculate_momentum()
        self.assertIsInstance(momentum, pd.DataFrame)
        self.assertFalse(momentum.isnull().any().bool())  # Ensure no NaN values

    def test_should_buy(self):
        """Test the should_buy method."""
        # Mock conditions where buy signal should trigger
        time_index = Timestamp("2023-01-05")
        self.assertTrue(self.strategy.should_buy(time_index))

        # Mock conditions where buy signal should not trigger
        time_index = Timestamp("2023-01-09")
        self.assertFalse(self.strategy.should_buy(time_index))

    def test_should_sell(self):
        """Test the should_sell method."""
        # Mock conditions where sell signal should trigger
        time_index = Timestamp("2023-01-09")
        self.assertTrue(self.strategy.should_sell(time_index))

        # Mock conditions where sell signal should not trigger
        time_index = Timestamp("2023-01-08")
        self.assertFalse(self.strategy.should_sell(time_index))

    def test_invalid_initialization(self):
        """Test invalid inputs to the MomentumStrategy constructor."""
        with self.assertRaises(ValueError):
            MomentumStrategy(pd.DataFrame(), self.threshold, self.period)  # Empty DataFrame
        with self.assertRaises(ValueError):
            MomentumStrategy(self.df, -1, self.period)  # Negative threshold
        with self.assertRaises(ValueError):
            MomentumStrategy(self.df, self.threshold, 1)  # Invalid period
        with self.assertRaises(ValueError):
            MomentumStrategy(self.df.drop(columns=["Close"]), self.threshold, self.period)  # Missing 'Close' column
