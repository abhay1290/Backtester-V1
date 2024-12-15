import unittest
import pandas as pd
from pandas import Timestamp
from unittest.mock import MagicMock

from src.backtesting.backtester import Backtester
from src.backtesting.strategies.momentum_strategy import MomentumStrategy


class TestBacktester(unittest.TestCase):
    def setUp(self):
        """Set up test data and mock strategy for the Backtester."""
        # Create mock data
        data = {
            "Close": [100, 102, 101, 103, 105, 107, 108, 110, 109, 108]
        }
        self.df = pd.DataFrame(data, index=pd.date_range("2023-01-01", periods=10, freq="min"))

        # Create mock strategy
        self.mock_strategy = MagicMock(spec=MomentumStrategy)
        self.mock_strategy.should_buy.side_effect = lambda x: x in [Timestamp("2023-01-01 00:01"), Timestamp("2023-01-02 00:03")]
        self.mock_strategy.should_sell.side_effect = lambda x: x in [Timestamp("2023-01-01 00:05")]

        # Initialize Backtester
        self.initial_capital = 10000
        self.commission = 2.0
        self.close_time_delta = 3  # minutes
        self.backtester = Backtester(self.mock_strategy, self.df, self.initial_capital, self.commission, self.close_time_delta)

    def test_initialization(self):
        """Test proper initialization of Backtester."""
        self.assertEqual(self.backtester.initial_capital, self.initial_capital)
        self.assertEqual(self.backtester.capital, self.initial_capital)
        self.assertEqual(self.backtester.commission, self.commission)
        self.assertEqual(self.backtester.close_time_delta, self.close_time_delta)
        self.assertEqual(len(self.backtester.trade_log), 0)

    def test_run_backtest(self):
        """Test the run method and trading flow."""
        self.backtester.run()

        # Verify trade logs
        trade_log = self.backtester.trade_log
        self.assertEqual(len(trade_log), 4)  # 2 BUYs + 2 CLOSEs
        self.assertEqual(trade_log[0]["action"], "BUY")
        self.assertEqual(trade_log[1]["action"], "CLOSE")
        self.assertEqual(trade_log[2]["action"], "SELL")
        self.assertEqual(trade_log[3]["action"], "CLOSE")

        # Verify position closing
        self.assertFalse(Timestamp("2023-01-01 00:04") in self.backtester.position_close_time)
        self.assertFalse(Timestamp("2023-01-01 00:08") in self.backtester.position_close_time)

        # Verify capital change
        self.assertLess(self.backtester.capital, self.initial_capital)  # Capital decreases due to commissions

    def test_open_position(self):
        """Test the open_position method."""
        self.backtester.current_index = Timestamp("2023-01-01 00:01")
        self.backtester.open_position("BUY", quantity=1)

        # Verify trade log
        trade = self.backtester.trade_log[-1]
        self.assertEqual(trade["action"], "BUY")
        self.assertEqual(trade["price"], 102)

        # Verify capital reduction
        expected_capital = self.initial_capital - 102 - self.commission
        self.assertEqual(self.backtester.capital, expected_capital)

    def test_close_position(self):
        """Test the close_position method."""
        # Mock opening a position
        self.backtester.current_index = Timestamp("2023-01-01 00:01")
        self.backtester.open_position("BUY", quantity=1)

        # Mock closing the position
        close_time = Timestamp("2023-01-01 00:04")
        self.backtester.close_position(close_time, quantity=1)

        # Verify trade log
        trade = self.backtester.trade_log[-1]
        self.assertEqual(trade["action"], "CLOSE")
        self.assertEqual(trade["price"], 105.0)

        # Verify returns
        self.assertEqual(len(self.backtester.trade_returns), 1)
        self.assertEqual(self.backtester.trade_returns[0], 3)  # 103 - 102

    def test_performance_metrics(self):
        """Test the performance metrics calculation."""
        self.backtester.run()
        self.assertGreater(len(self.backtester.trade_returns), 0)
        self.assertGreater(len(self.backtester.trade_returns_percentage), 0)

        # Verify final capital
        self.assertGreater(self.backtester.capital, 0)

    def test_edge_cases(self):
        """Test edge cases such as no trades and empty data."""
        # Edge case: No trades
        self.mock_strategy.should_buy.side_effect = lambda x: False
        self.mock_strategy.should_sell.side_effect = lambda x: False
        self.backtester.run()
        self.assertEqual(len(self.backtester.trade_log), 0)

        # Edge case: Empty data
        empty_df = pd.DataFrame(columns=["Close"])
        backtester_empty = Backtester(self.mock_strategy, empty_df, self.initial_capital, self.commission, self.close_time_delta)
        backtester_empty.run()
        self.assertEqual(len(backtester_empty.trade_log), 0)