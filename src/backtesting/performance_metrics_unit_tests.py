import unittest
from src.backtesting.PerformanceMetrics import PerformanceMetrics

class TestPerformanceMetrics(unittest.TestCase):
    def setUp(self):
        """Set up test data and parameters for the PerformanceMetrics class."""
        self.initial_capital = 10000
        self.final_capital = 11000
        self.trade_returns = [200, -100, 300, -50, 250]
        self.trade_returns_percentage = [0.02, -0.01, 0.03, -0.005, 0.025]
        self.commission = 5
        self.trade_log = [
            {"trade_id": 1, "profit": 200},
            {"trade_id": 2, "profit": -100},
            {"trade_id": 3, "profit": 300},
            {"trade_id": 4, "profit": -50},
            {"trade_id": 5, "profit": 250},
        ]
        self.open_positions = [
            {"trade_id": 6},
            {"trade_id": 7}
        ]
        self.metrics = PerformanceMetrics(
            self.initial_capital,
            self.final_capital,
            self.trade_returns,
            self.trade_returns_percentage,
            self.commission,
            self.trade_log,
            self.open_positions
        )

    def test_calculate_metrics(self):
        """Test the calculation of performance metrics."""
        metrics = self.metrics.calculate_metrics()

        self.assertEqual(metrics["starting_balance"], self.initial_capital)
        self.assertEqual(metrics["final_balance"], self.final_capital)
        self.assertEqual(metrics["gross_profit"], 750)  # Sum of positive returns
        self.assertEqual(metrics["gross_loss"], -150)  # Sum of negative returns
        self.assertEqual(metrics["net_profit"], 600)  # Gross profit + gross loss
        self.assertEqual(metrics["total_trades"], len(self.trade_log))
        self.assertEqual(metrics["total_open_trades"], len(self.open_positions))
        self.assertEqual(metrics["commission"], len(self.trade_log) * self.commission)
        self.assertEqual(metrics["winners"], 3)
        self.assertEqual(metrics["losers"], 2)
        self.assertEqual(metrics["biggest_winner"], 300)
        self.assertEqual(metrics["biggest_losser"], -100)

        self.assertIn("risk_reward_ratio", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("sortino_ratio", metrics)

    def test_print_metrics(self):
        """Test the print_metrics function (mocking stdout)."""
        try:
            self.metrics.print_metrics()
        except Exception as e:
            self.fail(f"print_metrics raised an exception: {e}")
