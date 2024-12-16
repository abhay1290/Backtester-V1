import logging
from email.policy import default

import numpy as np
from numpy import mean, std
from typing import List, Dict
from tabulate import tabulate

class PerformanceMetrics:
    """
    Handles the calculation and reporting of backtest performance metrics.

    Attributes:
        initial_capital (float): The starting capital for the backtest.
        final_capital (float): The ending capital after the backtest.
        trade_returns (List[float]): List of absolute returns for each trade.
        trade_returns_percentage (List[float]): List of percentage returns for each trade.
        commission (float): The commission cost per trade.
        trade_log (List[Dict]): Log of all executed trades.
        open_positions (List[Dict]): List of currently open positions.
        risk_free_rate (float): The assumed risk-free rate for calculating metrics like Sharpe and Sortino ratios (default 0.02).
    """

    def __init__(self, initial_capital: float, capital: float, trade_returns: List[float],
                 trade_returns_percentage: List[float], commission: float, trade_log: List[Dict],
                 open_positions: List[Dict], risk_free_rate: float = 0.02):
        """
        Initialize the performance metrics with relevant backtest data.

        Args:
            initial_capital (float): The starting capital for the backtest.
            capital (float): The final capital after the backtest.
            trade_returns (List[float]): Absolute returns from each trade.
            trade_returns_percentage (List[float]): Percentage returns from each trade.
            commission (float): The commission cost per trade.
            trade_log (List[Dict]): A list of dictionaries representing trade details.
            open_positions (List[Dict]): A list of dictionaries representing open positions.
            risk_free_rate (float): The assumed risk-free rate for performance calculations.
        """

        self.initial_capital = initial_capital
        self.final_capital = capital
        self.trade_returns = trade_returns
        self.trade_returns_percentage = trade_returns_percentage
        self.commission = commission
        self.trade_log = trade_log
        self.risk_free_rate = risk_free_rate
        self.open_positions = open_positions

    def calculate_metrics(self) -> Dict:
        """
        Calculate and return a dictionary of key performance metrics.

        Metrics include:
            - Starting and final portfolio values.
            - Gross profit, gross loss, and net profit.
            - Total trades executed and total open trades.
            - Commission costs.
            - Number of winning and losing trades.
            - Biggest winner and loser in monetary terms.
            - Risk-reward ratio.
            - Sharpe and Sortino ratios.
            - Portfolio volatility.

        Returns:
            Dict: A dictionary containing all calculated metrics.

        Raises:
            Exception: If an error occurs during metric calculation, logs the error and returns an empty dictionary.
        """
        try:
            gross_profit = sum([trade for trade in self.trade_returns if trade > 0]) if self.trade_returns else 0
            gross_loss = sum([trade for trade in self.trade_returns if trade < 0]) if self.trade_returns else 0
            net_profit = gross_profit + gross_loss
            total_trades = len(self.trade_log)
            total_commission = total_trades * self.commission

            winners = len([trade for trade in self.trade_returns if trade > 0])
            losers = len([trade for trade in self.trade_returns if trade < 0])
            biggest_winner = max(max(self.trade_returns,default=0),0)
            biggest_losser = min(min(self.trade_returns,default=0),0)

            risk_reward_ratio = abs((gross_profit / winners) / (gross_loss / losers)) if (losers > 0 and gross_loss != 0) or (winners > 0 and gross_profit != 0) else 0.0
            gross_returns = self.trade_returns_percentage
            excess_return = np.array(gross_returns) - self.risk_free_rate if len(gross_returns)>0 else []  # Assuming a 2% risk-free rate
            sharpe_ratio = mean(excess_return) / std(excess_return) if (std(excess_return) != 0 or len(excess_return)!=0) else 0.0
            downside_returns = [ret for ret in excess_return if ret < 0] if len(excess_return)>0 else []
            sortino_ratio = mean(excess_return) / std(downside_returns) if (std(downside_returns) != 0 or (len(excess_return)!=0) or (len(downside_returns)!=0)) else 0.0
            volatility = std(gross_returns) if len(gross_returns)>0 else 0.0

            return {
                "starting_balance": self.initial_capital,
                "final_balance": self.final_capital,
                "gross_profit": gross_profit,
                "gross_loss": gross_loss,
                "net_profit": net_profit,
                "total_trades": total_trades,
                "total_open_trades": len(self.open_positions),
                "commission": total_commission,
                "winners": winners,
                "losers": losers,
                "biggest_winner": biggest_winner,
                "biggest_losser": biggest_losser,
                "risk_reward_ratio": risk_reward_ratio,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "volatility": volatility,
            }

        except Exception as e:
            logging.error(f"Error while calculating performance metrics: {e}")
            return {}

    def print_metrics(self) -> None:
        """
        Print the calculated performance metrics in a readable tabular format.

        Outputs include:
            - Key metrics such as PnL, gross profit, gross loss, and ratios.
            - Detailed open positions and trade logs using the `tabulate` library.

        Logs:
            Error: Logs any exceptions that occur while printing the metrics.
        """
        metrics = self.calculate_metrics()
        try:
            print(f"Starting Portfolio Value: ${metrics['starting_balance']:.2f}")
            print(f"Final Portfolio Value: ${metrics['final_balance']:.2f}")
            print(f"Gross Profit: ${metrics['gross_profit']:.2f}")
            print(f"Gross Loss: ${metrics['gross_loss']:.2f}")
            print(f"Net Profit (PnL): ${metrics['net_profit']:.2f}")
            print(f"Total Commissions: ${metrics['commission']:.2f}")
            print(f"Total Trades Executed: {metrics['total_trades']}")
            print(f"Total Open Trades: {metrics['total_open_trades']}")
            print(f"Winners: {metrics['winners']}, Losers: {metrics['losers']}")
            print(f"Biggest Winner ($): {metrics['biggest_winner']:.2f}")
            print(f"Biggest Losser ($): {metrics['biggest_losser']:.2f}")
            print(f"Risk-Reward Ratio: {metrics['risk_reward_ratio']:.2f}" if metrics['risk_reward_ratio'] else "Risk-Reward Ratio: N/A")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}" if metrics['sharpe_ratio'] else "Sharpe Ratio: N/A")
            print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}" if metrics['sortino_ratio'] else "Sortino Ratio: N/A")
            print(f"Volatility: {metrics['volatility']:.2f}")

            print("Open_Positions:")
            print(tabulate(self.open_positions, headers="keys"))
            print("Trade_Log:")
            print(tabulate(self.trade_log, headers="keys"))

        except Exception as e:
            logging.error(f"Error while printing performance metrics: {e}")