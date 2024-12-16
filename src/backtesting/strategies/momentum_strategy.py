import logging

from pandas import Timestamp

from .base_strategy import Strategy
import pandas as pd


class MomentumStrategy(Strategy):
    """
    Momentum-based trading strategy that generates buy or sell signals
    based on the calculated momentum of price data.

    Attributes:
        data (pd.DataFrame): The market data used for backtesting.
        threshold (float): The momentum threshold for generating buy/sell signals.
        period (int): The rolling window size for calculating momentum.
        momentum (pd.Series): The computed momentum values for the given data.
    """

    def __init__(self, data: pd.DataFrame, threshold: float , period: int):
        """
        Initialize the momentum strategy with market data, a threshold, and a period.

        Args:
            data (pd.DataFrame): The market data, must contain a 'Close' column.
            threshold (float): The threshold value for generating signals.
            period (int): The rolling window size for momentum calculation.

        Raises:
            ValueError: If the input data is not a DataFrame, is empty, or lacks a 'Close' column.
                        If the period is not an integer greater than 1.
                        If the threshold is not a positive number.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("Data cannot be empty.")
        if not isinstance(period, int) or period <= 1:
            raise ValueError("Period must be an integer greater than 1.")
        if not isinstance(threshold, (int, float)) or threshold <= 0:
            raise ValueError("Threshold must be a positive number.")
        if 'Close' not in data.columns:
            raise ValueError("Data must contain a 'Close' column.")

        self.data = data
        self.threshold = threshold
        self.period = period
        self.momentum = self.calculate_momentum()

    def calculate_momentum(self) -> pd.Series:
        """
        Calculate momentum using the formula: (X_n - X_{n-1}) / X_n.

        Momentum is based on the rolling average of the closing prices over the specified period.

        Returns:
            pd.Series: The calculated momentum values.

        Raises:
            ValueError: If an error occurs during the calculation.
        """
        try:
            rolling_avg = self.data.rolling(window=self.period).mean()
            momentum = rolling_avg.diff() / rolling_avg
            momentum.dropna(inplace=True)
            return momentum
        except Exception as e:
            raise ValueError(f"Error in momentum calculation: {e}")

    def should_buy(self, time_index: Timestamp) -> bool:
        """
        Determine whether to generate a buy signal at a specific timestamp.

        Args:
            time_index (Timestamp): The timestamp to check for a buy signal.

        Returns:
            bool: True if momentum exceeds the threshold, False otherwise.

        Logs:
            Debug: If the timestamp is not found in the momentum index.
            Error: If a KeyError occurs.
            Exception: If an unexpected error occurs.
        """
        try:
            if time_index in self.momentum.index:
                value = float(self.momentum.loc[time_index,"Close"])
                return value > self.threshold
            else:
                logging.debug(f"Timestamp {time_index} not found in momentum index for MomentumStrategy Class.")
                return False

        except KeyError as e:
            logging.error(f"KeyError in should_buy for time index {time_index}: {e}")
            return False
        except Exception as e:
            logging.exception(f"Unexpected error in should_buy for time index {time_index}: {e}")
            return False

    def should_sell(self, time_index: Timestamp) -> bool:
        """
        Determine whether to generate a sell signal at a specific timestamp.

        Args:
            time_index (Timestamp): The timestamp to check for a sell signal.

        Returns:
            bool: True if momentum is below the negative threshold, False otherwise.

        Logs:
            Debug: If the timestamp is not found in the momentum index.
            Error: If a KeyError occurs.
            Exception: If an unexpected error occurs.
        """
        try:
            if time_index in self.momentum.index:
                value = float(self.momentum.loc[time_index,"Close"])
                return value < -self.threshold
            else:
                logging.debug(f"Timestamp {time_index} not found in momentum index for MomentumStrategy Class.")
                return False

        except KeyError as e:
            logging.error(f"KeyError in should_sell for time index {time_index}: {e}")
            return False
        except Exception as e:
            logging.exception(f"Unexpected error in should_sell for time index {time_index}: {e}")
            return False
