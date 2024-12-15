import logging

from pandas import Timestamp

from .base_strategy import Strategy
import pandas as pd


class MomentumStrategy(Strategy):
    """Example momentum-based strategy."""

    def __init__(self, data: pd.DataFrame, threshold: float , period: int):
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
        """Calculate momentum as (X_n - X_{n-1}) / X_n."""
        try:
            rolling_avg = self.data.rolling(window=self.period).mean()
            momentum = rolling_avg.diff() / rolling_avg
            momentum.dropna(inplace=True)
            return momentum
        except Exception as e:
            raise ValueError(f"Error in momentum calculation: {e}")

    def should_buy(self, time_index: Timestamp) -> bool:
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
