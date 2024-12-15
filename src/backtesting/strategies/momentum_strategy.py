import logging

from pandas import Timestamp

from .base_strategy import Strategy
import pandas as pd


class MomentumStrategy(Strategy):
    """Example momentum-based strategy."""

    def __init__(self, data: pd.DataFrame, threshold: float , period: int):
        self.data = data
        self.threshold = threshold
        self.period = period
        self.momentum = self.calculate_momentum()

    def calculate_momentum(self) -> pd.Series:
        """Calculate momentum as (X_n - X_{n-1}) / X_n."""
        try:
            rolling_avg = self.data.rolling(window=self.period).mean()
            momentum = rolling_avg.diff() / rolling_avg
            return momentum
        except Exception as e:
            raise ValueError(f"Error in momentum calculation: {e}")

    def should_buy(self, time_index: Timestamp) -> bool:
        try:
            if time_index in self.momentum.index:
                value = float(self.momentum.loc[time_index].iloc[0])
                return value > self.threshold
            else:
                logging.warning(f"Timestamp {time_index} not found in momentum index for MomentumStrategy Class.")
                #print(f"Timestamp {time_index} not found in momentum index.")
                return False
        except Exception as e:
            logging.error(f"Unexpected error in should_buy method for time index {time_index}: {e}")
            return False

    def should_sell(self, time_index: Timestamp) -> bool:
        try:
            if time_index in self.momentum.index:
                value = float(self.momentum.loc[time_index].iloc[0])
                return value < -self.threshold
            else:
                logging.warning(f"Timestamp {time_index} not found in momentum index for MomentumStrategy Class.")
                #print(f"Timestamp {time_index} not found in momentum index.")
                return False
        except Exception as e:
            logging.error(f"Unexpected error in should_sell method for time index {time_index}: {e}")
            return False
