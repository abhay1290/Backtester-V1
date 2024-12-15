import logging

import numpy as np
from numpy import mean, std
from pandas import Timestamp

from src.backtesting.strategies.momentum_strategy import MomentumStrategy
from src.backtesting.strategies.base_strategy import Strategy
from typing import List, Dict
import pandas as pd
from src.definitions import SPX_FUTURE_DATA, SPX_INDEX_DATA


class Backtester:
    """Simple backtester that goes over the data in incremental time
    steps. The size of the time steps depend on the granularity of the
    data. When working with 1-minute interval data, each time step is
    taken 1 minute into the future.
    """

    def __init__(self, strategy: Strategy, data: pd.DataFrame, initial_capital: float = 100000.0, commission: float = 2.0, close_time_delta: int = 10):
        self.strategy = strategy
        self.data = data
        self.capital = initial_capital
        self.commission = commission
        self.trade_log: List[Dict] = []
        self.current_index = None
        self.position_close_time = []
        self.gross_trade_returns = []
        self.close_time_delta = close_time_delta

    def run(self) -> None:
        """Run the backtest."""
        while self.next():
            try:
                self.check_strategy()
            except Exception as e:
                logging.error(f"Error in strategy check at {self.current_index}: {e}", exc_info=True)
                continue

            if self.current_index in self.position_close_time:
                try:
                    self.close_position(self.current_index)
                    self.position_close_time.remove(self.current_index)
                except Exception as e:
                    logging.error(f"Error closing position at {self.current_index}: {e}", exc_info=True)
                    continue


    def next(self) -> bool:
        """Move to the next time step."""
        try:
            if self.current_index is None:
                self.current_index = self.data.index[0]
                return True
            elif self.current_index in self.data.index:
                current_position = self.data.index.get_loc(self.current_index)

                if current_position < len(self.data) - 1:
                    self.current_index = self.data.index[current_position + 1]
                    return True
        except KeyError as e:
            logging.error(f"KeyError: {self.current_index} not found in the data index. {e}", exc_info=True)
        except IndexError as e:
            logging.error(f"IndexError: Index {self.current_index} is out of range. {e}", exc_info=True)
        return False

    def check_strategy(self) -> bool:
        """Check if we need to buy or sell."""
        if self.strategy.should_buy(self.current_index):
            self.open_position('BUY')
            close_time = self.current_index + pd.Timedelta(minutes=self.close_time_delta)
            self.position_close_time.append(close_time)
            return True

        elif self.strategy.should_sell(self.current_index):
            self.open_position('SELL')
            close_time = self.current_index + pd.Timedelta(minutes=self.close_time_delta)
            self.position_close_time.append(close_time)
            return True
        return False

    def open_position(self, action: str, quantity: int = 1) -> None:
        try:
            price = float(self.data.loc[self.current_index].iloc[0])
            if action == 'BUY':
                self.capital -=  (price * quantity) + self.commission   # no of share bought = 1
                logging.info(f"Opened BUY position at {price:.2f}, Quantity: {quantity}")

            elif action == 'SELL':
                self.capital += (price * quantity) - self.commission   # no of share sold = 1
                logging.info(f"Opened SELL position at {price:.2f}, Quantity: {quantity}")

            trade = {
                'time': self.current_index,
                'action': action,
                'price': price
            }

            self.trade_log.append(trade)

        except KeyError:
            logging.error(f"Error: Data missing for index {self.current_index} to open {action} position.")
        except ValueError as e:
            logging.error(f"Error: Invalid data for price at {self.current_index}. {e}")

    def get_trade_by_time(self, time: Timestamp) -> Dict:
        try:
            trades = [trade for trade in self.trade_log if trade['time'] == time and trade['action'] != 'CLOSE']
            if not trades:
                raise ValueError(f"No trade found at {time}.")
            return trades[0]

        except ValueError as e:
            logging.error(f"Error: {e}")
            return {}

    def close_position(self, close_time: Timestamp, quantity: int = 1) -> None:
        try:
            close_price = float(self.data.loc[close_time].iloc[0])
            open_time = close_time - pd.Timedelta(minutes=self.close_time_delta)
            trade = self.get_trade_by_time(open_time)
            action = trade['action']
            open_trade_price = trade['price']

            if action == 'BUY':
                self.capital += (close_price * quantity) - self.commission
                self.trade_log.append({'time': close_time, 'action': 'CLOSE', 'price': close_price})
                self.gross_trade_returns.append(((close_price - open_trade_price) * 100)/open_trade_price)
                logging.info(f"Closed BUY position at {close_time}, Price: {close_price:.2f}")

            elif action == 'SELL':
                self.capital -= (close_price * quantity) + self.commission
                self.trade_log.append({'time': close_time, 'action': 'CLOSE', 'price': close_price})
                self.gross_trade_returns.append(((open_trade_price - close_price) * 100)/open_trade_price)
                logging.info(f"Closed SELL position at {close_time}, Price: {close_price:.2f}")

        except KeyError:
            logging.error(f"Error: No data available for closing position at {close_time}.")
        except ValueError as e:
            logging.error(f"Error: {e}")


    def print_performance(self) -> None:
        """Print performance metrics."""
        try:
            starting_balance = 100000
            final_balance = self.capital
            pnl = final_balance - starting_balance
            total_trades = len(self.trade_log)
            commission = total_trades * self.commission
            winners = len([trade for trade in self.gross_trade_returns if trade > 0])
            biggest_winner = max(self.gross_trade_returns)
            biggest_losser = min(self.gross_trade_returns)
            losers = len([trade for trade in self.gross_trade_returns if trade < 0])
            gross_returns = self.gross_trade_returns
            formatted_gross_returns = [float(f"{value:.4f}") for value in gross_returns]
            excess_return = np.array(gross_returns) - 0.02
            formatted_excess_returns = [float(f"{value:.4f}") for value in excess_return]
            sharpe_ratio = mean(excess_return)/ std(excess_return)
            trade_log = self.trade_log

            print(f"Starting Portfolio Value: ${starting_balance:.2f}")
            print(f"Final Portfolio Value: ${final_balance:.2f}")
            print(f"pnl: {pnl:.2f}")
            print(f"Gross Trading Return: {sum(gross_returns):.2f}")
            print(f"Commissions: {commission:.2f}")
            print(f"Total Trades Executed: {total_trades}")
            print(f"Winners: {winners}, Losers: {losers}")
            print(f"Biggest Winner(%): {biggest_winner:.4f}")
            print(f"Biggest Losser(%): {biggest_losser:.4f}")
            print(f"Gross Return(%): {formatted_gross_returns}")
            print(f"Excess Return(assuming 2% risk free rate): {formatted_excess_returns}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            for i in trade_log:
                print(i)

        except Exception as e:
            print(f"Error while printing performance metrics: {e}")

if __name__ == "__main__":

    def load_data(file_path: str) -> pd.DataFrame:
        data = pd.read_csv(file_path)
        data['Datetime'] = pd.to_datetime(data['Datetime']).dt.tz_localize(None)
        data.set_index('Datetime', inplace=True)
        return data

    spx_data_1m = load_data(str(SPX_INDEX_DATA))
    spx_future_data_1m = load_data(str(SPX_FUTURE_DATA))

    spx_data = spx_data_1m[['Close']]
    spx_future_data = spx_future_data_1m[['Close']]

    backtester = Backtester(MomentumStrategy(spx_future_data, 0.0003, 5),spx_data)
    backtester.run()
    backtester.print_performance()