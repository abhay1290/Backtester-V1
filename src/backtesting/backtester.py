import logging
from pandas import Timestamp

from src.backtesting.PerformanceMetrics import PerformanceMetrics
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

    def __init__(self, strategy: Strategy, data: pd.DataFrame, initial_capital: float, commission: float, close_time_delta: int, exchange_close_time: Timestamp = Timestamp("16:00:00")):
        self.strategy = strategy
        self.data = data
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.trade_log: List[Dict] = []
        self.current_index = None
        self.current_date = None
        self.position_close_time = []
        self.trade_returns = []
        self.trade_returns_percentage =[]
        self.close_time_delta = close_time_delta
        self.exchange_close_time = exchange_close_time
        self.pending_trades: List[Dict] = []

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

            if len(self.pending_trades)>0 and self.pending_trades[0]['time'].date() == self.current_date - pd.Timedelta(days=1):
                try:
                    self.settle_pending_positions()
                except Exception as e:
                    logging.error(f"Error closing pending position at {self.current_index}: {e}", exc_info=True)
                    continue


    def next(self) -> bool:
        """Move to the next time step."""
        try:
            if self.current_index is None:
                self.current_index = self.data.index[0]
                self.current_date = self.current_index.date()
                return True
            elif self.current_index in self.data.index:
                current_position = self.data.index.get_loc(self.current_index)

                if current_position < len(self.data) - 1:
                    self.current_index = self.data.index[current_position + 1]
                    self.current_date = self.current_index.date()
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
            self.update_position_close_time('BUY')
            return True

        elif self.strategy.should_sell(self.current_index):
            self.open_position('SELL')
            self.update_position_close_time('SELL')
            return True
        return False

    def update_position_close_time(self, action: str):
        if self.current_index.time() >= (self.exchange_close_time - pd.Timedelta(minutes=self.close_time_delta)).time():
            pending_trade = {
                'time': self.current_index,
                'action': action
            }
            self.pending_trades.append(pending_trade)
        else:
            close_time = self.current_index + pd.Timedelta(minutes=self.close_time_delta)
            self.position_close_time.append(close_time)


    def settle_pending_positions(self) -> None:
        for trade in self.pending_trades[:]:
            self.close_pending_position(self.current_index, trade['time'])
            self.pending_trades.remove(trade)

    def open_position(self, action: str, quantity: int = 1) -> None:
        try:
            price = float(self.data.loc[self.current_index].iloc[0])
            if action == 'BUY':
                self.capital -=  (price * quantity) + self.commission   # no of share bought = 1
                print(f"Opened BUY position at {self.current_index}, {price:.2f}, Quantity: {quantity}, Capital: {self.capital:.2f}")
                logging.info(f"Opened BUY position at {self.current_index}, {price:.2f}, Quantity: {quantity}, Capital: {self.capital:.2f}")

            elif action == 'SELL':
                self.capital += (price * quantity) - self.commission   # no of share sold = 1
                print(f"Opened SELL position at {self.current_index}, {price:.2f}, Quantity: {quantity}, Capital: {self.capital:.2f}")
                logging.info(f"Opened SELL position at {self.current_index}, {price:.2f}, Quantity: {quantity}, Capital: {self.capital:.2f}")

            trade = {
                'time': self.current_index,
                'action': action,
                'price': price,
                'capital': self.capital
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
            close_price = self.get_close_price(close_time)
            open_time = close_time - pd.Timedelta(minutes=self.close_time_delta)
            trade = self.get_trade_by_time(open_time)
            action = trade['action']
            open_trade_price = trade['price']

            self.close_trade(action, close_time, close_price, open_trade_price, quantity)

        except KeyError:
            logging.error(f"Error: No data available for closing position at {close_time}.")
        except ValueError as e:
            logging.error(f"Error: {e}")
        except Exception as e:
            logging.error(f"Unexpected error during close_position: {e}", exc_info=True)

    def close_pending_position(self, close_time: Timestamp, open_time: Timestamp, quantity: int = 1) -> None:
        try:
            close_price = self.get_close_price(close_time)
            trade = self.get_trade_by_time(open_time)
            action = trade['action']
            open_trade_price = trade['price']

            self.close_trade(action, close_time, close_price, open_trade_price, quantity)

        except KeyError:
            logging.error(f"Error: No data available for closing position at {close_time}.")
        except ValueError as e:
            logging.error(f"Error: {e}")
        except Exception as e:
            logging.error(f"Unexpected error during close_pending_position: {e}", exc_info=True)

    def get_close_price(self, close_time: Timestamp) -> float:
        """Helper function to get the close price from the data."""
        try:
            return float(self.data.loc[close_time].iloc[0])
        except KeyError:
            raise ValueError(f"Data missing for {close_time}")
        except Exception as e:
            logging.error(f"Error retrieving close price for {close_time}: {e}")
            raise

    def close_trade(self, action: str, close_time: Timestamp, close_price: float, open_trade_price: float,
                     quantity: int) -> None:
        """Helper function to handle trade closure logic."""
        try:
            if action == 'BUY':
                self.capital += (close_price * quantity) - self.commission
                self.trade_log.append({'time': close_time, 'action': 'CLOSE', 'price': close_price, 'capital': self.capital})
                self.trade_returns.append(close_price - open_trade_price)
                self.trade_returns_percentage.append(((close_price - open_trade_price) * 100) / open_trade_price)
                print(f"Closed BUY position at {close_time}, Price: {close_price:.2f}, Capital: {self.capital:.2f}")
                logging.info(f"Closed BUY position at {close_time}, Price: {close_price:.2f}, Capital: {self.capital:.2f}")

            elif action == 'SELL':
                self.capital -= (close_price * quantity) + self.commission
                self.trade_log.append({'time': close_time, 'action': 'CLOSE', 'price': close_price, 'capital': self.capital})
                self.trade_returns.append(open_trade_price - close_price)
                self.trade_returns_percentage.append(((open_trade_price - close_price) * 100) / open_trade_price)
                print(f"Closed SELL position at {close_time}, Price: {close_price:.2f}, Capital: {self.capital:.2f}")
                logging.info(f"Closed SELL position at {close_time}, Price: {close_price:.2f}, Capital: {self.capital:.2f}")
            else:
                raise ValueError(f"Unexpected action: {action}")

        except ValueError as e:
            logging.error(f"Error: {e}")
        except Exception as e:
            logging.error(f"Unexpected error during close_trade: {e}", exc_info=True)

    def print_performance(self) -> None:
        """Delegate performance printing to PerformanceMetrics."""
        metrics = PerformanceMetrics(
            initial_capital=self.initial_capital,
            capital=self.capital,
            trade_returns=self.trade_returns,
            trade_returns_percentage=self.trade_returns_percentage,
            commission=self.commission,
            trade_log=self.trade_log,
            open_positions=self.pending_trades,
        )
        metrics.print_metrics()

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

    backtester = Backtester(MomentumStrategy(spx_future_data, 0.0003, 5),spx_data,100000, 2.0, 10)
    backtester.run()
    backtester.print_performance()