
from abc import ABC, abstractmethod
from pandas import Timestamp

class Strategy(ABC):
    """Strategy interface."""

    @abstractmethod
    def should_buy(self, time_index: Timestamp) -> bool:
        """Determines whether to buy at the given time index."""
        raise NotImplementedError

    @abstractmethod
    def should_sell(self, time_index: Timestamp) -> bool:
        """Determines whether to sell at the given time index."""
        raise NotImplementedError

