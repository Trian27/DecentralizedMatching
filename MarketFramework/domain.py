class PublicInfo:
    def __init__(self, last_feasible_bid: int, last_feasible_bid_quantity: int, first_infeasible_bid: int, first_infeasible_bid_quantity: int,
                 last_feasible_ask: int, last_feasible_ask_quantity: int, first_infeasible_ask: int, first_infeasible_ask_quantity: int,
                 clearing_price: float, epsilon: float):
        self._last_feasible_bid: int = last_feasible_bid
        self._last_feasible_bid_quantity: int = last_feasible_bid_quantity
        self._first_infeasible_bid: int = first_infeasible_bid
        self._first_infeasible_bid_quantity: int = first_infeasible_bid_quantity
        self._last_feasible_ask: int = last_feasible_ask
        self._last_feasible_ask_quantity: int = last_feasible_ask_quantity
        self._first_infeasible_ask: int = first_infeasible_ask
        self._first_infeasible_ask_quantity: int = first_infeasible_ask_quantity
        self._clearing_price: float = clearing_price
        self._epsilon: float = epsilon

    # Getter methods
    def get_last_feasible_bid(self) -> int:
        return self._last_feasible_bid

    def get_last_feasible_bid_quantity(self) -> int:
        return self._last_feasible_bid_quantity

    def get_first_infeasible_bid(self) -> int:
        return self._first_infeasible_bid

    def get_first_infeasible_bid_quantity(self) -> int:
        return self._first_infeasible_bid_quantity

    def get_last_feasible_ask(self) -> int:
        return self._last_feasible_ask

    def get_last_feasible_ask_quantity(self) -> int:
        return self._last_feasible_ask_quantity

    def get_first_infeasible_ask(self) -> int:
        return self._first_infeasible_ask

    def get_first_infeasible_ask_quantity(self) -> int:
        return self._first_infeasible_ask_quantity

    def get_clearing_price(self) -> float:
        return self._clearing_price
    
    def get_epsilon(self) -> float:
        return self._epsilon

    # Setter methods
    def set_last_feasible_bid(self, last_feasible_bid: int) -> None:
        self._last_feasible_bid = last_feasible_bid

    def set_last_feasible_bid_quantity(self, last_feasible_bid_quantity: int) -> None:
        self._last_feasible_bid_quantity = last_feasible_bid_quantity

    def set_first_infeasible_bid(self, first_infeasible_bid: int) -> None:
        self._first_infeasible_bid = first_infeasible_bid

    def set_first_infeasible_bid_quantity(self, first_infeasible_bid_quantity: int) -> None:
        self._first_infeasible_bid_quantity = first_infeasible_bid_quantity

    def set_last_feasible_ask(self, last_feasible_ask: int) -> None:
        self._last_feasible_ask = last_feasible_ask

    def set_last_feasible_ask_quantity(self, last_feasible_ask_quantity: int) -> None:
        self._last_feasible_ask_quantity = last_feasible_ask_quantity

    def set_first_infeasible_ask(self, first_infeasible_ask: int) -> None:
        self._first_infeasible_ask = first_infeasible_ask

    def set_first_infeasible_ask_quantity(self, first_infeasible_ask_quantity: int) -> None:
        self._first_infeasible_ask_quantity = first_infeasible_ask_quantity

    def set_clearing_price(self, clearing_price: float) -> None:
        self._clearing_price = clearing_price

    def set_epsilon(self, epsilon: float) -> None:
        self._epsilon = epsilon

class MarketWelfareStats:
    def __init__(self, surplus_pre_epsilon: int, surplus_post_epsilon: float, num_trades: int, market_cost: float):
        self._surplus_pre_epsilon: int = surplus_pre_epsilon
        self._surplus_post_epsilon: float = surplus_post_epsilon
        self._num_trades: int = num_trades
        self._market_cost: float = market_cost

    # Getter methods
    def get_surplus_pre_epsilon(self) -> int:
        return self._surplus_pre_epsilon
    def get_surplus_post_epsilon(self) -> float:
        return self._surplus_post_epsilon
    def get_num_trades(self) -> int:
        return self._num_trades
    def get_market_cost(self) -> float:
        return self._market_cost

    # Setter methods
    def set_surplus_pre_epsilon(self, surplus_pre_epsilon: int) -> None:
        self._surplus_pre_epsilon = surplus_pre_epsilon
    def set_num_trades(self, num_trades: int) -> None:
        self._num_trades = num_trades
    def set_market_cost(self, epsilon: float) -> None:
        self._market_cost = self._num_trades * epsilon
    def set_surplus_post_epsilon(self) -> None:
        self._surplus_post_epsilon = self._surplus_pre_epsilon + self._market_cost

class Buyer:
    def __init__(self, uuid: int, true_value: int, bid: int):
        self._uuid: int = uuid
        self._true_value: int = true_value
        self._bid: int = bid

    # Getter methods
    def get_uuid(self) -> int:
        return self._uuid

    def get_true_value(self) -> int:
        return self._true_value

    def get_bid(self) -> int:
        return self._bid

    # Setter methods
    def set_uuid(self, uuid: int) -> None:
        self._uuid = uuid

    def set_true_value(self, true_value: int) -> None:
        self._true_value = true_value

    def set_bid(self, bid: int) -> None:
        self._bid = bid

    def __str__(self):
        return f"Buyer(uuid={self.get_uuid()}, value={self.get_true_value()}, bid={self.get_bid()})"

class Seller:
    def __init__(self, uuid: int, true_cost: int, ask: int):
        self._uuid: int = uuid
        self._true_cost: int = true_cost
        self._ask: int = ask

    # Getter methods
    def get_uuid(self) -> int:
        return self._uuid

    def get_true_cost(self) -> int:
        return self._true_cost

    def get_ask(self) -> int:
        return self._ask

    # Setter methods
    def set_uuid(self, uuid: int) -> None:
        self._uuid = uuid

    def set_true_cost(self, true_cost: int) -> None:
        self._true_cost = true_cost

    def set_ask(self, ask: int) -> None:
        self._ask = ask

    def __str__(self):
        return f"Seller(uuid={self.get_uuid()}, value={self.get_true_cost()}, bid={self.get_ask()})"