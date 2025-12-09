import random
from .domain import PublicInfo, MarketWelfareStats, Buyer, Seller

class Market:
    '''This class provides the abstraction for the market state. Besides initialization the only things that should be changed
    in a simulation are the bids and asks of individual buyers and sellers through the update_buyer_bid and update_seller_ask methods.
    For the purposes of agent analysis, only getter methods from the PublicInfo section should be used.'''
    def __init__(self, buyers: list[Buyer], sellers: list[Seller], epsilon: float):
        self._buyers: list[Buyer] = buyers
        self._sellers: list[Seller] = sellers
        # Instantiate public info & welfare stats with sentinel defaults so update_market_info
        # can safely mutate them without AttributeError. Sentinel semantics:
        # - Prices: -1 means 'does not exist yet'
        # - Quantities: 0 when the associated price sentinel is -1
        # - Clearing price: -1.0 when no feasible trade exists yet (will be recomputed)
        self._public_info = PublicInfo(
            last_feasible_bid=-1,
            last_feasible_bid_quantity=0,
            first_infeasible_bid=-1,
            first_infeasible_bid_quantity=0,
            last_feasible_ask=-1,
            last_feasible_ask_quantity=0,
            first_infeasible_ask=-1,
            first_infeasible_ask_quantity=0,
            clearing_price=-1.0,
            epsilon=epsilon
        )
        self._market_welfare_stats = MarketWelfareStats(
            surplus_pre_epsilon=0,
            surplus_post_epsilon=0.0,
            num_trades=0,
            market_cost=0.0
        )
        self.update_market_info()

    # Getter methods
    def get_public_info(self) -> PublicInfo:
        return self._public_info

    def get_market_welfare_stats(self) -> MarketWelfareStats:
        return self._market_welfare_stats

    def get_buyers(self) -> list[Buyer]:
        return self._buyers

    def get_sellers(self) -> list[Seller]:
        return self._sellers

    # Setter methods
    '''These should only be used for analysis. Not for simulation.'''
    def set_public_info(self, public_info: PublicInfo) -> None:
        self._public_info = public_info

    def set_market_welfare_stats(self, market_welfare_stats: MarketWelfareStats) -> None:
        self._market_welfare_stats = market_welfare_stats

    def set_buyers(self, buyers: list[Buyer]) -> None:
        self._buyers = buyers

    def set_sellers(self, sellers: list[Seller]) -> None:
        self._sellers = sellers

    # General market methods
    @staticmethod
    def midpoint(bid: int, ask: int) -> float:
        '''Calculates the midpoint between a bid and ask.'''
        return (bid + ask) / 2

    def find_buyer(self, uuid: int) -> Buyer:
        for buyer in self._buyers:
            if buyer.get_uuid() == uuid:
                return buyer
        raise Exception(f"No buyer with uuid {uuid}")

    def find_seller(self, uuid: int) -> Seller:
        for seller in self._sellers:
            if seller.get_uuid() == uuid:
                return seller
        raise Exception(f"No seller with uuid {uuid}")

    def update_market_info(self, replicable: bool = False) -> None:
        '''Used to update all information in the market after a change from the buyers and/or sellers.
        Remember to call this method after every change otherwise market state will be inconsistent.'''
        if not replicable:
            # randomize for bids and asks of the same value to have a chance of transacting
            random.shuffle(self._buyers)
            random.shuffle(self._sellers)
            self._buyers.sort(key=lambda x: x.get_bid(), reverse=True)
            self._sellers.sort(key=lambda x: x.get_ask(), reverse=True)
        else:
            '''Sorts by true value/cost to ensure reproducibility.
            May lead to issues because favors buyers/sellers with better reservation values.'''
            self._buyers.sort(key=lambda x: (x.get_bid(), x.get_true_value()), reverse=True)
            self._sellers.sort(key=lambda x: (x.get_ask(), x.get_true_cost()), reverse=True)

        buyers_length = len(self._buyers)
        sellers_length = len(self._sellers)
        buyer_in = 0
        seller_in = sellers_length - 1

        last_feasible_bid: int = -1
        last_feasible_bid_quantity: int = 0

        last_feasible_ask: int = -1
        last_feasible_ask_quantity: int = 0

        surplus_pre_epsilon: int = 0
        num_trades = 0

        if self._buyers[0].get_bid() >= self._sellers[sellers_length-1].get_ask():
            last_feasible_bid = self._buyers[0].get_bid()
            last_feasible_ask = self._sellers[sellers_length-1].get_ask()
        else:
            first_infeasible_bid = self._buyers[0].get_bid()
            first_infeasible_ask = last_feasible_ask = self._sellers[sellers_length-1].get_ask()

        # conditions to transact
        while buyer_in < buyers_length and seller_in >= 0 and self._buyers[buyer_in].get_bid() >= self._sellers[seller_in].get_ask():

            current_transacting_bid = self._buyers[buyer_in].get_bid()
            if current_transacting_bid == last_feasible_bid:
                last_feasible_bid_quantity += 1
            else:
                last_feasible_bid_quantity = 1

            current_transacting_ask = self._sellers[seller_in].get_ask()
            if current_transacting_ask == last_feasible_ask:
                last_feasible_ask_quantity += 1
            else:
                last_feasible_ask_quantity = 1

            last_feasible_bid = current_transacting_bid
            last_feasible_ask = current_transacting_ask

            surplus_pre_epsilon += self._buyers[buyer_in].get_true_value() - self._sellers[seller_in].get_true_cost()
            num_trades += 1

            buyer_in += 1
            seller_in -= 1

        clearing_price: float = Market.midpoint(last_feasible_bid, last_feasible_ask)

        # while loop exits so all remaining bids are infeasible
        if buyer_in < buyers_length:
            first_infeasible_bid: int = self._buyers[buyer_in].get_bid()
            first_infeasible_bid_quantity: int = 1
            buyer_in += 1

            while buyer_in < buyers_length and self._buyers[buyer_in].get_bid() == first_infeasible_bid:
                first_infeasible_bid_quantity += 1
                buyer_in += 1
        else:
            first_infeasible_bid: int = -1
            first_infeasible_bid_quantity = 0

        # while loop exits so all remaining asks are infeasible
        if seller_in >= 0:
            first_infeasible_ask: int = self._sellers[seller_in].get_ask()
            first_infeasible_ask_quantity: int = 1
            seller_in -= 1

            while seller_in >= 0 and self._sellers[seller_in].get_ask() == first_infeasible_ask:
                first_infeasible_ask_quantity += 1
                seller_in -= 1
        else:
            first_infeasible_ask: int = -1
            first_infeasible_ask_quantity = 0

        public_info: PublicInfo = self.get_public_info()
        public_info.set_clearing_price(clearing_price)

        public_info.set_last_feasible_bid(last_feasible_bid)
        public_info.set_last_feasible_bid_quantity(last_feasible_bid_quantity)
        public_info.set_last_feasible_ask(last_feasible_ask)
        public_info.set_last_feasible_ask_quantity(last_feasible_ask_quantity)

        public_info.set_first_infeasible_bid(first_infeasible_bid)
        public_info.set_first_infeasible_bid_quantity(first_infeasible_bid_quantity)
        public_info.set_first_infeasible_ask(first_infeasible_ask)
        public_info.set_first_infeasible_ask_quantity(first_infeasible_ask_quantity)

        market_welfare_stats: MarketWelfareStats = self.get_market_welfare_stats()
        market_welfare_stats.set_surplus_pre_epsilon(surplus_pre_epsilon)
        market_welfare_stats.set_num_trades(num_trades)
        market_welfare_stats.set_market_cost(public_info.get_epsilon())
        market_welfare_stats.set_surplus_post_epsilon()

    def all_agent_random_deviation(self, min_buy: int, max_sell: int, replicable: bool = False) -> None:
        '''Deviates the bids and asks from true costs and prices respectively.
        Bids can be between the true value and the min_buy (determined by the simulation) inclusive. 
        Asks can be between the true cost and max_sell (determined by the simulation) inclusive.
        These deviations are uniformly random.'''
        for buyer in self._buyers:
            buyer.set_bid(buyer.get_true_value() - random.randint(0, buyer.get_true_value() - min_buy))
        for seller in self._sellers:
            seller.set_ask(seller.get_true_cost() + random.randint(0, max_sell - seller.get_true_cost()))
        self.update_market_info(replicable)

    def update_buyer_bid(self, buyer: Buyer, new_bid: int, replicable: bool = False) -> None:
        if new_bid == -1:
            self._buyers.remove(buyer)
        else:
            buyer.set_bid(new_bid)
        self.update_market_info(replicable)

    def update_seller_ask(self, seller: Seller, new_ask: int, replicable: bool = False) -> None:
        if new_ask == -1:
            self._sellers.remove(seller)
        else:
            seller.set_ask(new_ask)
        self.update_market_info(replicable)

    def __str__(self):
        return f"Market(buyers={self.get_buyers()}, sellers={self.get_sellers()}, epsilon={self.get_public_info().get_epsilon()})"