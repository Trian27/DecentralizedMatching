import random, copy
from typing import Tuple
import logging
logging.basicConfig(filename='simulation_limited_info.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from MarketFramework.domain import PublicInfo, Buyer, Seller
from MarketFramework.market import Market

def expected_buyer_surplus(market: Market, buyer: Buyer, new_bid: int) -> float:
    old_bid: int = buyer.get_bid()
    market.update_buyer_bid(buyer, new_bid)

    public_info: PublicInfo = market.get_public_info()

    last_feasible_bid: int = public_info.get_last_feasible_bid()
    last_feasible_bid_quantity: int = public_info.get_last_feasible_bid_quantity()
    first_infeasible_bid: int = public_info.get_first_infeasible_bid()
    first_infeasible_bid_quantity: int = public_info.get_first_infeasible_bid_quantity()
    clearing_price: float = public_info.get_clearing_price()
    epsilon: float = public_info.get_epsilon()

    if last_feasible_bid_quantity == 0:
        market.update_buyer_bid(buyer, old_bid)
        return 0.0

    expected_surplus: float = 0.0

    if (buyer.get_bid() > last_feasible_bid) or (first_infeasible_bid_quantity == 0) or (last_feasible_bid > first_infeasible_bid and buyer.get_bid() == last_feasible_bid):
        expected_surplus =  buyer.get_true_value() - clearing_price + epsilon
    elif buyer.get_bid() == last_feasible_bid:
        expected_surplus = (buyer.get_true_value() - clearing_price + epsilon) * (last_feasible_bid_quantity / (last_feasible_bid_quantity + first_infeasible_bid_quantity))
    else:
        expected_surplus = 0.0

    market.update_buyer_bid(buyer, old_bid)

    return expected_surplus

def expected_seller_surplus(market: Market, seller: Seller, new_ask: int) -> float:
    old_ask: int = seller.get_ask()
    market.update_seller_ask(seller, new_ask)

    public_info: PublicInfo = market.get_public_info()

    last_feasible_ask: int = public_info.get_last_feasible_ask()
    last_feasible_ask_quantity: int = public_info.get_last_feasible_ask_quantity()
    first_infeasible_ask: int = public_info.get_first_infeasible_ask()
    first_infeasible_ask_quantity: int = public_info.get_first_infeasible_ask_quantity()
    clearing_price: float = public_info.get_clearing_price()
    epsilon: float = public_info.get_epsilon()

    if last_feasible_ask_quantity == 0:
        market.update_seller_ask(seller, old_ask)
        return 0.0

    expected_surplus: float = 0.0

    if (seller.get_ask() < last_feasible_ask) or (first_infeasible_ask_quantity == 0) or (last_feasible_ask > first_infeasible_ask and seller.get_ask() == last_feasible_ask):
        expected_surplus =  clearing_price - seller.get_true_cost() + epsilon
    elif seller.get_ask() == last_feasible_ask:
        expected_surplus = (clearing_price - seller.get_true_cost() + epsilon) * (last_feasible_ask_quantity / (last_feasible_ask_quantity + first_infeasible_ask_quantity))
    else:
        expected_surplus = 0.0

    market.update_seller_ask(seller, old_ask)

    return expected_surplus

def calculate_best_bid(uuid: int, market: Market) -> bool:
    '''Calculates valid bid that maximizes expected value of trade for the buyer. We iterate through all bids that give us a chance to transact.
    If there is no bid that gives positive surplus (the agent has no incentive to deviate), we increment the bid by 1 if possible.
    If all bids that provide a chance of transacting result in negative surplus, we set the bid to the highest bid that will keep them out of the money.'''

    buyer: Buyer = market.find_buyer(uuid)
    original_bid: int = buyer.get_bid()
    true_value: int = buyer.get_true_value()

    best_bid: int = original_bid
    max_expected_surplus: float = expected_buyer_surplus(market, buyer, original_bid)

    public_info_pre_move: PublicInfo = market.get_public_info()
    last_feasible_bid_pre_move: int = public_info_pre_move.get_last_feasible_bid()
    last_feasible_bid_quantity_pre_move: int = public_info_pre_move.get_last_feasible_bid_quantity()
    first_infeasible_bid_pre_move: int = public_info_pre_move.get_first_infeasible_bid()
    first_infeasible_bid_quantity_pre_move: int = public_info_pre_move.get_first_infeasible_bid_quantity()
    last_feasible_ask_pre_move: int = public_info_pre_move.get_last_feasible_ask()
    last_feasible_ask_quantity_pre_move: int = public_info_pre_move.get_last_feasible_ask_quantity()
    first_infeasible_ask_pre_move: int = public_info_pre_move.get_first_infeasible_ask()
    first_infeasible_ask_quantity_pre_move: int = public_info_pre_move.get_first_infeasible_ask_quantity()

    in_the_money_pre_move: bool = False
    # for the purposes of frontier bid analysis, having a fractional chance of being in the money is equivalent to not being in the money.
    if last_feasible_bid_quantity_pre_move > 0:
        if (original_bid > last_feasible_bid_pre_move) or (first_infeasible_bid_quantity_pre_move == 0) or (original_bid == last_feasible_bid_pre_move and last_feasible_bid_pre_move > first_infeasible_bid_pre_move):
            in_the_money_pre_move = True

    frontier_bid: int = -1
    min_bid: int = -1

    if in_the_money_pre_move:
        # case where there is both first infeasible bid and last feasible ask. Always have a last feasible ask because in the money.
        if first_infeasible_bid_quantity_pre_move > 0:
            frontier_bid = max(first_infeasible_bid_pre_move + 1, last_feasible_ask_pre_move)
            min_bid = max(first_infeasible_bid_pre_move, last_feasible_ask_pre_move)
        # case where there is only a last feasible ask.
        elif last_feasible_ask_quantity_pre_move > 0:
            frontier_bid = last_feasible_ask_pre_move
            min_bid = last_feasible_ask_pre_move
    else:
        # at least one of the following must exist. At least one seller. If in the money, implies there is a buyer in the money. If out of the money, implies first infeasible ask existence.
        if last_feasible_bid_quantity_pre_move > 0 and first_infeasible_ask_quantity_pre_move > 0:
            frontier_bid = min(last_feasible_bid_pre_move + 1, first_infeasible_ask_pre_move)
            min_bid = min(last_feasible_bid_pre_move, first_infeasible_ask_pre_move)
        # case where there is only a last feasible bid
        elif last_feasible_bid_quantity_pre_move > 0:
            frontier_bid = last_feasible_bid_pre_move + 1
            min_bid = last_feasible_bid_pre_move
        # case where there is only a first infeasible ask
        elif first_infeasible_ask_quantity_pre_move > 0:
            frontier_bid = first_infeasible_ask_pre_move
            min_bid = first_infeasible_ask_pre_move

    for bid in range(frontier_bid, min_bid - 1, -1):
        expected_surplus = expected_buyer_surplus(market, buyer, bid)
        # this is top heavy. If same surplus as before, this assumes buyers will bid high. Maybe counter-intuitive.
        if expected_surplus > max_expected_surplus:
            max_expected_surplus = expected_surplus
            best_bid = bid   

    '''If there is no bid that results in a positive expected surplus, we increment the bid by 1,
    as long as it is less than the true value. This represents the belief that presenting a higher bid
    will, in the future, give a higher probability of transacting. It is the only behavior that is
    non-myopic. For the game, it serves the purpose of making sure agents aren't stuck at their initial
    deviations. Note here we DO restrict the +1 movement to the true_value. This is because the buyer only
    increases their bid in such a way if all other rational bids result in no transaction. If a seller were to match
    their new bid in a future round, and it was higher than their true value, that would result in a clearing
    price too high for this buyer.'''
    if max_expected_surplus == 0 and original_bid < true_value:
        best_bid = original_bid + 1

    # if cannot have a chance of transacting with positive surplus, bid at highest value that keeps them out of the money.
    if max_expected_surplus < 0:
        best_bid = min_bid - 1
        max_expected_surplus = 0

    market.update_buyer_bid(buyer, best_bid)
    return best_bid != original_bid

def calculate_best_ask(uuid: int, market: Market) -> bool:
    '''Calculates valid ask that maximizes expected value of trade for the seller. We iterate through all asks that give us a chance to transact.
    If there is no ask that gives positive surplus (the agent has no incentive to deviate), we decrement the ask by 1 if possible.
    If all asks that provide a chance of transacting result in negative surplus, we set the ask to the highest ask that will keep them out of the money.'''

    seller: Seller = market.find_seller(uuid)
    original_ask: int = seller.get_ask()
    true_cost: int = seller.get_true_cost()

    best_ask: int = original_ask
    max_expected_surplus: float = expected_seller_surplus(market, seller, original_ask)

    public_info_pre_move: PublicInfo = market.get_public_info()
    last_feasible_bid_pre_move: int = public_info_pre_move.get_last_feasible_bid()
    last_feasible_bid_quantity_pre_move: int = public_info_pre_move.get_last_feasible_bid_quantity()
    first_infeasible_bid_pre_move: int = public_info_pre_move.get_first_infeasible_bid()
    first_infeasible_bid_quantity_pre_move: int = public_info_pre_move.get_first_infeasible_bid_quantity()
    last_feasible_ask_pre_move: int = public_info_pre_move.get_last_feasible_ask()
    last_feasible_ask_quantity_pre_move: int = public_info_pre_move.get_last_feasible_ask_quantity()
    first_infeasible_ask_pre_move: int = public_info_pre_move.get_first_infeasible_ask()
    first_infeasible_ask_quantity_pre_move: int = public_info_pre_move.get_first_infeasible_ask_quantity()

    in_the_money_pre_move: bool = False
    # for the purposes of frontier ask analysis, having a fractional chance of being in the money is equivalent to not being in the money.
    if last_feasible_ask_quantity_pre_move > 0:
        if (original_ask < last_feasible_ask_pre_move) or (first_infeasible_ask_quantity_pre_move == 0) or (original_ask == last_feasible_ask_pre_move and last_feasible_ask_pre_move < first_infeasible_ask_pre_move):
            in_the_money_pre_move = True

    frontier_ask: int = -1
    max_ask: int = -1

    if in_the_money_pre_move:
        # case where there is both first infeasible ask and last feasible bid. Know there will always be last feasible bid because seller is in the money
        if first_infeasible_ask_quantity_pre_move > 0:
            frontier_ask = min(first_infeasible_ask_pre_move - 1, last_feasible_bid_pre_move)
            max_ask = min(first_infeasible_ask_pre_move, last_feasible_bid_pre_move)
        # case where there is only a last feasible bid.
        elif last_feasible_bid_quantity_pre_move > 0:
            frontier_ask = last_feasible_bid_pre_move
            max_ask = last_feasible_bid_pre_move
    else:
        # at least one of the following must exist. At least one buyer. If in the money, implies there is a seller in the money. If out of the money, implies first infeasible bid existence.
        if last_feasible_ask_quantity_pre_move > 0 and first_infeasible_bid_quantity_pre_move > 0:
            frontier_ask = max(last_feasible_ask_pre_move - 1, first_infeasible_bid_pre_move)
            max_ask = max(last_feasible_ask_pre_move, first_infeasible_bid_pre_move)
        # case where there is only a last feasible ask
        elif last_feasible_ask_quantity_pre_move > 0:
            frontier_ask = last_feasible_ask_pre_move - 1
            max_ask = last_feasible_ask_pre_move
        # case where there is only a first infeasible bid
        elif first_infeasible_bid_quantity_pre_move > 0:
            frontier_ask = first_infeasible_bid_pre_move
            max_ask = first_infeasible_bid_pre_move

    for ask in range(frontier_ask, max_ask + 1, 1):
        expected_surplus = expected_seller_surplus(market, seller, ask)
        # this is bottom heavy. If same surplus as before, this assumes sellers will ask low. Maybe counter-intuitive.
        if expected_surplus > max_expected_surplus:
            max_expected_surplus = expected_surplus
            best_ask = ask

    '''If there is no ask that results in a positive expected surplus, we increment the ask by 1,
    as long as it is greater than the true cost. This represents the belief that presenting a higher ask
    will, in the future, give a higher probability of transacting. It is the only behavior that is
    non-myopic. For the game, it serves the purpose of making sure agents aren't stuck at their initial
    deviations. Note here we DO restrict the -1 movement to the true_cost. This is because the seller only
    increases their ask in such a way if all other rational ask result in no transaction. If a buyer were to match
    their new ask in a future round, and it was lower than their true cost, that would result in a clearing
    price too low for this seller.'''
    if max_expected_surplus == 0 and original_ask > true_cost:
        best_ask = original_ask - 1

    # if cannot have a chance of transacting with positive surplus, bid at highest value that keeps them out of the money.
    if max_expected_surplus < 0:
        best_ask = max_ask + 1
        max_expected_surplus = 0

    market.update_seller_ask(seller, best_ask)
    return best_ask != original_ask

def myopic_unilateral_deviation(market: Market) -> bool:
    '''We are given a market state. We pick randomly an agent, through UUID, to make their move. 
    If this agent is the same as the last turn, then we move on since we have already picked their optimal move.
    If the agent is a buyer, we calculate the best bid for them. If the agent is a seller, we calculate the best ask for them.
    We continue this for 1000 iterations (there are 1000 picks of an agent) or until we reach convergence.
    To find when we have reached convergence we check when the length of inactive_agents is equal to the length of buyers + sellers.
    Each entry in inactive_agents is correlated to the UIUD of an agent. The sellers are indexed from buyers_length to buyers_length + sellers_length (so we don't have to generate multiple numbers). For example, if there are 5 buyers then 5 correlates to the seller with UUID 0. If any agent makes a move, then we clear the whole set of inactive agents.
    This is because one agent's move can make previously inactive agents active again. If we reach 1000 iterations, we return False, otherwise we return True.'''

    buyers_length = len(market.get_buyers())
    sellers_length = len(market.get_sellers())

    inactive_agents: set[int] = set()

    counter = 0
    last_agent = -1

    iteration_threshold = 1000

    while (len(inactive_agents) < buyers_length + sellers_length) and (counter < iteration_threshold):
        curr_agent = random.randint(0, buyers_length + sellers_length - 1) # will refer to UUID of buyer, or UIUD of seller + len(buyers)

        if curr_agent == last_agent:
            counter -= 1
            continue

        last_agent = curr_agent
        if curr_agent < buyers_length:
            if calculate_best_bid(curr_agent, market):
                inactive_agents.clear()
            else:
                inactive_agents.add(curr_agent)
        else:
            seller_uuid = curr_agent - buyers_length
            if calculate_best_ask(seller_uuid, market):
                inactive_agents.clear()
            else:
                inactive_agents.add(curr_agent)

        counter += 1

    return counter < iteration_threshold

def simulation(num_buyers: int, num_sellers: int, min_buy: int, max_buy: int, min_sell: int, max_sell: int, epsilon: float) -> Tuple[int, float, int, list[int], list[float], list[int], list[list[int | None]], list[list[float | None]], list[list[int | None]], int]:
    '''Creates a market with num_buyers buyers and num_sellers sellers. The true costs are randomly generated between min_buy and max_buy.
    The true prices are randomly generated from min_sell to max_sell. The bids and asks are initially the same as the true costs and prices (before initial deviation).
    We then calculate the market surplus and number of trades before any deviation. We then deviate the bids and asks randomly (initial deviation).
    We deviate the bids and asks randomly 10 times (deep copy the lists beforehand). Each time we deviate we do the following:
        We recalculate the market surplus and number of trades since this can be drastically different.
        We then run the myopic_unilateral_deviation function 10 times. Each time we deep copy the buyers and sellers array.
        This is so that we have 10 iterations where we start with random agents (testing to see if order matters).
        We record the surplus and number of trades after each deviation in an array (including how many times we did not reach equilibrium).'''
    # Create Buyer objects: uuid=i, true_value=random, bid=true_value initially
    buyers = [Buyer(uuid=i, true_value=(value := random.randint(min_buy, max_buy)), bid=value) for i in range(num_buyers)]
    # Create Seller objects: uuid=i, true_cost=random, ask=true_cost initially  
    sellers = [Seller(uuid=i, true_cost=(value := random.randint(min_sell, max_sell)), ask=value) for i in range(num_sellers)]

    market: Market = Market(buyers, sellers, epsilon)
    market_welfare_stats_b4_dev = market.get_market_welfare_stats()
    surplus_pre_epsilon_b4_dev = market_welfare_stats_b4_dev.get_surplus_pre_epsilon()
    surplus_post_epsilon_b4_dev = market_welfare_stats_b4_dev.get_surplus_post_epsilon()
    trades_b4_dev = market_welfare_stats_b4_dev.get_num_trades()

    times_no_equilibrium = 0

    surplus_pre_epsilon_after_rand: list[int] = [] # each element represents surplus pre epsilon after an initial deviation
    surplus_post_epsilon_after_rand: list[float] = [] # each element represents surplus post epsilon after an initial deviation
    trades_after_rand: list[int] = [] # each element represents number of trades after an initial deviation

    surplus_pre_epsilon_post_dev: list[list[int|None]] = [] # each element is list (see surplus_pre_epsilon_post_dev_iteration)
    surplus_post_epsilon_post_dev: list[list[float|None]] = [] # each element is list (see surplus_post_epsilon_post_dev_iteration)
    trades_post_dev: list[list[int|None]]= [] # each element is list (see trades_post_dev_iteration)

    threshold_1 = 100
    for _ in range(threshold_1):
        market_copy_1: Market = copy.deepcopy(market)
        market_copy_1.all_agent_random_deviation(min_buy, max_sell)

        market_welfare_stats_after_rand_iteration = market_copy_1.get_market_welfare_stats()
        surplus_pre_epsilon_after_rand_iteration = market_welfare_stats_after_rand_iteration.get_surplus_pre_epsilon()
        surplus_post_epsilon_after_rand_iteration = market_welfare_stats_after_rand_iteration.get_surplus_post_epsilon()
        trades_after_rand_iteration = market_welfare_stats_after_rand_iteration.get_num_trades()

        surplus_pre_epsilon_after_rand.append(surplus_pre_epsilon_after_rand_iteration)
        surplus_post_epsilon_after_rand.append(surplus_post_epsilon_after_rand_iteration)
        trades_after_rand.append(trades_after_rand_iteration)
        
        surplus_pre_epsilon_post_dev_iteration: list[int|None] = [] # each element represents pre epsilon surplus for a random agent order
        surplus_post_epsilon_post_dev_iteration: list[float|None] = [] # each element represents post epsilon surplus for a random agent order
        trades_post_dev_iteration: list[int|None] = [] # each element represents number of trades for a random agent order
        threshold_2 = 100
        for j in range(threshold_2):
            market_copy_2 = copy.deepcopy(market_copy_1) # deep copy so we can test for multiple random agent orders
            equilibrium_reached = myopic_unilateral_deviation(market_copy_2)
            if equilibrium_reached:
                market_welfare_stats_after_dev = market_copy_2.get_market_welfare_stats()
                surplus_pre_epsilon_after_dev = market_welfare_stats_after_dev.get_surplus_pre_epsilon()
                surplus_post_epsilon_after_dev = market_welfare_stats_after_dev.get_surplus_post_epsilon()
                trades_after_dev = market_welfare_stats_after_dev.get_num_trades()
                if (surplus_pre_epsilon_after_dev != surplus_pre_epsilon_b4_dev) or (surplus_post_epsilon_after_dev != surplus_post_epsilon_b4_dev) or (trades_after_dev != trades_b4_dev):
                    logging.info("DIFFERENCE IN SURPLUS/NUMBER OF TRADES")
                    logging.info(f"Market before initial deviation: {market}")
                    logging.info(f"Market after myopic deviation: {market_copy_2}")

                surplus_pre_epsilon_post_dev_iteration.append(surplus_pre_epsilon_after_dev)
                surplus_post_epsilon_post_dev_iteration.append(surplus_post_epsilon_after_dev)
                trades_post_dev_iteration.append(trades_after_dev)
            else:
                logging.info("NO EQUILIBRIUM REACHED")
                logging.info(f"Market before initial deviation: {market}")
                logging.info(f"Market after initial deviation: {market_copy_1}")
                times_no_equilibrium += 1
                surplus_pre_epsilon_post_dev_iteration.append(None)
                surplus_post_epsilon_post_dev_iteration.append(None)
                trades_post_dev_iteration.append(None)
        surplus_pre_epsilon_post_dev.append(surplus_pre_epsilon_post_dev_iteration)
        surplus_post_epsilon_post_dev.append(surplus_post_epsilon_post_dev_iteration)
        trades_post_dev.append(trades_post_dev_iteration)

    return surplus_pre_epsilon_b4_dev, surplus_post_epsilon_b4_dev, trades_b4_dev, surplus_pre_epsilon_after_rand, surplus_post_epsilon_after_rand, trades_after_rand, surplus_pre_epsilon_post_dev, surplus_post_epsilon_post_dev, trades_post_dev, times_no_equilibrium

def main():
    '''Runs the market simulation method 10 times with 5 buyers and 5 sellers.
    Each time the simulation is run, 10 initial deviations are set,
    and for each initial deviation, 10 actual simulations are run,
    to create randomness in agent order.'''
    
    # fix a seed to reproduce
    random.seed(10)

    threshold = 100
    for _ in range(threshold):
        simulation(5, 5, 0, 20, 0, 20, 2/3)

if __name__ == "__main__":
    main()