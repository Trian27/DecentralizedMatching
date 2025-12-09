'''This simulation aims to find if prices shoot up faster if agents bid at the frontier
versus using a plus 1 strategy. We will test the following scenarios:
1) Everyone uses the +1/-1 strategy.
2) One buyer bids at frontier, everyone else uses the +1/-1 strategy.
3) All buyers bid at the frontier. Sellers use the -1 strategy.
4) All agents bid/ask at the frontier.

Note that for the +1/-1 strategy we do NOT ONLY stop at reservation. Because agents are not allowed to 
decrease their bid, or increase their ask, we will also keep going if the frontier bid/ask is profitable.'''

import random, copy
from typing import Tuple, Sequence
import logging
import csv
logging.basicConfig(filename='simulation_limited_info.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from MarketFramework.domain import PublicInfo, Buyer, Seller
from MarketFramework.market import Market

ONE_BUYER_FRONTIER = 0
PM_ONE = True
SCENARIOS = {"SCENARIO_1": 1, "SCENARIO_2": 2, "SCENARIO_3": 3, "SCENARIO_4": 4}

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

def calculate_best_bid(uuid: int, market: Market, strategy: bool) -> bool:
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
    
    min_bid = max(min_bid, original_bid) # by restriction, can go no lower than current/original bid

    for bid in range(frontier_bid, min_bid - 1, -1):
        expected_surplus = expected_buyer_surplus(market, buyer, bid)
        # this is top heavy. If same surplus as before, this assumes buyers will bid high. Maybe counter-intuitive.
        if expected_surplus > max_expected_surplus:
            max_expected_surplus = expected_surplus
            best_bid = bid 
    
    if strategy == PM_ONE and max_expected_surplus > 0: # frontier offers postiive surplus
        best_bid = original_bid + 1
    elif max_expected_surplus == 0 and original_bid < true_value:
        # Case where the frontier is out of reach but not at reserve so decrement anyway.
        best_bid = original_bid + 1
    elif max_expected_surplus < 0:
        # if we are at negative surplus then drop out of the market. Simulated by bidding -1.
        best_bid = -1
        max_expected_surplus = 0

    market.update_buyer_bid(buyer, best_bid)
    return best_bid != original_bid

def calculate_best_ask(uuid: int, market: Market, strategy: bool) -> bool:
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

    max_ask = min(max_ask, original_ask) # by restriction, can go no higher than current/original ask

    for ask in range(frontier_ask, max_ask + 1, 1):
        expected_surplus = expected_seller_surplus(market, seller, ask)
        # this is bottom heavy. If same surplus as before, this assumes sellers will ask low. Maybe counter-intuitive.
        if expected_surplus > max_expected_surplus:
            max_expected_surplus = expected_surplus
            best_ask = ask

    if strategy == PM_ONE and max_expected_surplus > 0:
        best_ask = original_ask - 1
    elif max_expected_surplus == 0 and original_ask > true_cost:
        # The case where the frontier is out of reach, but still not at reserve, so decrement anyway.
        best_ask = original_ask - 1
    elif max_expected_surplus < 0:
        # if cannot have a chance of transacting with positive surplus, drop out. Simulated by asking -1
        best_ask = -1
        max_expected_surplus = 0

    market.update_seller_ask(seller, best_ask)
    return best_ask != original_ask

def myopic_unilateral_deviation(market: Market, scenario: int) -> bool:
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
            changed_bid = False
            if scenario == SCENARIOS["SCENARIO_1"]: # all pm
                changed_bid = calculate_best_bid(curr_agent, market, PM_ONE)
            elif scenario == SCENARIOS["SCENARIO_2"]: # only 1 buyer frontier
                if curr_agent == 0: # choose the first buyer to be frontier buyer WLOG
                    changed_bid = calculate_best_bid(curr_agent, market, not PM_ONE)
                else:
                    changed_bid = calculate_best_bid(curr_agent, market, PM_ONE)
            elif scenario == SCENARIOS["SCENARIO_3"] or scenario == SCENARIOS["SCENARIO_4"]: # all buyers frontier
                changed_bid = calculate_best_bid(curr_agent, market, not PM_ONE)
            if changed_bid:
                inactive_agents.clear()
            else:
                inactive_agents.add(curr_agent)
        else:
            seller_uuid = curr_agent - buyers_length
            changed_ask = False
            if scenario == SCENARIOS["SCENARIO_4"]: # sellers frontier
                changed_ask = calculate_best_ask(seller_uuid, market, not PM_ONE)
            else:
                changed_ask = calculate_best_ask(seller_uuid, market, PM_ONE)
            if changed_ask:
                inactive_agents.clear()
            else:
                inactive_agents.add(curr_agent)

        counter += 1

    return counter < iteration_threshold

def calculate_list_stats(values: Sequence[float | int | None]) -> Tuple[float | None, float | None]:
    """Helper to compute Mean and Median from a list of values, ignoring None."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None, None
    
    # Mean
    mean_val = sum(valid) / len(valid)
    
    # Median
    sorted_vals = sorted(valid)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        median_val = (sorted_vals[mid-1] + sorted_vals[mid]) / 2
    else:
        median_val = sorted_vals[mid]
        
    return mean_val, median_val

def simulation(num_buyers: int, num_sellers: int, min_buy: int, max_buy: int, min_sell: int, max_sell: int, epsilon: float):
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
    clearing_price_b4_dev = market.get_public_info().get_clearing_price()

    times_no_equilibrium = 0

    surplus_pre_epsilon_after_rand: list[int] = [] # each element represents surplus pre epsilon after an initial deviation
    surplus_post_epsilon_after_rand: list[float] = [] # each element represents surplus post epsilon after an initial deviation
    trades_after_rand: list[int] = [] # each element represents number of trades after an initial deviation
    clearing_price_after_rand: list[float] = [] # each element represents clearing price after an initial deviation

    surplus_pre_epsilon_post_dev: list[dict[int, list[int|None]]] = [] # each element is dict (see surplus_pre_epsilon_post_dev_scenario)
    surplus_post_epsilon_post_dev: list[dict[int, list[float|None]]] = [] # each element is dict (see surplus_post_epsilon_post_dev_scenario)
    trades_post_dev: list[dict[int, list[int|None]]]= [] # each element is dict (see trades_post_dev_scenario)
    clearing_price_post_dev: list[dict[int, list[float|None]]] = [] # each element is a dict (see clearing_price_post_dev_scenario)

    threshold_1 = 100
    for _ in range(threshold_1):
        market_copy_1: Market = copy.deepcopy(market)
        market_copy_1.all_agent_random_deviation(min_buy, max_sell)

        market_welfare_stats_after_rand_iteration = market_copy_1.get_market_welfare_stats()
        surplus_pre_epsilon_after_rand_iteration = market_welfare_stats_after_rand_iteration.get_surplus_pre_epsilon()
        surplus_post_epsilon_after_rand_iteration = market_welfare_stats_after_rand_iteration.get_surplus_post_epsilon()
        trades_after_rand_iteration = market_welfare_stats_after_rand_iteration.get_num_trades()
        clearing_price_after_rand_iteration = market_copy_1.get_public_info().get_clearing_price()

        surplus_pre_epsilon_after_rand.append(surplus_pre_epsilon_after_rand_iteration)
        surplus_post_epsilon_after_rand.append(surplus_post_epsilon_after_rand_iteration)
        trades_after_rand.append(trades_after_rand_iteration)
        clearing_price_after_rand.append(clearing_price_after_rand_iteration)

        # because different scenarios may cause different agent orders, must take averages over agent orders
        surplus_pre_epsilon_post_dev_scenario: dict[int, list[int|None]] = {} # each element represents a list of pre epsilon surplus values for that scenario
        surplus_post_epsilon_post_dev_scenario: dict[int, list[float|None]] = {} # each element represents a list of post epsilon surplus values for that scenario
        trades_post_dev_scenario: dict[int, list[int|None]] = {} # each element represents a list of the number of trades executed for that scenario
        clearing_price_post_dev_scenario: dict[int, list[float|None]] = {} # each element represents a list of clearing price values for that scenario
        for scenario in SCENARIOS.values():
            surplus_pre_epsilon_post_dev_iteration: list[int|None] = [] # each element represents pre epsilon surplus for a random agent order
            surplus_post_epsilon_post_dev_iteration: list[float|None] = [] # each element represents post epsilon surplus for a random agent order
            trades_post_dev_iteration: list[int|None] = [] # each element represents number of trades for a random agent order
            clearing_price_post_dev_iteration: list[float|None] = [] # each element represents the clearing price for a random agent order

            threshold_2 = 100
            for _ in range(threshold_2):
                market_copy_2 = copy.deepcopy(market_copy_1) # deep copy so we can test for multiple random agent orders
                equilibrium_reached = myopic_unilateral_deviation(market_copy_2, scenario)
                if equilibrium_reached:
                    market_welfare_stats_after_dev = market_copy_2.get_market_welfare_stats()
                    surplus_pre_epsilon_after_dev = market_welfare_stats_after_dev.get_surplus_pre_epsilon()
                    surplus_post_epsilon_after_dev = market_welfare_stats_after_dev.get_surplus_post_epsilon()
                    trades_after_dev = market_welfare_stats_after_dev.get_num_trades()
                    clearing_price_after_dev = market_copy_2.get_public_info().get_clearing_price()
                    if (surplus_pre_epsilon_after_dev != surplus_pre_epsilon_b4_dev) or (surplus_post_epsilon_after_dev != surplus_post_epsilon_b4_dev) or (trades_after_dev != trades_b4_dev):
                        logging.info("DIFFERENCE IN SURPLUS/NUMBER OF TRADES IN SCENARIO {scenario}")
                        logging.info(f"Market before initial deviation: {market}")
                        logging.info(f"Market after myopic deviation: {market_copy_2}")

                    surplus_pre_epsilon_post_dev_iteration.append(surplus_pre_epsilon_after_dev)
                    surplus_post_epsilon_post_dev_iteration.append(surplus_post_epsilon_after_dev)
                    trades_post_dev_iteration.append(trades_after_dev)
                    clearing_price_post_dev_iteration.append(clearing_price_after_dev)
                else:
                    logging.info("NO EQUILIBRIUM REACHED IN SCENARIO {scenario}")
                    logging.info(f"Market before initial deviation: {market}")
                    logging.info(f"Market after initial deviation: {market_copy_1}")
                    times_no_equilibrium += 1
                    surplus_pre_epsilon_post_dev_iteration.append(None)
                    surplus_post_epsilon_post_dev_iteration.append(None)
                    trades_post_dev_iteration.append(None)
                    clearing_price_post_dev_iteration.append(None)
            surplus_pre_epsilon_post_dev_scenario[scenario] = surplus_pre_epsilon_post_dev_iteration
            surplus_post_epsilon_post_dev_scenario[scenario] = surplus_post_epsilon_post_dev_iteration
            trades_post_dev_scenario[scenario] = trades_post_dev_iteration
            clearing_price_post_dev_scenario[scenario] = clearing_price_post_dev_iteration

        surplus_pre_epsilon_post_dev.append(surplus_pre_epsilon_post_dev_scenario)
        surplus_post_epsilon_post_dev.append(surplus_post_epsilon_post_dev_scenario)
        trades_post_dev.append(trades_post_dev_scenario)
        clearing_price_post_dev.append(clearing_price_post_dev_scenario)

    return (surplus_pre_epsilon_b4_dev, surplus_post_epsilon_b4_dev, trades_b4_dev, clearing_price_b4_dev,
            surplus_pre_epsilon_after_rand, surplus_post_epsilon_after_rand, trades_after_rand, clearing_price_after_rand,
            surplus_pre_epsilon_post_dev, surplus_post_epsilon_post_dev, trades_post_dev, clearing_price_post_dev,
            times_no_equilibrium)

def main():
    '''Runs the market simulation method.'''
    random.seed(10)

    # Define headers
    headers = ['Trial', 'Rand_Iter', 'Total_No_Eq',
               'Base_Price', 'Base_SurplusPre', 'Base_SurplusPost', 'Base_Trades',
               'Rand_Price', 'Rand_SurplusPre', 'Rand_SurplusPost', 'Rand_Trades']

    # Add headers for each scenario and metric
    for s in SCENARIOS.values():
        for m in ['Price', 'SurplusPre', 'SurplusPost', 'Trades']:
            headers.append(f'S{s}_{m}_Mean')
            headers.append(f'S{s}_{m}_Median')

    with open('frontier_vs_plus1_results.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        threshold = 100 # Number of trials (Market Initializations)
        print(f"Starting simulation of {threshold} trials...")

        for t in range(threshold):
            # Unpack the raw data returned by simulation
            (base_spre, base_spost, base_trades, base_price,
             rand_spre, rand_spost, rand_trades, rand_price,
             dev_spre, dev_spost, dev_trades, dev_price,
             no_eq) = simulation(5, 5, 0, 20, 0, 20, 2/3)

            # Process the raw data: Iterate over the threshold_1 randomizations
            num_rand_iterations = len(rand_spre)
            
            for i in range(num_rand_iterations):
                row = {}
                row['Trial'] = t
                row['Rand_Iter'] = i
                
                # Base stats (constant for this trial)
                row['Base_Price'] = base_price
                row['Base_SurplusPre'] = base_spre
                row['Base_SurplusPost'] = base_spost
                row['Base_Trades'] = base_trades
                
                # Rand stats (specific to this iteration i)
                row['Rand_Price'] = rand_price[i]
                row['Rand_SurplusPre'] = rand_spre[i]
                row['Rand_SurplusPost'] = rand_spost[i]
                row['Rand_Trades'] = rand_trades[i]

                # Scenario stats: Aggregate the inner list (threshold_2) for this iteration i
                # dev_price[i] is a dict: {scenario_id: [list of prices]}
                for s_id in SCENARIOS.values():
                    # Price
                    p_mean, p_med = calculate_list_stats(dev_price[i][s_id])
                    row[f'S{s_id}_Price_Mean'] = p_mean
                    row[f'S{s_id}_Price_Median'] = p_med
                    
                    # Surplus Pre
                    spre_mean, spre_med = calculate_list_stats(dev_spre[i][s_id])
                    row[f'S{s_id}_SurplusPre_Mean'] = spre_mean
                    row[f'S{s_id}_SurplusPre_Median'] = spre_med
                    
                    # Surplus Post
                    spost_mean, spost_med = calculate_list_stats(dev_spost[i][s_id])
                    row[f'S{s_id}_SurplusPost_Mean'] = spost_mean
                    row[f'S{s_id}_SurplusPost_Median'] = spost_med
                    
                    # Trades
                    t_mean, t_med = calculate_list_stats(dev_trades[i][s_id])
                    row[f'S{s_id}_Trades_Mean'] = t_mean
                    row[f'S{s_id}_Trades_Median'] = t_med

                row['Total_No_Eq'] = no_eq
                writer.writerow(row)

if __name__ == "__main__":
    main()