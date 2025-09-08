import random, copy
from typing import Tuple
import pdb
import logging
logging.basicConfig(filename='simulationNewPrice.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEBUG_PRINT = False

def find_buyer(buyers: list[list[int]], buyer_uuid: int) -> list[int]:
    for curr_buyer in buyers:
        if curr_buyer[0] == buyer_uuid:
            return curr_buyer
    raise Exception(f"No buyer with uuid {buyer_uuid}")

def find_seller(sellers:list[list[int]], seller_uuid: int) -> list[int]:
    for curr_seller in sellers:
        if curr_seller[0] == seller_uuid:
            return curr_seller
    raise Exception(f"No seller with uuid {seller_uuid}")

def clearing_price(buyers: list[list[int]], sellers: list[list[int]]) -> float:
    '''Calculates the clearing price given a bid and ask. Right now is set at the midpoint.'''
    buyers.sort(key=lambda x: (x[2]), reverse=True)
    sellers.sort(key=lambda x: (x[2]), reverse=True)
    # # This is the sorting algorithm that we will use for the simulation to ensure reproducibility
    # buyers.sort(key=lambda x: (x[2],x[1]), reverse=True)
    # sellers.sort(key=lambda x: (x[2],x[1]), reverse=True)

    buyers_length = len(buyers)
    sellers_length = len(sellers)
    buyer_in = 0
    seller_in = sellers_length - 1
    while buyer_in < buyers_length and seller_in >= 0 and buyers[buyer_in][2] >= sellers[seller_in][2]:
        buyer_in += 1
        seller_in -= 1
    buyer_in -= 1
    seller_in += 1
    '''If there are no transacting pairs, we return -1.
    We chose not to throw an error to maintain consistency (see market_surplus_plus_clearing_price)'''
    if (buyer_in < 0) or (seller_in >= sellers_length):
        return -1 
    return midpoint(buyers[buyer_in][2], sellers[seller_in][2])

def midpoint(bid: int, ask: int) -> float:
    '''Calculates the midpoint between a bid and ask.'''
    return (bid + ask) / 2

def initial_deviation(buyers: list[list[int]], sellers: list[list[int]], min_buy: int, max_sell: int) -> None:
    '''Deviates the bids and asks from true costs and prices respectively.
    Bids can be between the true cost and the min_buy (determined earlier in the simulation) inclusive. 
    Asks can be between the true price and max_sell (determined earlier in the simulation) inclusive.
    These deviations are uniformly random.'''
    index = 0
    buyers_length = len(buyers)
    sellers_length = len(sellers)
    while index < buyers_length or index < sellers_length:
        if index < buyers_length:
            buyers[index][2] = buyers[index][1] - random.randint(0, buyers[index][1]-min_buy)
        if index < sellers_length:
            sellers[index][2] = sellers[index][1] + random.randint(0, max_sell - sellers[index][1])
        index += 1

def calculate_market_surplus(buyers: list[list[int]], sellers: list[list[int]]) -> Tuple[int, int]:
    '''Calculates the market surplus and number of trades given a list of buyers and sellers.
    Market surplus is calculated as the sum of the surplus of each trade (see appendix for definition).
    We also record the number of trades that take place.
    Notably, the surplus calculation does not depend on the clearing price (see appendix for proof).'''

    # This will be the true shuffling and sorting algorithm before matching
    random.shuffle(buyers)
    random.shuffle(sellers)
    buyers.sort(key=lambda x: (x[2]), reverse=True)
    sellers.sort(key=lambda x: (x[2]), reverse=True)

    # # This is the sorting algorithm that we will use for the simulation to ensure reproducibility
    # buyers.sort(key=lambda x: (x[2],x[1]), reverse=True)
    # sellers.sort(key=lambda x: (x[2],x[1]), reverse=True)

    buyers_length = len(buyers)
    sellers_length = len(sellers)
    surplus = 0
    buyer_in = 0
    num_trades = 0
    seller_in = sellers_length - 1
    # transact when there are still buyers and sellers and the buyers bid is higher than the seller's ask
    while buyer_in < buyers_length and seller_in >= 0 and buyers[buyer_in][2] >= sellers[seller_in][2]: 
        surplus += buyers[buyer_in][1] - sellers[seller_in][1] # total surplus of transaction is the difference between the true cost and the true price
        buyer_in += 1
        seller_in -= 1
        num_trades += 1
    return surplus, num_trades

def market_surplus_plus_clearing_price(buyers: list[list[int]], sellers: list[list[int]]) -> Tuple[float, int, int]:
    # This will be the true shuffling and sorting algorithm before matching
    random.shuffle(buyers)
    random.shuffle(sellers)
    buyers.sort(key=lambda x: (x[2]), reverse=True)
    sellers.sort(key=lambda x: (x[2]), reverse=True)

    # # This is the sorting algorithm that we will use for the simulation to ensure reproducibility
    # buyers.sort(key=lambda x: (x[2],x[1]), reverse=True)
    # sellers.sort(key=lambda x: (x[2],x[1]), reverse=True)

    buyers_length = len(buyers)
    sellers_length = len(sellers)
    surplus = 0
    buyer_in = 0
    num_trades = 0
    seller_in = sellers_length - 1
    # transact when there are still buyers and sellers and the buyers bid is higher than the seller's ask
    while buyer_in < buyers_length and seller_in >= 0 and buyers[buyer_in][2] >= sellers[seller_in][2]:
        surplus += buyers[buyer_in][1] - sellers[seller_in][1] # total surplus of transaction is the difference between the true cost and the true price
        buyer_in += 1
        seller_in -= 1
        num_trades += 1
    buyer_in -= 1
    seller_in += 1
    # if no trades happened, set the clearing price to -1
    if num_trades == 0:
        clearing_price = -1
    else:
        clearing_price = midpoint(buyers[buyer_in][2], sellers[seller_in][2])
    return clearing_price, surplus, num_trades

def expected_buyer_surplus(true_cost: int, bid: int, epsilon: float, buyers: list[list[int]], sellers: list[list[int]]) -> Tuple[float, bool]:
    '''Calculates the expected surplus of a buyer given a bid. We also track if there were buyers at that bid who did not 
    transact.'''
    buyers.sort(key=lambda x: x[2], reverse=True)
    sellers.sort(key=lambda x: x[2], reverse=True)
    buyers_length = len(buyers)
    sellers_length = len(sellers)
    buyer_in = 0
    seller_in = sellers_length - 1
    bid_total_surplus = 0
    count = 0
    market_transacts = False

    while buyer_in < buyers_length and seller_in >= 0 and buyers[buyer_in][2] >= sellers[seller_in][2]:
        # a transacting buyer with this bid
        if (buyers[buyer_in][2] == bid):
            count += 1
        market_transacts = True
        buyer_in += 1
        seller_in -= 1

    # we only really care about the clearing price
    if market_transacts:
        buyer_in -= 1
        seller_in += 1
        clearing_price = midpoint(buyers[buyer_in][2], sellers[seller_in][2])
        bid_total_surplus = count * (true_cost - clearing_price+ epsilon)
        # reset the indices for the next operation
        buyer_in += 1
        seller_in -= 1

    bid_all_transacting = True
    # there may be buyers at this bid we have not accounted for yet
    while buyer_in < buyers_length and buyers[buyer_in][2] >= bid:
        if (buyers[buyer_in][2] == bid):
            count += 1
            bid_all_transacting = False
        buyer_in += 1

    return bid_total_surplus/count, bid_all_transacting

def expected_seller_surplus(true_price: int, ask: int, epsilon: float, buyers: list[list[int]], sellers: list[list[int]]) -> Tuple[float, bool]:
    '''Calculates the expected surplus of a seller given an ask. We also track if there were sellers at that ask who
    did not transact.'''
    buyers.sort(key=lambda x: x[2], reverse=True)
    sellers.sort(key=lambda x: x[2], reverse=True)
    buyers_length = len(buyers)
    sellers_length = len(sellers)
    buyer_in = 0
    seller_in = sellers_length - 1
    ask_total_surplus = 0
    count = 0
    market_transacts = False

    while buyer_in < buyers_length and seller_in >= 0 and buyers[buyer_in][2] >= sellers[seller_in][2]:
        if (sellers[seller_in][2] == ask):
            count += 1
        market_transacts = True
        buyer_in += 1
        seller_in -= 1
    
    # we only really care about the clearing price
    if market_transacts:
        buyer_in -= 1
        seller_in += 1
        clearing_price = midpoint(buyers[buyer_in][2], sellers[seller_in][2])
        ask_total_surplus = count * (clearing_price - true_price + epsilon)
        # reset the indices for the next operation
        buyer_in += 1
        seller_in -= 1

    ask_all_transacting = True
    # there may be sellers at this ask we have not accounted for yet
    while seller_in >= 0 and sellers[seller_in][2] <= ask:
        if (sellers[seller_in][2] == ask):
            count += 1
            ask_all_transacting = False
        seller_in -= 1

    return ask_total_surplus/count, ask_all_transacting

def calculate_best_bid(buyer_uuid: int, epsilon: float, buyers: list[list[int]], sellers: list[list[int]]) -> bool:
    '''Calculates valid bid that maximizes expected value of trade for the buyer. We iterate through all possible bids.
    We know the min bid to consider is the lowest ask since all lower bids result in 0 surplus.
    We do NOT restrict the max bid to their true cost. Instead, since there is one clearing price for the market,
    and we assume that sellers are always present, we can stop when the buyer is guaranteed to transact, aka there are no buyers
    with such a bid that do not transact. We can also terminate when the clearing price is too high which results in a negative expected surplus. Then we choose the bid that maximizes expected surplus. If there is no bid that gives positive surplus
    (the agent has no incentive to deviate), we increment the bid by 1 if possible.'''

    buyers.sort(key=lambda x: x[2], reverse=True)
    sellers.sort(key=lambda x: x[2], reverse=True)
    buyer = find_buyer(buyers, buyer_uuid)
    
    original_bid = buyer[2]
    best_bid = original_bid
    true_cost = buyer[1]
    max_expected_surplus, _ = expected_buyer_surplus(true_cost, original_bid, epsilon, buyers, sellers)
    min_cost = sellers[len(sellers)-1][2] # lowest seller ask

    bid_all_transacting = False
    buyer[2] = min_cost
    while not bid_all_transacting:
        expected_surplus, bid_all_transacting = expected_buyer_surplus(true_cost, buyer[2], epsilon, buyers, sellers)
        if expected_surplus < 0:
            # if it occurs that the current bid leads to a negative surplus because of changing market conditions, revert to true cost
            if max_expected_surplus < 0:
                best_bid = buyer[1]
            break
        if expected_surplus > max_expected_surplus:
            max_expected_surplus = expected_surplus
            best_bid = buyer[2]
        buyer[2] += 1

    '''If there is no bid that results in a positive expected surplus, we increment the bid by 1,
    as long as it is less than the true cost. This represents the belief that presenting a higher bid
    will, in the future, give a higher probability of transacting. It is the only behavior that is
    non-myopic. For the game, it serves the purpose of making sure agents aren't stuck at their initial
    deviations. Note here we DO restrict the +1 movement to the true_cost. This is because the buyer only
    increases their bid in such a way if all other rational bids result in no transaction. If a seller were to match
    their new bid in a future round, and it was higher than their true cost, that would result in a clearing
    price too high for this buyer.'''
    if max_expected_surplus == 0 and original_bid < true_cost:
        best_bid = original_bid + 1

    buyer[2] = best_bid

    if DEBUG_PRINT and best_bid != original_bid:
        print("Buyer {} changes original bid {} to {}".format(buyer[0], original_bid, best_bid))   
        # pdb.set_trace()

    return best_bid != original_bid

def calculate_best_ask(seller_uuid: int, epsilon: float, buyers: list[list[int]], sellers: list[list[int]]) -> bool:
    '''Calculates valid ask that maximizes expected value of trade for the seller. We iterate through all possible asks.
    We know the max ask to consider is the highest bid since all higher asks result in 0 surplus.
    We do NOT restrict the min ask to their true price. Instead, since there is one clearing price for the market,
    and we assume that buyers are always present, we can stop when the seller is guaranteed to transact. That is,
    there are no sellers with such an ask that do not transact. We can also terminate when the clearing price is too
    low which results in a negative expected surplus. Then we choose the ask that maximizes expected surplus.
    If there is no ask that gives positive expected surplus, we decrement the ask by 1 if possible.'''

    buyers.sort(key=lambda x: x[2], reverse=True)
    sellers.sort(key=lambda x: x[2], reverse=True)
    seller = find_seller(sellers, seller_uuid)
    
    original_ask = seller[2]
    best_ask = original_ask
    true_price = seller[1]
    max_expected_surplus, _ = expected_seller_surplus(true_price, original_ask, epsilon, buyers, sellers)
    max_price = buyers[0][2] # highest bid

    ask_all_transacting = False
    seller[2] = max_price
    while not ask_all_transacting:
        expected_surplus, ask_all_transacting = expected_seller_surplus(true_price, seller[2], epsilon, buyers, sellers)
        if expected_surplus < 0:
            # if it occurs that the current ask leads to a negative surplus because of changing market conditions, revert to true price
            if max_expected_surplus < 0:
                best_ask = seller[1]
            break
        if expected_surplus > max_expected_surplus:
            max_expected_surplus = expected_surplus
            best_ask = seller[2]
        seller[2] -= 1

    '''If there is no ask that results in a positive expected surplus, we decrement the ask by 1,
    as long as it is less than the true cost. This represents the belief that presenting a lower ask
    will, in the future, give a higher probability of transacting. It is the only behavior that is
    non-myopic. For the game, it serves the purpose of making sure agents aren't stuck at their initial
    deviations. Note here we DO restrict the -1 movement to the true_price. This is because the seller 
    only decreases their ask in such a way if all the other rational asks result in no transaction.
    If a buyer were to match their new ask in a future round, and it was lower than their true price,
    that would result in a clearing price too low for this seller.'''
    if max_expected_surplus == 0 and original_ask > true_price:
        best_ask = original_ask - 1

    seller[2] = best_ask

    if DEBUG_PRINT and best_ask != original_ask:
        print("Seller {} changes original ask {} to {}".format(seller[0], original_ask, best_ask))   
        # pdb.set_trace()

    return best_ask != original_ask

def myopic_unilateral_deviation(epsilon: float, buyers: list[list[int]], sellers: list[list[int]]) -> bool:
    '''We are given a buyers list and sellers list. We pick randomly an agent, through UUID, to make their move. 
    If this agent is the same as the last turn, then we move on since we have already picked their optimal move.
    If the agent is a buyer, we calculate the best bid for them. If the agent is a seller, we calculate the best ask for them.
    We continue this for 1000 iterations (there are 1000 picks of an agent) or until we reach convergence.
    To find when we have reached convergence we check when the length of inactive_agents is equal to the length of buyers + sellers.
    Each entry in inactive_agents is correlated to the UIUD of an agent. The sellers are indexed from buyers_length to buyers_length + sellers_length (so we don't have to generate multiple numbers). For example, if there are 5 buyers then 5 correlates to the seller with UUID 0. If any agent makes a move, then we clear the whole set of inactive agents.
    This is because one agent's move can make previously inactive agents active again. If we reach 1000 iterations, we return False, otherwise we return True.'''

    buyers_length = len(buyers)
    sellers_length = len(sellers)

    inactive_agents: set[int] = set()

    counter = 0
    last_agent = -1

    iteration_threshold = 1000

    while (len(inactive_agents) < buyers_length + sellers_length) and (counter < iteration_threshold):
        curr_agent = random.randint(0, buyers_length + sellers_length - 1) # will refer to UUID of buyer, or UIUD of seller + len(buyers)

        if DEBUG_PRINT:
            print("cnt: {}; agt: {}".format(counter, curr_agent))

        if curr_agent == last_agent:
            continue

        last_agent = curr_agent
        if curr_agent < buyers_length:
            if calculate_best_bid(curr_agent, epsilon, buyers, sellers):
                inactive_agents.clear()
                buyers.sort(key=lambda x: x[2], reverse=True)
            else:
                inactive_agents.add(curr_agent)
        else:
            seller_uiud = curr_agent - buyers_length
            if calculate_best_ask(seller_uiud, epsilon, buyers, sellers):
                inactive_agents.clear()
                sellers.sort(key=lambda x: x[2], reverse=True)
            else:
                inactive_agents.add(curr_agent)

        counter += 1
    
    if DEBUG_PRINT:
        if len(inactive_agents) == buyers_length + sellers_length:
            print("No more unilateral deviation")
        else:
            print("Convergence not reached")
        print(buyers)
        print(sellers)

    return counter < iteration_threshold

def simulation(num_buyers: int, num_sellers: int, min_buy: int, max_buy: int, min_sell: int, max_sell: int) -> Tuple[int, int, list[int], list[int], list[list[int|None]], list[list[int|None]], int]:
    '''Creates a market with num_buyers buyers and num_sellers sellers. The true costs are randomly generated between min_buy and max_buy.
    The true prices are randomly generated from min_sell to max_sell. The bids and asks are initially the same as the true costs and prices (before initial deviation).
    Each buyer is an array of type [uuid, true_cost, bid]. Each seller is an array of type [uuid, true_price, ask].
    We then calculate the market surplus and number of trades before any deviation. We then deviate the bids and asks randomly (initial deviation).
    We deviate the bids and asks randomly 10 times (deep copy the lists beforehand). Each time we deviate we do the following:
        We recalculate the market surplus and number of trades since this can be drastically different.
        We then run the myopic_unilateral_deviation function 10 times. Each time we deep copy the buyers and sellers array.
        This is so that we have 10 iterations where we start with random agents (testing to see if order matters).
        We record the surplus and number of trades after each deviation in an array (including how many times we did not reach equilibrium).'''
    buyers = [[i, value := random.randint(min_buy, max_buy), value] for i in range(num_buyers)]
    sellers = [[i, value := random.randint(min_sell, max_sell), value] for i in range(num_sellers)]

    surplus_b4_dev, trades_b4_dev = calculate_market_surplus(buyers, sellers)

    times_no_equilibrium = 0

    # market surplus is always an int, not a float
    surplus_after_rand: list[int] = [] # each element represents surplus after an initial deviation
    trades_after_rand: list[int] = [] # each element represents number of trades after an initial deviation

    surplus_post_dev: list[list[int|None]] = [] # each element is list (see surplus_post_dev_iteration)
    trades_post_dev: list[list[int|None]]= [] # each element is list (see trades_post_dev_iteration)

    threshold_1 = 100
    for _ in range(threshold_1):
        # logging.info("Initial Deviation {}".format(i))
        buyers_copy_1 = copy.deepcopy(buyers) # deep copy so we can test for multiple intial deviations
        sellers_copy_1 = copy.deepcopy(sellers) # deep copy so we can test for multiple intial deviations
        initial_deviation(buyers_copy_1, sellers_copy_1, min_buy, max_sell)
        surplus_after_rand_iteration, trades_after_rand_iteration = calculate_market_surplus(buyers_copy_1, sellers_copy_1)

        surplus_after_rand.append(surplus_after_rand_iteration)
        trades_after_rand.append(trades_after_rand_iteration)

        if DEBUG_PRINT:
            print(surplus_b4_dev, trades_b4_dev)
            print(buyers_copy_1, sellers_copy_1)
            print(surplus_after_rand_iteration, trades_after_rand_iteration)
            pdb.set_trace()
        
        surplus_post_dev_iteration: list[int|None] = [] # each element represents surplus for a random agent order
        trades_post_dev_iteration: list[int|None] = [] # each element represents number of trades for a random agent order
        threshold_2 = 100
        for j in range(threshold_2):
            buyers_copy_2 = copy.deepcopy(buyers_copy_1) # deep copy so we can test for multiple random agent orders
            sellers_copy_2 = copy.deepcopy(sellers_copy_1) # deep copy so we can test for multiple random agent orders
            
            equilibrium_reached = myopic_unilateral_deviation(2/3, buyers_copy_2, sellers_copy_2)
            if equilibrium_reached:
                surplus_after_dev, trades_after_dev = calculate_market_surplus(buyers_copy_2, sellers_copy_2)
                if surplus_after_dev != surplus_b4_dev or trades_after_dev != trades_b4_dev:
                    logging.info("DIFFERENCE IN SURPLUS/NUMBER OF TRADES")
                    logging.info(f"Buyers before initial deviation: {buyers}")
                    logging.info(f"Sellers before initial deviation: {sellers}")
                    logging.info(f"Buyers after myopic deviation: {buyers_copy_2}")
                    logging.info(f"Sellers after myopic deviation: {sellers_copy_2}")
                    logging.info(f"Surplus before initial deviation:  {surplus_b4_dev}")
                    logging.info(f"Surplus after myopic deviation:  {surplus_after_dev}")
                    logging.info(f"Number of trades before initial deviation:  {trades_b4_dev}")
                    logging.info(f"Number of trades after myopic deviation:  {trades_after_dev}")

                if DEBUG_PRINT:
                    print("Iter {}: {} vs {}; {} vs {}".format(j, surplus_after_dev, surplus_b4_dev, trades_after_dev, trades_b4_dev))
                    # if surplus_after_dev < surplus_b4_dev:
                    #     pdb.set_trace()

                surplus_post_dev_iteration.append(surplus_after_dev)
                trades_post_dev_iteration.append(trades_after_dev)
            else:
                logging.info("NO EQUILIBRIUM REACHED")
                logging.info(f"Buyers before initial deviation: {buyers}")
                logging.info(f"Sellers before initial deviation: {sellers}")
                logging.info(f"Buyers after initial deviation: {buyers_copy_1}")
                logging.info(f"Sellers after initial deviation: {sellers_copy_1}")
                times_no_equilibrium += 1
                surplus_post_dev_iteration.append(None)
                trades_post_dev_iteration.append(None)
        surplus_post_dev.append(surplus_post_dev_iteration)
        trades_post_dev.append(trades_post_dev_iteration)

    return surplus_b4_dev, trades_b4_dev, surplus_after_rand, trades_after_rand, surplus_post_dev, trades_post_dev, times_no_equilibrium

def main():
    '''Runs the market simulation method 10 times with 5 buyers and 5 sellers.
    Each time the simulation is run, 10 initial deviations are set,
    and for each initial deviation, 10 actual simulations are run,
    to create randomness in agent order.'''
    
    # fix a seed to reproduce
    random.seed(10)

    threshold = 100
    for _ in range(threshold):
        simulation(5, 5, 0, 20, 0, 20)

if __name__ == "__main__":
    main()