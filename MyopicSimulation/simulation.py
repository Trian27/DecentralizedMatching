import random, copy
from typing import Union, Tuple
import pdb
import logging
logging.basicConfig(filename='simulation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEBUG_PRINT = False

def clearing_price(bid: int, ask: int) -> int:
    '''Calculates the clearing price given a bid and ask. Right now is set at the midpoint.'''
    return (bid + ask) / 2

def initial_deviation(buyers: list, sellers: list, min_buy: int, max_sell: int) -> None:
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

def calculate_market_surplus(buyers: list, sellers: list) -> Tuple[int, int]:
    '''Calculates the market surplus and number of trades given a list of buyers and sellers.
    Market surplus is calculated as the sum of the surplus of each trade (see appendix for definition). We determine trades 
    by matching the highest bid with the lowest ask, the second highest bid with the second lowest ask, etc,
    until there are no more possible trades. We also record the number of trades that take place.
    Notably, the surplus calculation does not depend on the clearing price (see appendix for proof).'''

    # This will be the true shuffling and sorting algorithm before matching
    # random.shuffle(buyers)
    # random.shuffle(sellers)
    # buyers.sort(key=lambda x: (x[2]), reverse=True)
    # sellers.sort(key=lambda x: (x[2]), reverse=True)

    # This is the sorting algorithm that we will use for the simulation to ensure reproducibility
    buyers.sort(key=lambda x: (x[2],x[1]), reverse=True)
    sellers.sort(key=lambda x: (x[2],x[1]), reverse=True)

    buyers_length = len(buyers)
    sellers_length = len(sellers)
    surplus = 0
    buyer_in = 0
    seller_in = sellers_length - 1
    while buyer_in < buyers_length and seller_in >= 0 and buyers[buyer_in][2] >= sellers[seller_in][2]: # only transact when there are still buyers and sellers and the buyers bid is higher than the seller's ask
        surplus += buyers[buyer_in][1] - sellers[seller_in][1] # total surplus of transaction is the difference between the true cost and the true price
        buyer_in += 1
        seller_in -= 1
    num_trades = buyer_in
    return surplus, num_trades

def expected_buyer_surplus(true_cost: int, bid: int, buyers: list, sellers: list) -> float:
    '''Calculates the expected surplus of a buyer given a bid. We have a list of all buyers and sellers.
    Then we do our market clearing function, and for each bid/ask pair with the same bid as input, we calculate the surplus for the given buyer.
    We add that up. Then we divide by the amount of times their bid appears in the buyers list. This is their expected surplus at this bid. '''
    buyers.sort(key=lambda x: x[2], reverse=True)
    sellers.sort(key=lambda x: x[2], reverse=True)
    buyers_length = len(buyers)
    sellers_length = len(sellers)
    buyer_in = 0
    seller_in = sellers_length - 1
    bid_total_surplus = 0
    count = 0

    while buyer_in < buyers_length and seller_in >= 0 and buyers[buyer_in][2] >= sellers[seller_in][2]:
        if (buyers[buyer_in][2] == bid):
            bid_total_surplus += true_cost - clearing_price(bid, sellers[seller_in][2]) # adding up surplus for each transaction involving this bid price.
        buyer_in += 1
        seller_in -= 1
    
    for i in range(buyers_length):
        if (buyers[i][2] == bid): # count how many times this bid appears in the buyers list
            count += 1

    return (bid_total_surplus/count)
    

def expected_seller_surplus(true_price: int, ask: int, buyers: list, sellers: list) -> float:
    '''Calculates the expected surplus of a seller given an ask. We have a list of all buyers and sellers.
    Then we do our market clearing function, and for each bid/ask pair with the same ask as input, we calculate the surplus for the given seller.
    We add that up. Then we divide by the amount of times their ask appears in the sellers list. This is their expected surplus at this ask. '''
    buyers.sort(key=lambda x: x[2], reverse=True)
    sellers.sort(key=lambda x: x[2], reverse=True)
    buyers_length = len(buyers)
    sellers_length = len(sellers)
    buyer_in = 0
    seller_in = sellers_length - 1
    ask_total_surplus = 0
    count = 0

    while buyer_in < buyers_length and seller_in >= 0 and buyers[buyer_in][2] >= sellers[seller_in][2]: # adding up surplus for each transaction involving this ask price.
        if (sellers[seller_in][2] == ask):
            ask_total_surplus += clearing_price(buyers[buyer_in][2], ask) - true_price

        buyer_in += 1
        seller_in -= 1
    
    for i in range(sellers_length):
        if (sellers[i][2] == ask): # count how many times this ask appears in the sellers list
            count += 1

    return (ask_total_surplus/count)

def calculate_best_bid(buyer_uiud: int, buyers: list, sellers: list) -> bool:
    '''Calculates valid bid that maximizes expected value of trade for buyer. We iterate through all possible bids.
    We know the max bid is their true cost. We also know their min bid is the lowest seller's ask (since if they bid lower they cannot transact).
    Then we calculate the expected surplus for each bid and choose the bid that maximizes expected surplus.
    If there is no bid that gives positive surplus (the agent has no incentive to deviate), we increment the bid by 1 if possible.'''
    buyer = []
    for curr_buyer in buyers:
        if curr_buyer[0] == buyer_uiud:
            buyer = curr_buyer
            break
    
    original_bid = buyer[2]
    best_bid = original_bid
    true_cost = buyer[1]
    max_expected_surplus = 0
    min_cost = sellers[len(sellers)-1][2] # lowest seller ask

    for bid in range(min_cost, true_cost+1):
        buyer[2] = bid
        expected_surplus = expected_buyer_surplus(true_cost, bid, buyers, sellers)
        if expected_surplus > max_expected_surplus:
            max_expected_surplus = expected_surplus
            best_bid = bid
        
    if max_expected_surplus == 0 and original_bid < true_cost: # if no positive surplus position, increment bid by 1 (as long as it is less than true cost)
            best_bid += 1

    buyer[2] = best_bid

    if DEBUG_PRINT and best_bid != original_bid:
        print("Buyer {} changes original bid {} to {}".format(buyer[0], original_bid, best_bid))   
        # pdb.set_trace()

    return best_bid != original_bid

def calculate_best_ask(seller_uiud: int, buyers: list, sellers: list) -> bool:
    '''Calculates valid bid that maximizes expected value of trade for seller. We iterate through all possible asks.
    We know the min ask is their true price. We also know their max ask is the highest bidder's bid (since if they ask higher they cannot transact).
    Then we calculate the expected surplus for each ask and choose the ask that maximizes expected surplus.
    If there is no ask that gives positive surplus (the agent has no incentive to deviate), we decrement the ask by 1 if possible.'''

    seller = []
    for curr_seller in sellers:
        if curr_seller[0] == seller_uiud:
            seller = curr_seller
            break

    original_ask = seller[2]
    best_ask = original_ask
    true_price = seller[1] 
    max_expected_surplus = 0
    max_price = buyers[0][2] # highest buyer bid

    for ask in range(true_price, max_price+1):
        seller[2] = ask
        expected_surplus = expected_seller_surplus(true_price, ask, buyers, sellers)
        if expected_surplus > max_expected_surplus:
            max_expected_surplus = expected_surplus
            best_ask = ask

    if max_expected_surplus == 0 and original_ask > true_price: # if no positive surplus position, decrement ask by 1 (as long as it is greater than true price)
        best_ask -= 1

    seller[2] = best_ask

    if DEBUG_PRINT and best_ask != original_ask:
        print("Seller {} changes original ask {} to {}".format(seller[0], original_ask, best_ask))   
        # pdb.set_trace()

    return best_ask != original_ask

def myopic_unilateral_deviation(buyers: list, sellers: list) -> bool:
    '''We are given a buyers list and sellers list. We pick randomly an agent, through UUID, to make their move. 
    If this agent is the same as the last turn, then we move on since we have already picked their optimal move.
    If the agent is a buyer, we calculate the best bid for them. If the agent is a seller, we calculate the best ask for them.
    We continue this for 1000 iterations (there are 1000 picks of an agent) or until we reach convergence.
    To find when we have reached convergence we check when the length of inactive_agents is equal to the length of buyers + sellers.
    Each entry in inactive_agents is correlated to the UIUD of an agent. The sellers are indexed from buyers_length to buyers_length + sellers_length (so we don't have to generate multiple numbers).
    For example, if there are 5 buyers then 5 correlates to the seller with UUID 0. If any agent makes a move, then we clear the whole set of inactive agents.
    This is because one agent's move can make previously inactive agents active again. If we reach 1000 iterations, we return False, otherwise we return True.'''

    buyers_length = len(buyers)
    sellers_length = len(sellers)

    inactive_agents = set()

    counter = 0
    last_agent = -1

    iteration_threshold = 1000


    while (len(inactive_agents) < buyers_length + sellers_length) and (counter < iteration_threshold):
        curr_agent = random.randint(0, buyers_length + sellers_length - 1) # will refer to UUID of buyer, or UIUD of seller + len(buyers)
        if DEBUG_PRINT:
            print("cnt: {}; agt: {}".format(counter, curr_agent))

        if curr_agent == last_agent:
            counter += 1
            continue

        last_agent = curr_agent
        if curr_agent < buyers_length:
            best_bid = buyers[curr_agent][2]
            if calculate_best_bid(curr_agent, buyers, sellers):
                inactive_agents.clear()
                buyers.sort(key=lambda x: x[2], reverse=True)
            else:
                inactive_agents.add(curr_agent)
        else:
            seller_uiud = curr_agent - buyers_length
            best_ask = sellers[seller_uiud][2]
            if calculate_best_ask(seller_uiud, buyers, sellers):
                inactive_agents.clear()
                sellers.sort(key=lambda x: x[2], reverse=True)
            else:
                inactive_agents.add(curr_agent)

        counter += 1
    
    if DEBUG_PRINT:
        print("No more unilateral deviation")
        print(buyers)
        print(sellers)
    return counter < iteration_threshold 


def simulation(num_buyers: int, num_sellers: int, min_buy: int, max_buy: int, min_sell: int, max_sell: int) -> Tuple[int, int, list, list, list[list], list[list], int]:
    '''Creates a market with num_buyers buyers and num_sellers sellers. The true costs are randomly generated between min_bid and max_bid.
    The true prices are randomly generated from min_sell to max_sell. The bids and asks are initially the same as the true costs and prices (before initial deviation).
    Each buyer is an array of type [uuid, true_cost, bid]. Each seller is an array of type [index, true_price, ask].
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

    surplus_after_rand = [] # each element represents surplus after an initial deviation
    trades_after_rand = [] # each element represents number of trades after an initial deviation

    surplus_post_dev = [] # each element is list (see surplus_post_dev_iteration)
    trades_post_dev = [] # each element is list (see trades_post_dev_iteration)

    threshold_1 = 100
    for i in range(threshold_1):
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
        
        surplus_post_dev_iteration = [] # each element represents surplus for a random agent order
        trades_post_dev_iteration = [] # each element represents number of trades for a random agent order
        threshold_2 = 100
        for j in range(threshold_2):
            buyers_copy_2 = copy.deepcopy(buyers_copy_1) # deep copy so we can test for multiple random agent orders
            sellers_copy_2 = copy.deepcopy(sellers_copy_1) # deep copy so we can test for multiple random agent orders
            
            equilibrium_reached = myopic_unilateral_deviation(buyers_copy_2, sellers_copy_2)
            if equilibrium_reached:
                surplus_after_dev, trades_after_dev = calculate_market_surplus(buyers_copy_2, sellers_copy_2)

                if DEBUG_PRINT:
                    print("Iter {}: {} vs {}; {} vs {}".format(j, surplus_after_dev, surplus_b4_dev, trades_after_dev, trades_b4_dev))
                    # if surplus_after_dev < surplus_b4_dev:
                    #     pdb.set_trace()

                surplus_post_dev_iteration.append(surplus_after_dev)
                trades_post_dev_iteration.append(trades_after_dev)
            else:
                times_no_equilibrium += 1
                surplus_post_dev_iteration.append(None)
                trades_post_dev_iteration.append(None)
        surplus_post_dev.append(surplus_post_dev_iteration)
        trades_post_dev.append(trades_post_dev_iteration)

    return surplus_b4_dev, trades_b4_dev, surplus_after_rand, trades_after_rand, surplus_post_dev, trades_post_dev, times_no_equilibrium

def main():
    '''We run simulation (method) 100 times to get 100 trials of buyer and seller arrays.
    1000 value and cost profiles; for each profile, 100 simulations with random agent orders.
    Then for each run of the simulation we calculate the average surplus_post_dev/surplus_b4_dev and average trades_post_dev/trades_b4_dev.'''
    
    # fix a seed to reproduce
    random.seed(4)

    threshold = 1000
    for i in range(threshold):
        surplus_b4_dev, trades_b4_dev, surplus_after_rand, trades_after_rand, surplus_post_dev, trades_post_dev, times_no_equilibrium = simulation(5, 5, 0, 20, 0, 20)

        total_surplus = 0
        total_trades = 0

        count_surplus = 0
        count_trades = 0

        after_dev_length = len(surplus_post_dev)

        logging.info(f"Iteration {i}")

        for j in range(after_dev_length):

            total_surplus_per_iteration = 0
            total_trades_per_iteration = 0

            count_surplus_per_iteration = 0
            count_trades_per_iteration = 0

            spj = surplus_post_dev[j]
            tpj = trades_post_dev[j]
            
            spj_length = len(spj)
            for k in range(spj_length):
                if spj[k] is not None:
                    total_surplus_per_iteration += spj[k]
                    count_surplus_per_iteration += 1
                if tpj[k] is not None:
                    total_trades_per_iteration += tpj[k]
                    count_trades_per_iteration += 1
            
            total_surplus += total_surplus_per_iteration
            total_trades += total_trades_per_iteration

            count_surplus += count_surplus_per_iteration
            count_trades += count_trades_per_iteration

            logging.info(f"Statistics for Initial Deviation {j}")
            if count_surplus_per_iteration * surplus_b4_dev != 0:
                logging.info(f"Average surplus_post_dev_{j}/surplus_b4_dev: {total_surplus_per_iteration/(count_surplus_per_iteration * surplus_b4_dev)}")
            else:
                logging.info(f"Average surplus_post_dev{j}/surplus_b4_dev: N/A")
            if count_trades_per_iteration * trades_b4_dev != 0:
                logging.info(f"Average trades_post_dev_{j}/trades_b4_dev: {total_trades_per_iteration/(count_trades_per_iteration * trades_b4_dev)}")
            else:
                logging.info(f"Average trades_post_dev_{j}/trades_b4_dev: N/A")
        
        logging.info(f"Overall Statistics for Iteration {i}")
        if count_surplus * surplus_b4_dev != 0:
            logging.info(f"Average surplus_post_dev_total/surplus_b4_dev: {total_surplus/(count_surplus * surplus_b4_dev)}")
        else:
            logging.info("Average surplus_post_dev_total/surplus_b4_dev: N/A")
        if count_trades * trades_b4_dev != 0:
            logging.info(f"Average trades_post_dev_total/trades_b4_dev: {total_trades/(count_trades * trades_b4_dev)}")
        else:
             logging.info("Average trades_post_dev_total/trades_b4_dev: N/A")
        logging.info(f"Times no equilibrium: {times_no_equilibrium}")

if __name__ == "__main__":
    main()