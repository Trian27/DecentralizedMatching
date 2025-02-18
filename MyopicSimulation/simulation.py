import random, copy
from typing import Union, Tuple
import logging
logging.basicConfig(filename='simulation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            buyers[index][2] = buyers[index][1] - random.randint(min_buy, buyers[index][1])
        if index < sellers_length:
            sellers[index][2] = sellers[index][1] + random.randint(0, max_sell - sellers[index][1])
        index += 1

def calculate_market_surplus(buyers: list, sellers: list) -> Tuple[int, int]:
    '''Calculates the market surplus and number of trades given a list of buyers and sellers.
    Market surplus is calculated as the sum of the surplus of each trade (see appendix for definition). We determine trades 
    by matching the highest bid with the lowest ask, the second highest bid with the second lowest ask, etc,
    until there are no more possible trades. We also record the number of trades that take place.
    Notably, the surplus calculation does not depend on the clearing price (see appendix for proof).'''
    random.shuffle(buyers)
    random.shuffle(sellers)
    buyers.sort(key=lambda x: x[2], reverse=True)
    sellers.sort(key=lambda x: x[2], reverse=True)
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
    Then we calculate the expected surplus for each bid and choose the bid that maximizes expected surplus.'''

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

    buyer[2] = best_bid

    return best_bid != original_bid

def calculate_best_ask(seller_uiud: int, buyers: list, sellers: list) -> bool:
    '''Calculates valid bid that maximizes expected value of trade for seller. We iterate through all possible asks.
    We know the min ask is their true price. We also know their max ask is the highest bidder's bid (since if they ask higher they cannot transact).
    Then we calculate the expected surplus for each ask and choose the ask that maximizes expected surplus.'''

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

    seller[2] = best_ask

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
        if curr_agent == last_agent:
            continue
        last_agent = curr_agent
        if curr_agent < buyers_length:

            if calculate_best_bid(curr_agent, buyers, sellers):
                inactive_agents.clear()
                buyers.sort(key=lambda x: x[2], reverse=True)
            else:
                inactive_agents.add(curr_agent)
        else:
            seller_uiud = curr_agent - buyers_length
            if calculate_best_ask(seller_uiud, buyers, sellers):
                inactive_agents.clear()
                sellers.sort(key=lambda x: x[2], reverse=True)
            else:
                inactive_agents.add(curr_agent)

        counter += 1
        
    return counter < iteration_threshold 


def five_buy_and_sell() -> Tuple[int, int, int, int, list, list]:
    '''Creates a market with 5 buyers and 5 sellers. The true costs are randomly generated between 0 and 15.
    The true prices are randomly generated from 5 to 20. The bids and asks are initially the same as the true costs and prices (before initial deviation).
    Each buyer is an array of type [uuid, true_cost, bid]. Each seller is an array of type [index, true_price, ask].
    We then calculate the market surplus and number of trades before any deviation. We then deviate the bids and asks randomly (initial deviation).
    Then we recalculate the market surplus and number of trades since this can be drastically different.
    We then run the myopic_unilateral_deviation function 1000 times. Each time we deep copy the buyers and sellers array.
    This is so that we have 1000 iterations where we start with random agents (testing to see if order matters).
    We record the surplus and number of trades after each deviation in an array (including how many times we did not reach equilibrium).'''
    buyers = [[i, value := random.randint(0, 15), value] for i in range(5)]
    sellers = [[i, value := random.randint(5, 20), value] for i in range(5)]
    surplus_b4_dev, trades_b4_dev = calculate_market_surplus(buyers, sellers)
    times_no_equilibrium = 0

    initial_deviation(buyers, sellers, 0, 20)

    surplus_after_rand, trades_after_rand = calculate_market_surplus(buyers, sellers)

    surplus_post_dev = []
    trades_post_dev = []

    threshold = 1000
    for i in range(threshold):
        buyers_copy = copy.deepcopy(buyers)
        sellers_copy = copy.deepcopy(sellers)
        
        equilibrium_reached = myopic_unilateral_deviation(buyers_copy, sellers_copy)
        if equilibrium_reached:
            surplus_after_dev, trades_after_dev = calculate_market_surplus(buyers_copy, sellers_copy)
            surplus_post_dev.append(surplus_after_dev)
            trades_post_dev.append(trades_after_dev)
        else:
            times_no_equilibrium += 1
            surplus_post_dev.append(None)
            trades_post_dev.append(None)

    return surplus_b4_dev, trades_b4_dev, surplus_after_rand, trades_after_rand, surplus_post_dev, trades_post_dev, times_no_equilibrium

def main():
    '''We run five_buy_and_sell 1000 times to get 1000 trials of buyer and seller arrays.
    Then for each run of the simulation we calculate the average surplus_post_dev/surplus_b4_dev and average trades_post_dev/trades_b4_dev.'''
    threshold = 1000
    for i in range(threshold):
        surplus_b4_dev, trades_b4_dev, surplus_after_rand, trades_after_rand, surplus_post_dev, trades_post_dev, times_no_equilibrium = five_buy_and_sell()

        total_surplus = 0
        total_trades = 0

        count_surplus = 0
        count_trades = 0

        after_dev_length = len(surplus_post_dev)
        for j in range(after_dev_length):
            if surplus_post_dev[j] is not None: # We have reached equilibrium so increment the surplus statistics.
                count_surplus += 1
                total_surplus += surplus_post_dev[j]

            if trades_post_dev[j] is not None: # We have reached equilibrium so increment the trades statistics.
                count_trades += 1
                total_trades += trades_post_dev[j]

        logging.info(f"Iteration {i}")
        if count_surplus * surplus_b4_dev != 0:
            logging.info(f"Average surplus_post_dev/surplus_b4_dev: {total_surplus/(count_surplus * surplus_b4_dev)}")
        else:
            logging.info("Average surplus_post_dev/surplus_b4_dev: N/A") # This can trigger if we never reach equilibrium, there are never any trades or trades result in 0 surplus, or a combo.
        if count_trades * trades_b4_dev != 0:
            logging.info(f"Average trades_post_dev/trades_b4_dev: {total_trades/(count_trades * trades_b4_dev)}")
        else:
            logging.info("Average trades_post_dev/trades_b4_dev: N/A") # This can trigger if we never reach equilibrium, there are never any trades, or a combo.
        logging.info(f"Times no equilibrium: {times_no_equilibrium}")

if __name__ == "__main__":
    main()