import random, copy
from typing import Union, Tuple

def clearing_price(bid: int, ask: int) -> int:
    return (bid + ask) / 2

def initial_deviation(buyers: list, sellers: list) -> None:
    index = 0
    buyers_length = len(buyers)
    sellers_length = len(sellers)
    while index < buyers_length or index < sellers_length:
        if index < buyers_length:
            buyers[index][2] = buyers[index][1] - random.randint(0, buyers[index][1])
        if index < sellers_length:
            sellers[index][2] = sellers[index][1] + random.randint(0, 20 - sellers[index][1])
        index += 1

# This doesn't depend on clearing price. See appendix for proof.
def calculate_market_surplus(buyers: list, sellers: list) -> Tuple[int, int]:
    random.shuffle(buyers)
    random.shuffle(sellers)
    buyers.sort(key=lambda x: x[2], reverse=True)
    sellers.sort(key=lambda x: x[2], reverse=True)
    buyers_length = len(buyers)
    sellers_length = len(sellers)
    surplus = 0
    buyer_in = 0
    seller_in = sellers_length - 1
    while buyer_in < buyers_length and seller_in >= 0 and buyers[buyer_in][2] >= sellers[seller_in][2]:
        surplus += buyers[buyer_in][1] - sellers[seller_in][1]
        buyer_in += 1
        seller_in -= 1
    num_trades = buyer_in
    return surplus, num_trades

def expected_buyer_surplus(true_cost: int, bid: int, buyers: list, sellers: list) -> float:
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
            bid_total_surplus += true_cost - clearing_price(bid, sellers[seller_in][2])
        buyer_in += 1
        seller_in -= 1
    
    for i in range(buyers_length):
        if (buyers[i][2] == bid):
            count += 1
    
    if bid_total_surplus == 0:
        return 0
    
    return (bid_total_surplus/count)
    

def expected_seller_surplus(true_price: int, ask: int, buyers: list, sellers: list) -> float:
    buyers.sort(key=lambda x: x[2], reverse=True)
    sellers.sort(key=lambda x: x[2], reverse=True)
    buyers_length = len(buyers)
    sellers_length = len(sellers)
    buyer_in = 0
    seller_in = sellers_length - 1
    ask_total_surplus = 0
    count = 0

    while buyer_in < buyers_length and seller_in >= 0 and buyers[buyer_in][2] >= sellers[seller_in][2]:
        if (sellers[seller_in][2] == ask):
            ask_total_surplus += clearing_price(buyers[buyer_in][2], ask) - true_price

        buyer_in += 1
        seller_in -= 1
    
    for i in range(sellers_length):
        if (sellers[i][2] == ask):
            count += 1
    
    if ask_total_surplus == 0:
        return 0
    return (ask_total_surplus/count)

def calculate_best_bid(buyer_index: int, buyers: list, sellers: list) -> int:
    '''Calculates valid bid that maximizes expected value of trade for buyer'''
    best_bid = buyers[buyer_index][2]
    true_cost = buyers[buyer_index][1]
    max_expected_surplus = 0
    min_cost = sellers[len(sellers)-1][2]

    for bid in range(min_cost, true_cost+1):
        expected_surplus = expected_buyer_surplus(true_cost, bid, buyers, sellers)
        if expected_surplus > max_expected_surplus:
            max_expected_surplus = expected_surplus
            best_bid = bid

    return best_bid

def calculate_best_ask(seller_index: int, buyers: list, sellers: list) -> int:
    '''Calculates valid ask that maximizes expected value of trade for seller'''
    best_ask = sellers[seller_index][2]
    true_price = sellers[seller_index][1]
    max_expected_surplus = 0
    max_price = buyers[0][2]

    for ask in range(true_price, max_price+1):
        expected_surplus = expected_seller_surplus(true_price, ask, buyers, sellers)
        if expected_surplus > max_expected_surplus:
            max_expected_surplus = expected_surplus
            best_ask = ask

    return best_ask

def myopic_unilateral_deviation(buyers: list, sellers: list) -> bool:
    '''To find when we have reached convergence we check when 
    the length of inactive_agents is equal to the length of buyers + sellers.
    Each entry in inactive_agents is correlated to the UIUD of an agent.'''
    buyers_length = len(buyers)
    sellers_length = len(sellers)

    inactive_agents = set()

    counter = 0
    last_agent = -1

    iteration_threshold = 10000

    while (len(inactive_agents) < buyers_length + sellers_length) and (counter < iteration_threshold):
        curr_agent = random.randint(0, buyers_length + sellers_length - 1)
        if curr_agent == last_agent:
            continue
        last_agent = curr_agent
        if curr_agent < buyers_length:

            best_bid = calculate_best_bid(curr_agent, buyers, sellers)
            if best_bid != buyers[curr_agent][2]:
                buyers[curr_agent][2] = best_bid
                inactive_agents.clear()
                buyers.sort(key=lambda x: x[2], reverse=True)
            else:
                inactive_agents.add(buyers[curr_agent][0])
        else:
            seller_index = curr_agent - buyers_length
            best_ask = calculate_best_ask(seller_index, buyers, sellers)
            if best_ask != sellers[seller_index][2]:
                sellers[seller_index][2] = best_ask
                inactive_agents.clear()
                sellers.sort(key=lambda x: x[2], reverse=True)
            else:
                inactive_agents.add(sellers[seller_index][0]+buyers_length)

        counter += 1
        
    return counter < iteration_threshold 


def five_buy_and_sell() -> Tuple[int, int, int, int, list, list]:
    #buyers are arrays of type [index, true_cost, bid]
    #sellers are arrays of type [index, true_price, ask]
    buyers = [[i, value := random.randint(0, 15), value] for i in range(5)]
    sellers = [[i, value := random.randint(5, 20), value] for i in range(5)]
    surplus_b4_dev, trades_b4_dev = calculate_market_surplus(buyers, sellers)

    initial_deviation(buyers, sellers)

    surplus_after_rand, trades_after_rand = calculate_market_surplus(buyers, sellers)

    surplus_post_dev = []
    trades_post_dev = []

    threshold = 100
    for i in range(threshold):
        buyers_copy = copy.deepcopy(buyers)
        sellers_copy = copy.deepcopy(sellers)
        
        equilibrium_reached = myopic_unilateral_deviation(buyers_copy, sellers_copy)
        if equilibrium_reached:
            surplus_after_dev, trades_after_dev = calculate_market_surplus(buyers_copy, sellers_copy)
            surplus_post_dev.append(surplus_after_dev)
            trades_post_dev.append(trades_after_dev)
        else:
            surplus_post_dev.append(None)
            trades_post_dev.append(None)

    return surplus_b4_dev, trades_b4_dev, surplus_after_rand, trades_after_rand, surplus_post_dev, trades_post_dev

def main():
    threshold = 100
    for i in range(threshold):
        surplus_b4_dev, trades_b4_dev, surplus_after_rand, trades_after_rand, surplus_post_dev, trades_post_dev = five_buy_and_sell()

        total_surplus = 0
        total_trades = 0

        count_surplus = 0
        count_trades = 0

        after_dev_length = len(surplus_post_dev)
        for j in range(after_dev_length):
            if surplus_post_dev[j] is not None:
                count_surplus += 1
                total_surplus += surplus_post_dev[j]

            if trades_post_dev[j] is not None:
                count_trades += 1
                total_trades += trades_post_dev[j]

        print(f"Iteration {i}")
        if count_surplus * surplus_b4_dev != 0:
            print(f"Average surplus_post_dev/surplus_b4_dev: {total_surplus/(count_surplus * surplus_b4_dev)}")
        else:
            print("Average surplus_post_dev/surplus_b4_dev: N/A")
        if count_trades * trades_b4_dev != 0:
            print(f"Average trades_post_dev/trades_b4_dev: {total_trades/(count_trades * trades_b4_dev)}")
        else:
            print("Average trades_post_dev/trades_b4_dev: N/A")


if __name__ == "__main__":
    main()