# Using midpoint as clearing price - if clearing price negative just set utility to zero (no transaction will occur)
import random
from typing import Union

def generate_payoffs() -> list[list[int]]:
    true_cost = random.randint(0, 20)
    true_price = random.randint(0, 20)
    payoffs = []
    for i in range(20): # this will be the bids for the buyer
        payoff_i =[]
        for j in range(20): # this will be the asks for the seller
            clearing_price = (i+j)/2
            if i >= j and clearing_price <= true_cost and clearing_price >= true_price:
                payoff_i.append((true_cost - clearing_price, clearing_price - true_price))
            else:
                payoff_i.append((0, 0))
        payoffs.append(payoff_i)
    return payoffs

def exact_potentials(payoffs) -> Union[bool, list[list[int]]]:
    # Will do in two passes
    # First pass - fill in by row by row first
    # Second pass - make sure every direction is consistent
    potentials = [[0 for i in range(20)] for j in range(20)]
    potentials[0][0] = 0

    # First pass, taking potentials, calculating by left
    for i in range(20): # 20x20 payoff matrix
        for j in range(i):
            if i == 0 and j != 0:
                potentials[i][j] = potentials[i][j-1] + payoffs[i][j][1] - payoffs[i][j-1][1]
            else:
                potentials[i][j] = potentials[i-1][j] + payoffs[i][j][0] - payoffs[i-1][j][0]
                
    # Second pass, checking consistency
    for i in range(20):
        for j in range(20):
            left = None
            up = None 
            right = None 
            down = None
            if i > 0:
                up = potentials[i-1][j] + payoffs[i][j][0] - payoffs[i-1][j][0]
            if i < 19:
                down = potentials[i+1][j] + payoffs[i][j][0] - payoffs[i+1][j][0]
            if j > 0:
                left = potentials[i][j-1] + payoffs[i][j][1] - payoffs[i][j-1][1]
            if j < 19:
                right = potentials[i][j+1] + payoffs[i][j][1] - payoffs[i][j+1][1]

            # Have to check consistency of directions while being mindful if any of them are None
            valid_vars = set()
            if up is not None:
                valid_vars.add(up)
            if down is not None:
                valid_vars.add(down)
            if left is not None:
                valid_vars.add(left)
            if right is not None:
                valid_vars.add(right)
            
            if len(valid_vars) > 1:
                return False, potentials

    return True, potentials
    
def main():
    times_no_exact = 0
    for i in range(50):
        payoffs = generate_payoffs()
        is_exact, potentials = exact_potentials(payoffs)
        if not is_exact:
            times_no_exact += 1
            print("Payoffs: ", payoffs)
        else:
            print("Potentials: ", potentials)
    print("Times no exact: ", times_no_exact)

if __name__ == "__main__":
    main()