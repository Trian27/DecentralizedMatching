TRUE_VALUE = 10 # true cost of both buyers
buyers = [[0,6], [1,5]] # [buyer_uuid, bid]
sellers = [[0,6], [1,1]] # [seller_uuid, ask]
turn = 1

def max_bid(buyers, sellers, buyer_uuid):
    buyer = []
    for curr_buyer in buyers:
        if curr_buyer[0] == buyer_uuid:
            buyer = curr_buyer
            break
    # process above is to find the buyer with the given buyer_uuid
    
    best_bid = buyer[1]
    max_expected_surplus = 0

    for i in range (11):
        buyer[1] = i
        buyers.sort(key=lambda x: x[1], reverse=True)

        buyer_index = 0
        seller_index = 1

        bid_total_surplus = 0
        count = 0

        while buyer_index < 2 and seller_index >= 0 and buyers[buyer_index][1] >= sellers[seller_index][1]:
            if (buyers[buyer_index][1] == i):
                bid_total_surplus += TRUE_VALUE - (i + sellers[seller_index][1])/2
            buyer_index += 1
            seller_index -= 1
        
        for j in range(2):
            if (buyers[j][1] == i):
                count += 1

        expected_surplus = bid_total_surplus / count
        if expected_surplus > max_expected_surplus:
            max_expected_surplus = expected_surplus
            best_bid = i

    buyer[1] = best_bid

previous_bids = set()

while True:
        
    max_bid(buyers, sellers, turn)
    turn = (turn+1)%2
    
    buyers.sort(key=lambda x: x[1], reverse=True)
    print(buyers)

    node = ((buyers[0][0],buyers[0][1]),(buyers[1][0],buyers[1][1]))
    if node in previous_bids:
        print("Loop found")
        break
    previous_bids.add(node)