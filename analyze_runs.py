import os
import json

runs = []
file_list = os.listdir("runs/")
for file in file_list:
    filepath = os.path.join("runs/",file)
    with open(filepath) as f: 
        record = json.load(f)
        runs.append(record)

outcomes = {}
for run in runs:
    outcomes[run["outcome"]] = outcomes.get(run["outcome"], 0) + 1

deal_list = []
for run in runs:
    if run["outcome"] == "deal":
        deal_list.append(run)
seller_win_list = []
for run in deal_list:
    scenario = run["scenario"]
    buyer_reservation = scenario["buyer_reservation"]
    seller_reservation = scenario["seller_reservation"]
    deal_price = run["final_price"]
    zopa = buyer_reservation - seller_reservation
    seller_win = (deal_price - seller_reservation) / zopa
    seller_win_list.append(seller_win)
if len(seller_win_list) == 0:
    avg_seller_win = None   
else: 
    avg_seller_win = sum(seller_win_list) / len(seller_win_list)

print("Outcome counts:", outcomes)
print("Number of deals:", len(deal_list))
print("Average seller share of ZOPA:", avg_seller_win)