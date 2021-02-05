import argparse
import pandas as pd
import numpy as np
from stock_scraper import *
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="nasdaq.csv", type=str, help="The name of the csv with the stock list")
args = parser.parse_args()
csv = pd.read_csv(args.csv)
stocks = csv["Symbol"]
useable_stocks =  []
for stock in tqdm(stocks):
    _, result = get_fundamentals(stock)
    if result == 0:
        continue
    else:
        useable_stocks.append(stock)
out = pd.DataFrame(useable_stocks, columns=["Symbol"])
out.to_csv("nasdaq")