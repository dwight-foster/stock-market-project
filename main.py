import numpy as np
import time
import pandas as pd
from stock_scraper import get_fundamentals
from collections import deque
import random
from model import LSTM
from dqn import *
from statistics import mean
from tqdm import tqdm
import os
import sys
sys.path.insert(1, './stock-market-project-master')

batch_size = 64
input_sizes = [1, 5, 50]
hidden_size = 300
num_layers = 2
dropout = 0.5
output_size = 5
lr = 0.0001
seq_length = 50
epochs = 100
model = LSTM(input_sizes, hidden_size, num_layers, dropout, output_size)
model.cuda()
csv = pd.read_csv("./stock-market-project-master/nasdaq.csv")
stocks = random.choices(csv["Symbol"], k=batch_size)
hidden = model.init_state(batch_size)
ppo = DQN(model, lr, stocks, output_size, hidden)
reward_list = deque(maxlen=100)
last_profit = 0
for e in tqdm(range(epochs)):
    reward = 0
    reward, profits, stocks_owned, hidden, cash, total, value = ppo.compute_loss(reward, hidden)
    data = {"Stocks": (list(stocks_owned.keys())), "Number Owned": list(stocks_owned.values()), "Value": list(value.values())}
    stocks_csv = pd.DataFrame(data)
    stocks_csv.to_csv("stocks_owned.csv")
    iter_profit = profits-last_profit
    last_profit = profits
    reward_list.append(float(reward.cpu().data))
    print(f"\nReward: {reward}, Mean Reward: {mean(reward_list)}")
    print(f"\nTotal profits: {profits}, Profits this run: {iter_profit}, Money in cash: {cash}, Value in stocks: {total - cash}, Total money: {total}, stocks owned: \n{stocks_csv.head()}")
    # time.sleep(300)
