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
from yahoo_fin.stock_info import get_day_gainers


batch_size = 3
input_sizes = [1, 6, 20]
hidden_size = 300
num_layers = 2
dropout = 0.5
output_size = 5
lr = 0.0001
seq_length = 20
epochs = 100000
model = LSTM(input_sizes, hidden_size, num_layers, dropout, output_size)
model.cuda()
csv = pd.read_csv("nasdaq.csv")
stocks = {}
num_models = 15
hidden = {}
ppo = {}
reward_list = {}
last_profit = {}
for i in range(num_models):
    hidden[i] = model.init_state(batch_size)
    stocks[i] = random.choices(csv["Symbol"], k=batch_size)
    ppo[i] = DQN(model, lr, stocks[i], output_size, hidden[i], batch_size)
    reward_list[i] = deque(maxlen=100)
    last_profit[i] = 0
rewards = {}
for e in tqdm(range(epochs)):
    for i in range(num_models):
        rewards[i] = 0
        reward, profits, stocks_owned, hidden_layer, cash, total, value, transactions = ppo[i].compute_loss(rewards[i], hidden[i])
        hidden[i] = hidden_layer
        data = {"Stocks": (list(stocks_owned.keys())), "Number Owned": list(stocks_owned.values()), "Value": list(value.values()), "Transactions": transactions}
        stocks_csv = pd.DataFrame(data)
        stocks_csv.to_csv(f"stocks_owned{i+1}.csv")
        iter_profit = profits-last_profit[i]
        last_profit[i] = profits
        reward_list[i].append(float(reward))
        print(f"\nModel {i+1}, Reward: {reward}, Mean Reward: {mean(reward_list[i])}")
        print(f"\nTotal profits: {profits}, Profits this run: {iter_profit}, Money in cash: {cash}, Value in stocks: {total - cash}, Total money: {total}, stocks owned: \n{stocks_csv.head()}")
    # time.sleep(300)
    torch.save(model.state_dict(), "model.pt")