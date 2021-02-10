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

batch_size = 3
input_sizes = [1, 6, 50]
hidden_size = 300
num_layers = 2
dropout = 0.5
output_size = 5
lr = 0.0001
seq_length = 50
epochs = 100
model = LSTM(input_sizes, hidden_size, num_layers, dropout, output_size)
model.cuda()
csv = pd.read_csv("nasdaq.csv")
stocks = random.choices(csv["Symbol"], k=batch_size)
stocks2 = random.choices(csv["Symbol"], k=batch_size)

hidden = model.init_state(batch_size)
hidden2 = model.init_state(batch_size)
ppo = DQN(model, lr, stocks, output_size, hidden, batch_size)
ppo2 = DQN(model, lr, stocks2, output_size, hidden2, batch_size)
reward_list = deque(maxlen=100)
reward_list2 = deque(maxlen=100)
last_profit = 0
last_profit2 = 0
for e in tqdm(range(epochs)):
    reward = 0
    reward2 = 0
    reward, profits, stocks_owned, hidden, cash, total, value, transactions = ppo.compute_loss(reward, hidden)
    data = {"Stocks": (list(stocks_owned.keys())), "Number Owned": list(stocks_owned.values()), "Value": list(value.values()), "Transactions": transactions}
    stocks_csv = pd.DataFrame(data)
    stocks_csv.to_csv("stocks_owned.csv")
    iter_profit = profits-last_profit
    last_profit = profits
    reward_list.append(float(reward))
    print(f"\nModel 1, Reward: {reward}, Mean Reward: {mean(reward_list)}")
    print(f"\nTotal profits: {profits}, Profits this run: {iter_profit}, Money in cash: {cash}, Value in stocks: {total - cash}, Total money: {total}, stocks owned: \n{stocks_csv.head()}")
    # time.sleep(300)
    reward2, profits2, stocks_owned2, hidden2, cash2, total2, value2, transactions2 = ppo2.compute_loss(reward2, hidden2)
    data2 = {"Stocks": (list(stocks_owned2.keys())), "Number Owned": list(stocks_owned2.values()), "Value": list(value2.values()), "Transactions": transactions2}
    stocks_csv2 = pd.DataFrame(data2)
    stocks_csv2.to_csv("stocks_owned_model2.csv")
    iter_profit2 = profits2-last_profit2
    last_profit2 = profits2
    reward_list2.append(float(reward2))
    print(f"\nModel 2, Reward: {reward2}, Mean Reward: {mean(reward_list2)}")
    print(f"\nTotal profits: {profits2}, Profits this run: {iter_profit2}, Money in cash: {cash2}, Value in stocks: {total2 - cash2}, Total money: {total2}, stocks owned: \n{stocks_csv2.head()}")
    torch.save(model.state_dict(), "model.pt")