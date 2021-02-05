
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

batch_size = 64
input_sizes = [1, 68]
hidden_size = 300
num_layers = 2
dropout = 0.5
output_size = 1
lr = 0.0001
seq_length = 50
epochs = 100
model = LSTM(input_sizes, hidden_size, num_layers, dropout, output_size)
model.cuda()
csv = pd.read_csv("nasdaq.csv")
stocks = random.choices(csv["Symbol"], k=batch_size)
ppo = DQN(model, lr, stocks)
reward_list = deque(maxlen=100)

for e in tqdm(range(epochs)):
    reward = 0
    reward, profits, stocks_owned = ppo.compute_loss(reward)
    stocks_csv = pd.DataFrame([list(stocks_owned.keys()), list(stocks_owned.values())], columns=["Stocks", "Number Owned"])
    stocks_csv.to_csv("stocks_owned.csv")
    profits_csv = pd.DataFrame([reward, profits], columns=["Reward", "Profits"])
    profits_csv.to_csv("statistics.csv")

    reward_list.append(reward)
    print(f"Reward: {reward}, Mean Reward: {mean(reward_list)}")
    print(f"\nTotal profits: {profits}, stocks owned: {stocks_owned}")
    time.sleep(300)