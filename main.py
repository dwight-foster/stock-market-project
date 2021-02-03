import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import numpy as np
import time
import pandas as pd
from stock_scraper import get_fundamentals
from collections import deque
import random
from model import LSTM


def get_stock(csv):
    price = csv["Values"][70]
    esp_next_q = csv["Values"][26]
    esp_next_y = csv["Values"][28]
    esp_next_five = csv["Values"][29]
    sales_q_q = csv["Values"][32]
    price = torch.tensor(price)
    esp_next_q = torch.tensor(esp_next_q)
    esp_next_y = esp_next_y.split("%")[0]
    esp_next_five = esp_next_five.split("%")[0]
    sales_q_q = sales_q_q.split("%")[0]
    esp_next_y = torch.tensor(esp_next_y)
    esp_next_five = torch.tensor(esp_next_five)
    sales_q_q = torch.tensor(sales_q_q)
    return price, torch.cat([esp_next_q, esp_next_y, esp_next_five, sales_q_q])


def get_data(stocks, prices=None, seq_length=50):
    if prices == None:
        prices = {}
        zeros = [0 for i in range(seq_length)]
        for i in stocks:
            prices[stocks] = deque(zeros, maxlen=seq_length)

    for i in stocks:
        csv = get_fundamentals(i)
        price, feature = get_stock(csv)
        feature = feature.unsqueeze(0)
        price_deque = prices[i]
        price_deque.append(price)
        prices[i] = price_deque
        prices_T = torch.stack([price_deque.unsqueeze(0)], dim=0)
        features = torch.stack([feature], dim=0)
    return prices_T, features, prices

def get_price(stock):
    csv = get_fundamentals(i)
    price, feature = get_stock(csv)
    return price

def get_action(model, stocks, prices=None):
    prices_T, features, prices = get_data(stocks, prices)
    prices_T = prices_T.cuda()
    features = features.cuda()
    pred = model([prices_T, features])
    pred *= 2
    return pred, prices


batch_size = 64
input_sizes = [1, 4]
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
optimizer = optim.Adam(model.parameters(), lr)
criterion = nn.MSE()
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 200, 0.5)
prices = None
stocks_owned = {}
stocks_value = {}
for stock in stocks:
    stocks_owned[stock] = 0
    stocks_value[stock] = 0
start_money = 1000
current_money = start_money
total_value = 1000
for e in epochs:
    reward = 0
    actions, prices = get_action(model, stocks, prices)
    print(actions.shape)
    for i in range(actions.shape[0]):
        if stocks_owned[stocks[i]] - int(actions[i]) < 0:
            reward -= 1
        elif int(actions[i]) < 0:
            price = get_price(stocks[i])
            current_money += (int(actions[i])*-1) * price
            stocks_owned[stocks[i]] += int(actions[i])
            stocks_value[stocks[i]] = stocks_owned[stocks[i]] * price
            total_value = current_money + sum(stocks_value.items())
        else:
            price = get_price(stocks[i])
            cost = price * int(actions[i])
            if current_money - cost < 0:
                reward -= 5
            else:
                stocks_owned[stocks[i]] += int(actions[i])
                stocks_value[stocks[i]] = stocks_owned[stocks[i]] * price
                total_value = current_money + sum(stocks_value.items())
