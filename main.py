import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import numpy as np
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


def get_data(stocks, prices):
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


batch_size = 64
input_sizes = [50, 4]
hidden_size = 300
num_layers = 2
dropout = 0.5
output_size = 1
lr = 0.0001
model = LSTM(input_sizes, hidden_size, num_layers, dropout, output_size)

csv = pd.read_csv("nasdaq.csv")
stocks = random.choices(csv["Symbol"], k=batch_size)
optimizer = optim.Adam(model.parameters(), lr)
criterion = nn.MSE()