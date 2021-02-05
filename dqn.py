from model import *
from stock_scraper import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from collections import deque


def get_stock(csv):
    price = csv["Values"][70]
    esp_next_q = csv["Values"][26]
    esp_next_y = csv["Values"][28]
    esp_next_five = csv["Values"][29]
    sales_q_q = csv["Values"][32]
    if esp_next_q == "-":
        esp_next_q = 0

    if esp_next_y == "-":
        esp_next_y = 0
    else:
        esp_next_y = float(esp_next_y.split("%")[0])
    if esp_next_five == "-":
        esp_next_five = 0.
    else:
        esp_next_five = float(esp_next_five.split("%")[0])
    if sales_q_q == "-":
        sales_q_q = 0
    else:
        sales_q_q = float(sales_q_q.split("%")[0])
    features = torch.tensor([esp_next_y, esp_next_five, sales_q_q, float(esp_next_q)])
    price = torch.tensor(float(price))
    return price, features


class DQN:
    def __init__(self, model, lr, stocks, gamma=0.99, epsilon=0.95, epsilon_decay=0.1):
        self.model = model
        self.lr = lr
        self.optimizer = optim.Adam(model.parameters(), self.lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.prices = None
        self.stocks_owned = {}
        self.stocks_value = {}
        self.stocks = stocks
        for stock in self.stocks:
            self.stocks_owned[stock] = 0
            self.stocks_value[stock] = 0
        self.usable_stocks = 0
        self.start_money = 1000
        self.current_money = self.start_money
        self.total_value = 1000
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, 200, 0.5)

    def init_values(self):
        for i in range(self.usable_stocks):
            print(self.usable_stocks)
            stock = self.stocks[i]
            self.stocks_owned[stock] = 0
            self.stocks_value[stock] = 0

    def get_data(self, stocks, prices=None, seq_length=50):
        if prices == None:
            prices = {}
            zeros = [0 for i in range(seq_length)]
            for i in stocks:
                prices[i] = deque(zeros, maxlen=seq_length)
        prices_T = []
        features = []
        for i in stocks:
            csv, result = get_fundamentals(i)
            if result == 0:
                continue
            price, feature = get_stock(csv)

            feature = feature.unsqueeze(0)
            price_deque = prices[i]
            price_deque.append(price)
            prices[i] = price_deque
            price_deque = torch.tensor([price_deque])
            prices_T.append(price_deque)
            features.append(feature)
        prices_T = torch.stack(prices_T, dim=0)
        features = torch.stack(features, dim=0)
        return prices_T.float(), features.squeeze(1), prices

    def get_price(self, stock):
        csv = get_fundamentals(stock)
        price, feature = get_stock(csv)
        return price

    def get_action(self, model, stocks, current_stocks, hidden, prices=None):
        prices_T, features, prices = self.get_data(stocks, prices)
        prices_T = prices_T.cuda()
        features = features.cuda()
        current_stocks = current_stocks.cuda().view(current_stocks.shape[1],  current_stocks.shape[0])
        features = torch.cat([features, current_stocks], dim=1)
        print(features.shape)
        prices_T = torch.reshape(prices_T, (prices_T.shape[2], prices_T.shape[0], 1))
        pred, hidden = self.model([prices_T, features], hidden)
        pred *= 2
        return pred, hidden, prices

    def run(self, reward, hidden):
        actions, hidden, self.prices = self.get_action(self.model, self.stocks,
                                               torch.tensor([list(self.stocks_owned.values())]), hidden, self.prices)
        print(actions.shape)
        for i in range(actions.shape[0]):
            if self.stocks_owned[self.stocks[i]] - int(actions[i]) < 0:
                self.reward -= 1
            elif int(actions[i]) < 0:
                price = self.get_price(self.stocks[i])
                self.current_money += (int(actions[i]) * -1) * price
                self.stocks_owned[self.stocks[i]] += int(actions[i])
                self.stocks_value[self.stocks[i]] = self.stocks_owned[self.stocks[i]] * price
                self.total_value = self.current_money + sum(self.stocks_value.items())
            else:
                price = self.get_price(self.stocks[i])
                cost = price * int(self.actions[i])
                if self.current_money - cost < 0:
                    reward -= 5
                else:
                    self.stocks_owned[self.stocks[i]] += int(actions[i])
                    self.stocks_value[self.stocks[i]] = self.stocks_owned[self.stocks[i]] * price
                    self.total_value = self.current_money + sum(self.stocks_value.items())

        returns = self.total_value - self.start_money
        reward += returns
        return reward, hidden

    def compute_loss(self, reward, hidden):
        reward, hidden = self.run(reward, hidden)
        return reward, self.total_value, self.stocks_owned, hidden
