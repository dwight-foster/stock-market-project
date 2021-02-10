from model import *
from stock_scraper import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from collections import deque, namedtuple
import random


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
    def __init__(self, model, lr, stocks, action_size, hidden,
                 gamma=0.99, epsilon=0.95, epsilon_decay=0.9, TAU=1e-3):
        self.model = model
        self.target_model = model
        self.lr = lr
        self.optimizer = optim.Adam(model.parameters(), self.lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.TAU = TAU
        self.prices = None
        self.stocks_owned = {}
        self.stocks_value = {}
        self.stocks = stocks
        print(len(self.stocks))
        for stock in self.stocks:
            self.stocks_owned[stock] = 0
            self.stocks_value[stock] = 0.
        self.transactions = []
        self.action_size = action_size
        self.target_hidden = hidden
        self.usable_stocks = 0
        self.start_money = 1000
        self.current_money = self.start_money
        self.total_value = 1000
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, 200, 0.5)
        self.possible_actions = [-2, -1, 0, 1, 2]

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
            self.stocks_value[i] = float(price * self.stocks_owned[i])
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
        csv, result = get_fundamentals(stock)
        price, feature = get_stock(csv)
        return price

    def get_action(self, model, stocks, current_stocks, hidden, prices=None):
        prices_T, features, prices = self.get_data(stocks, prices)
        prices_T = prices_T.cuda()
        features = features.cuda()
        current_stocks = current_stocks.cuda().view(current_stocks.shape[1], current_stocks.shape[0])
        features = torch.cat([features, current_stocks], dim=1)
        prices_T = torch.reshape(prices_T, (prices_T.shape[2], prices_T.shape[0], 1))
        with torch.no_grad():
            pred, _ = self.model([prices_T, features], hidden)
        if random.random() > self.epsilon:
            action = pred.cpu().data.max(1, keepdim=True)[1]
            self.update_epsilon()
        else:
            action = [random.choice(np.arange(self.action_size)) for i in range(pred.shape[0])]
            action = torch.tensor(action, dtype=torch.int64).unsqueeze(1)
        return action.detach(), prices, prices_T, features

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def run(self, reward, hidden):
        self.transactions.clear()
        print(len(self.stocks_owned))
        actions, self.prices, prices_T, features = self.get_action(self.model, self.stocks,
                                                                           torch.tensor(
                                                                               [list(self.stocks_owned.values())]),
                                                                           hidden,
                                                                           self.prices)
        for i in range(actions.shape[0]):
            action = self.possible_actions[actions[i]]
            if (int(self.stocks_owned[self.stocks[i]]) + int(action)) < 0:
                self.transactions.append(0)
                reward -= 1
            elif int(action) < 0:
                price = self.get_price(self.stocks[i])
                self.transactions.append(action)
                self.current_money += (int(action) * -1) * price
                self.stocks_owned[self.stocks[i]] += int(action)
                self.stocks_value[self.stocks[i]] = float(self.stocks_owned[self.stocks[i]] * price.data)
                self.total_value = self.current_money + sum(self.stocks_value.values())
            else:
                price = self.get_price(self.stocks[i])
                cost = price * int(action)
                if self.current_money - cost < 0:
                    self.transactions.append(0)
                    reward -= 5
                else:
                    self.transactions.append(action)
                    self.stocks_owned[self.stocks[i]] += int(action)
                    self.stocks_value[self.stocks[i]] = float(self.stocks_owned[self.stocks[i]] * price.data)
                    self.total_value = self.current_money + sum(list(self.stocks_value.values()))
                    self.current_money -= cost
        returns = self.total_value - self.start_money
        reward += returns
        return reward, returns, prices_T, features, actions

    def compute_loss(self, reward, hidden):
        reward, returns, prices_T, features, actions = self.run(reward, hidden)
        

        Q_targets_next,_ = self.target_model([prices_T, features], hidden)
        Q_targets_next = Q_targets_next.detach().data.max(1, keepdim=True)[1]
        Q_targets = reward + (self.gamma * Q_targets_next)
        Q_expected, hidden = self.model([prices_T, features], hidden)
        hidden[0].detach_()
        hidden[1].detach_()
        Q_expected = Q_expected.gather(1, actions.cuda())
        loss = self.criterion(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update()
        return reward, returns, self.stocks_owned, hidden, self.current_money, self.total_value, self.stocks_value, self.transactions

    def soft_update(self):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)


class ReplayBuffer():
    def __init__(self, replay_size, batch_size, seed):
        self.memory = deque(maxlen=replay_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state"])
        self.seed = random.seed(seed)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).cuda()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).cuda()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).cuda()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).cuda()

        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.memory)
