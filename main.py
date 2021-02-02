import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import numpy as np
import pandas as pd

def get_data(csv, prices):
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