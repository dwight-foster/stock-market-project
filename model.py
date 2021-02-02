import torch.nn as nn
import torch


class LSTM(nn.Module):
    def __init__(self, input_sizes, hidden_size, num_layers, dropout, output_size):
        super(model, self).__init__()
        self.lstm_size = input_sizes[0]
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_sizes[0], hidden_size, num_layers, dropout)
        self.fc_in = nn.Sequential(nn.Linear(input_sizes[1], 32), nn.Dropout(dropout), nn.ReLU(), nn.Linear(32, 256),
                                   nn.Dropout(dropout),
                                   nn.ReLU(), nn.Linear(256, 32), nn.ReLU)
        self.fc_out = nn.Sequential(nn.Linear(input_sizes[0] + hidden_size + 32, output_size))

    def forward(self, inputs):
        x = self.lstm(inputs[0])
        x2 = self.fc_in(inputs[1])
        x = x.view(x.shape[0], sum(x.shape[1:-1]))
        x = torch.cat([x, x2], dim=1)
        out = self.fc_out(x)
        return out

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))
