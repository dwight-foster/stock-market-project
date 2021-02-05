import torch.nn as nn
import torch
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_sizes, hidden_size, num_layers, dropout, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_sizes[0], hidden_size, num_layers=num_layers, dropout=dropout)
        self.fc_in = nn.Sequential(nn.Linear(input_sizes[1], 32), nn.Dropout(dropout), nn.ReLU(), nn.Linear(32, 256),
                                   nn.Dropout(dropout),
                                   nn.ReLU(), nn.Linear(256, 32), nn.ReLU())
        self.fc_out = nn.Sequential(nn.Linear(input_sizes[0] + hidden_size + 32, output_size))

    def forward(self, inputs, hidden):
        x, h, c = self.lstm(inputs[0], hidden)
        x2 = self.fc_in(inputs[1])
        x = x.view(x.shape[0], sum(x.shape[1:-1]))
        x = torch.cat([x, x2], dim=1)
        out = self.fc_out(x)
        out = F.tanh(out)
        return out, h

    def init_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda(),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda())
