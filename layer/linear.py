from torch import nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_proj = F.dropout(F.relu(self.linear1(x)), p=self.dropout_rate)
        x_proj = self.linear2(x_proj)
        return x_proj


