import numpy as np
import torch
import torch.nn as nn


class GRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUNetwork, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape dovrebbe essere [batch_size, sequence_length, input_size]
        gru_out, _ = self.gru(x)
        # Prendi l'ultimo output del GRU
        out = self.fc(gru_out[:, -1, :])
        return out