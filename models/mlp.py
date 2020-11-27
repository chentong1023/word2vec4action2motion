from typing import List

import torch
import torch.nn as nn

__all__ = ["MLP"]


class MLP(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, hidden_size: List[int]
    ):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.layers = []
        for i, h in enumerate(hidden_size):
            if i == 0:
                self.layers.append(nn.Linear(input_size, h))
            else:
                self.layers.append(nn.Linear(hidden_size[i - 1], h))
        self.layers = nn.ModuleList(self.layers)
        self.output_layer = nn.Linear(hidden_size[-1], output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        x = self.output_layer(x)
        return x
