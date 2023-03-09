import numpy as np
import torch
import torch.nn as nn
import matplotlib
import math
from scipy.special import comb
import scipy


class NeuralNetworkPolicy(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    model = NeuralNetworkPolicy(1, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for e in range(10):
        loss = 0.
        wealth = 1.

        portions = torch.tensor([0.5, 0.5, 0.5, 0.5])
        for i in range(73):
            arr = torch.tensor([1., 1., 1., 1.])
            wealth *= torch.dot(portions, arr)
            portions = model(torch.tensor([wealth]))
            loss -= wealth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('loss:', loss)




