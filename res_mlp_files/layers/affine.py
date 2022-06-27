import torch
import torch.nn as nn


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((1, 1, dim)))
        self.beta = nn.Parameter(torch.zeros((1, 1, dim)))

    def forward(self, x):
        # return torch.addcmul(self.beta, self.alpha, x)
        return self.beta + self.alpha * x
