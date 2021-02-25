import torch
from torch import nn


class Scale(nn.Module):
    def __init__(self, init_value=1.0, dtype=torch.float32):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=dtype))

    def forward(self, x):
        return x * self.scale
