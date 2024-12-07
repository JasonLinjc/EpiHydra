import torch.nn as nn

from model.utils import compile_decorator


class FFN(nn.Module):
    def __init__(self, length, dim, output_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Flatten(-2,-1),
            nn.BatchNorm1d(length),
            nn.ReLU(),
            nn.Linear(length, output_dim)
        )

    # @compile_decorator
    def forward(self, x):
        x = self.ffn(x)
        return x
