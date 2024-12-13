import torch.nn as nn

from model.transformer.rms_norm import RMSNorm
from model.utils import compile_decorator


class FFN(nn.Module):
    def __init__(self, length, dim, output_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim*length, 128),
            nn.Flatten(-2,-1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    # @compile_decorator
    def forward(self, x):
        x = self.ffn(x)
        return x


class SwishGLU(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.input_mapping = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU()
        )
        self.gate = nn.Linear(embed_dim, hidden_dim)
        self.output_mapping = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = RMSNorm(embed_dim, elementwise_affine=True, eps=1e-5)

    def forward(self, x):
        x = x.transpose(1, 2)
        short_cut = x

        hidden = self.input_mapping(x)
        gate_matrix = self.gate(x)

        x = hidden*gate_matrix
        x = self.output_mapping(x)

        x = x+short_cut
        x = self.norm(x)
        return x.transpose(1, 2)