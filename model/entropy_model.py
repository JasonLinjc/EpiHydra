import torch
from pytorch_lightning import LightningModule

from model.hyper_model import NTPHyperModel
from model.transformer.attention import MultiheadFlashAttention
import torch.nn as nn

from model.transformer.transformer import FlashTransformerLayer
from model.utils import seq2onehot


class EntropyModel(NTPHyperModel):
    def __init__(self, hidden_dim, num_heads, max_length, output_channels=5):
        super().__init__()
        self.lr = 0.0001
        self.weight_decay = 1e-6
        self.best_val_loss = 100
        self.max_length = max_length

        self.input_embed = nn.Embedding(output_channels, hidden_dim)

        self.criterion = nn.CrossEntropyLoss()
        self.backbone = nn.Sequential(
            FlashTransformerLayer(hidden_dim=hidden_dim, num_heads=num_heads, max_seq_len=max_length, causal=True,
                                  dropout=0.1),
            FlashTransformerLayer(hidden_dim=hidden_dim, num_heads=num_heads, max_seq_len=max_length, causal=True,
                                  dropout=0.1),
            FlashTransformerLayer(hidden_dim=hidden_dim, num_heads=num_heads, max_seq_len=max_length, causal=True,
                                  dropout=0.1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.LayerNorm([32]),
            nn.Linear(32, output_channels),
        )# Batch, Length-1, 4

    def forward(self, x):
        # x = x[:, :-1]
        x = self.input_embed(x).transpose(1, 2)
        x = self.backbone(x)
        output = self.classifier(x.transpose(1, 2))
        return output

    def calculate_loss(self, data):
        ## get target mask
        # target_mask = target!=129
        # bs = seq.shape(0)
        # seq = torch.ones_like(data)*self.sos
        seq = data[:, :-1]
        # target = bs.append()
        # seq = seq[:, :-1]
        ## forward
        pred = self.forward(seq)
        target = data[:, 1:]
        ## mask output and target
        # pred = pred[target_mask, :]
        # target = target[target_mask]

        loss = self.criterion(pred.flatten(0, 1), target.flatten(0, 1))
        return loss

