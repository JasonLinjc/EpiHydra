import torch
from pytorch_lightning import LightningModule
from model.transformer.attention import MultiheadFlashAttention
import torch.nn as nn

from model.transformer.transformer import FlashTransformerLayer
from model.utils import seq2onehot


class EntropyModel(LightningModule):
    def __init__(self, hidden_dim, num_heads, max_seq_len, output_channel=5):
        super().__init__()
        self.lr = 0.0001
        self.weight_decay = 1e-6
        self.best_val_loss = 100

        self.input_embed = nn.Embedding(output_channel, hidden_dim)

        self.criterion = nn.CrossEntropyLoss()
        self.backbone = nn.Sequential(
            FlashTransformerLayer(hidden_dim=hidden_dim, num_heads=num_heads, max_seq_len=max_seq_len, causal=True,
                                  dropout=0.1),
            FlashTransformerLayer(hidden_dim=hidden_dim, num_heads=num_heads, max_seq_len=max_seq_len, causal=True,
                                  dropout=0.1),
            FlashTransformerLayer(hidden_dim=hidden_dim, num_heads=num_heads, max_seq_len=max_seq_len, causal=True,
                                  dropout=0.1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.LayerNorm([32]),
            nn.Linear(32, output_channel),
        )# Batch, Length-1, 4

    def forward(self, x):
        x = self.input_embed(x).transpose(1,2)
        x = self.backbone(x)
        output = self.classifier(x.transpose(1,2))
        return output

    def calculate_loss(self, seq):
        label = seq[..., 1:]
        pred = self.forward(seq[..., :-1])
        loss = self.criterion(pred.flatten(0,1), label.flatten(0,1))
        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = self.calculate_loss(x)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss}

    def on_validation_start(self) -> None:
        self.epoch_loss = []

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        loss = self.calculate_loss(x)
        self.epoch_loss.append(loss)

        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        self.epoch_loss = torch.Tensor(self.epoch_loss)
        self.epoch_loss = self.epoch_loss.mean()

        if self.epoch_loss<self.best_val_loss:
            self.best_val_loss = self.epoch_loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        loss = self.calculate_loss(x)
        # loss = self.calculate_loss(target, **output_dict)
        self.log('test_loss', loss, on_step=True, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        self.log('best_val_loss', self.best_val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer