from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.cnn import EPCOTConvLayer
from model.hyper_model import RegHyperModel, ClassHyperModel
from model.transformer.diff_attention import MultiheadFlashDiff2
from model.utils import seq2onehot


class DiffTransformerEncoder(nn.Module):
    def __init__(self, max_seq_len, hidden_dim, num_layers, num_heads, norm=None):
        super().__init__()
        encoder = []
        for i in range(num_layers):
            encoder.append(MultiheadFlashDiff2(embed_dim=hidden_dim, depth=i, max_seq_len=max_seq_len, num_heads=num_heads))

        self.layers = nn.ModuleList(encoder)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x

class DiffTransformerReg(RegHyperModel):
    def __init__(self, args):
        super().__init__()
        self.lr = args.lr
        self.best_val_pr = 0
        self.weight_decay = args.weight_decay

        # self.input_embed = nn.Embedding(num_embeddings=5, embedding_dim=args.hidden_dim)
        self.input_embed = nn.Conv1d(4,512,10)

        self.backbone = DiffTransformerEncoder(args.max_seq_len, args.hidden_dim, args.enc_layers, args.num_heads)

        self.classifier = nn.Sequential(
            nn.Linear(args.max_seq_len-9,1),
            nn.Flatten(-2,-1),
            nn.LayerNorm(args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 3),
        )

    def forward(self, x):
        x = seq2onehot(x)
        x = self.input_embed(x)
        x = self.backbone(x.transpose(1,2))
        x = self.classifier(x.transpose(1,2))
        return {'output': x}

    def calculate_loss(self, target, output, mask=None):
        if mask is not None:
            target = target[mask]
            output = output[mask]

        loss = F.mse_loss(output, target)
        return loss

class DiffTransformerClass(ClassHyperModel):
    def __init__(self, args):
        super().__init__()
        self.lr = args.lr
        self.best_val_auprc = 0
        self.weight_decay = args.weight_decay
        self.dnase = args.dnase

        if args.dnase:
            input_dim = 5
        else:
            input_dim = 4

        self.input_cnn = nn.Sequential(
            EPCOTConvLayer(input_dim, args.hidden_dim, kernel_size=10),
            nn.LayerNorm([args.hidden_dim, args.max_seq_len]),
        )

        self.backbone = DiffTransformerEncoder(args.max_seq_len, args.hidden_dim, args.enc_layers)

        self.classifier = nn.Sequential(
            nn.Linear(args.max_seq_len,1),
            nn.Flatten(-2,-1),
            nn.LayerNorm(args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.num_class),
        )

    def forward(self, x, dnase, target, mode='eval'):
        x = seq2onehot(x)
        if self.dnase:
            x = torch.concat((x,dnase.unsqueeze(1)),dim=1)
        x = self.input_cnn(x-7)
        x = self.backbone(x.transpose(1,2))
        x = self.classifier(x.transpose(1,2))
        return {'mode':mode, 'output': x}

    def calculate_loss(self, target, mode, output, mask=None):
        if mask is not None:
            target = target[mask]
            output = output[mask]

        output = output.float()

        bce = F.binary_cross_entropy_with_logits(output, target)
        self.log(mode + '_bce_loss', bce, on_step=True, on_epoch=True, sync_dist=True)

        return bce