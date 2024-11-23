from typing import Any

import numpy as np
import torch
from torch import nn

from .hyper_model import RegHyperModel, ClassHyperModel
import torch.nn.functional as F

from .utils import seq2onehot, focal_loss


class ExperimentArgs():
    def __init__(self, lr=0.0001, loss_type='focal', from_ckpt=False, num_class=245, enc_layers=1, dec_layers=2,
                 hidden_dim=512, dropout=0.2, mask_label=False, dim_feedforward=1024, weight_decay=1e-6,
                 dnase=True, load_backbone=False, freeze_backbone=False, max_seq_len = 1600, rope_theta = 10000, num_heads=4):
        self.lr = lr
        self.loss_type = loss_type
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.mask_label = mask_label
        self.from_ckpt = from_ckpt
        self.weight_decay = weight_decay
        self.load_backbone = load_backbone
        self.dnase = dnase
        self.freeze_backbone = freeze_backbone
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len


class EPCOTBackboneClass(ClassHyperModel):
    def __init__(self, args):
        super().__init__()
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.loss_type = args.loss_type
        self.best_val_auprc = 0

        if args.dnase:
            in_dim = 5
        else:
            in_dim=4

        self.dnase = args.dnase

        self.backbone = EPCOTEncoder(in_dim)

        self.classifier = nn.Sequential(
            nn.Linear(80,1),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(-2,-1),
            nn.LayerNorm([512]),
            nn.Linear(512, args.num_class),

        )

    def forward(self, x, dnase, target=None, mode='train'):
        x = seq2onehot(x)
        if self.dnase:
            x = torch.concat((x, dnase.unsqueeze(1)), dim=1)
        x = self.backbone(x, dnase)
        x = self.classifier(x)

        return {'mode': mode, 'output': x}

    def calculate_loss(self, target, mode, output, mask=None):
        if mask is not None:
            target = target[mask]
            output = output[mask]

        output = output.float()

        bce = F.binary_cross_entropy_with_logits(output, target)
        self.log(mode + '_bce_loss', bce, on_step=True, on_epoch=True, sync_dist=True)

        return bce

class EPCOTBackboneReg(RegHyperModel):
    def __init__(self, args):
        super().__init__()
        self.backbone = EPCOTEncoder(args)
        self.classifier = nn.Sequential(
            nn.Linear(10, 1),
            nn.Flatten(-2,-1),
            nn.ReLU(),
            nn.LayerNorm([512]),
            nn.Linear(512, 3),
        )

        self.lr = args.lr
        self.best_val_pr= 0
        self.weight_decay = args.weight_decay

    def forward(self, x):
        x = self.backbone(x)

        output = self.classifier(x)
        return {'output': output}

    def calculate_loss(self, target, output, mask=None):
        if mask is not None:
            target = target[mask]
            output = output[mask]

        loss = F.mse_loss(output, target)
        return loss
class EPCOTConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding='same')
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x

class EPCOTEncoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            EPCOTConvLayer(in_dim, 256, kernel_size=10),
            nn.Dropout(p=0.1),
            EPCOTConvLayer(256, 256, kernel_size=10),
        )
        self.conv_block2 = nn.Sequential(
            EPCOTConvLayer(256, 360, kernel_size=8),
            nn.Dropout(p=0.1),
            EPCOTConvLayer(360, 360, kernel_size=8),
        )
        self.conv_block3 = nn.Sequential(
            EPCOTConvLayer(360, 512, kernel_size=8),
            nn.Dropout(p=0.2),
            EPCOTConvLayer(512, 512, kernel_size=8),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
        )

        self.pooling_layer1 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.1),
        )
        self.pooling_layer2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.BatchNorm1d(360),
            nn.Dropout(p=0.1),
        )

    def forward(self, x):

        x = self.conv_block1(x)
        x = self.pooling_layer1(x)
        x = self.conv_block2(x)
        x = self.pooling_layer2(x)
        x = self.conv_block3(x)
        return x