import numpy as np

from cnn import EPCOTEncoder, EPCOTBackboneClass
from hyper_model import ClassHyperModel
from model.transformer.transformer import Transformer
from utils import seq2onehot

import math
import torch
from torch import nn
import torch.nn.functional as F
# from apex.normalization import FusedRMSNorm as RMSNorm


class EPCOTModel(ClassHyperModel):
    def __init__(self, args):
        super().__init__()
        self.backbone = EPCOTEncoder(args)
        self.transformer = Transformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers
        )
        hidden_dim = self.transformer.d_model
        self.input_proj = nn.Conv1d(self.backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(args.num_class, hidden_dim)
        self.fc = GroupWiseLinear(args.num_class, hidden_dim, bias=True)

        if args.load_backbone:
            pretrain_model = EPCOTBackboneClass.load_from_checkpoint(args.load_backbone, args=args)
            self.backbone = pretrain_model.backbone
            if args.freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False

        self.dnase = args.dnase

        self.num_class = args.num_class
        self.lr = args.lr
        self.loss_type = args.loss_type
        self.best_val_auprc = 0
        self.weight_decay = args.weight_decay

        self.label_input = torch.Tensor(np.arange(args.num_class)).view(1, -1).long()


    def forward(self, x, dnase, target=None, mode='train'):
        x = seq2onehot(x)
        if self.dnase:
            x = torch.cat((x, dnase.unsqueeze(1)), dim=1)

        src = self.backbone(x)
        label_inputs = self.label_input.repeat(src.size(0), 1).cuda()
        label_embed = self.query_embed(label_inputs)

        src = self.input_proj(src)
        hs = self.transformer(src, label_embed)
        out = F.sigmoid(self.fc(hs))
        return {'mode': mode, 'output': out}

    def calculate_loss(self, target, mode, output, mask=None):
        if mask is not None:
            target = target[mask]
            output = output[mask]

        bce = F.binary_cross_entropy(output, target)
        self.log(mode + '_bce_loss', bce, on_step=True, on_epoch=True, sync_dist=True)

        return bce

class GroupWiseLinear(nn.Module):
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)
    def forward(self, x):
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


