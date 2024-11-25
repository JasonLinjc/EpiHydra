from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.cnn import EPCOTConvLayer, EPCOTEncoder
from model.hyper_model import RegHyperModel, ClassHyperModel
from model.transformer.diff_attention import MultiheadFlashDiff2
from model.transformer.rms_norm import RMSNorm
from model.utils import seq2onehot


class DiffTransformerEncoder(nn.Module):
    def __init__(self, args, norm=None):
        super().__init__()
        encoder = []
        for i in range(args.enc_layers):
            encoder.append(DiffTransformerLayer(hidden_dim=args.hidden_dim, depth=i, max_seq_len=args.max_seq_len, num_heads=args.num_heads, dim_feedforward=args.dim_feedforward))


        # self.layers = nn.ModuleList(encoder)
        self.layers = nn.Sequential(*encoder)
        self.enc_layers = args.enc_layers
        self.norm = norm

    def forward(self, x):
        # for layer in self.layers:
        #     x = layer(x)
        x = self.layers(x)

        if self.norm is not None:
            x = self.norm(x)

        return x

class DiffTransformerReg(RegHyperModel):
    def __init__(self, args):
        super().__init__()
        self.lr = args.lr
        self.best_val_pr = 0
        self.weight_decay = args.weight_decay

        self.input_embed = nn.Embedding(num_embeddings=5, embedding_dim=args.hidden_dim)
        # self.input_embed = nn.Conv1d(4,512,10)

        self.backbone = DiffTransformerEncoder(args)

        self.classifier = nn.Sequential(
            nn.Linear(args.max_seq_len,1),
            nn.Flatten(-2,-1),
            nn.LayerNorm(args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 3),
        )

    def forward(self, x):
        # x = seq2onehot(x)
        x = self.input_embed(x-7)
        x = self.backbone(x)
        x = self.classifier(x.transpose(1,2))
        return {'output': x}

    def calculate_loss(self, target, output, mask=None):
        if mask is not None:
            target = target[mask]
            output = output[mask]

        loss = F.mse_loss(output, target)
        return loss

class CNNDiffTransformerReg(RegHyperModel):
    def __init__(self, args):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            EPCOTConvLayer(4, 256, kernel_size=10),
            nn.Dropout(p=0.1),
            EPCOTConvLayer(256, 256, kernel_size=10),
        )
        self.conv_block2 = nn.Sequential(
            EPCOTConvLayer(256, 360, kernel_size=8),
            nn.Dropout(p=0.1),
            EPCOTConvLayer(360, 360, kernel_size=8),
        )

        self.diff_t = nn.Sequential(
            MultiheadFlashDiff2(embed_dim=512, depth=0, max_seq_len=10, num_heads=4),
            MultiheadFlashDiff2(embed_dim=512, depth=1, max_seq_len=10, num_heads=4),
            MultiheadFlashDiff2(embed_dim=512, depth=1, max_seq_len=10, num_heads=4),
            MultiheadFlashDiff2(embed_dim=512, depth=1, max_seq_len=10, num_heads=4),
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
        x = seq2onehot(x)
        x = self.conv_block1(x)
        x = self.pooling_layer1(x)
        x = self.conv_block2(x)
        x = self.pooling_layer2(x)
        x = self.diff_t(x.transpose(1,2))

        output = self.classifier(x.transpose(1,2))
        return {'output': output}

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

class DiffTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, max_seq_len, depth, dim_feedforward):
        super().__init__()
        self.attn_layer = MultiheadFlashDiff2(embed_dim=hidden_dim, depth=depth, max_seq_len=max_seq_len, num_heads=num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, hidden_dim)
        )
        self.rms_norm = RMSNorm(hidden_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x):
        x = self.attn_layer(x)
        short_cut = x
        x = self.ffn(x)
        x = x+short_cut
        x = self.rms_norm(x)
        return x
