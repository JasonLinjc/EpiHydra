from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.cnn import EPCOTConvLayer, EPCOTEncoder
from model.ffn import SwishGLU
from model.hyper_model import RegHyperModel, ClassHyperModel
from model.transformer.diff_attention import MultiheadFlashDiff2, DiffAttnLayer
from model.transformer.rms_norm import RMSNorm
from model.utils import seq2onehot, compile_decorator, AutomaticWeightedLoss


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

        # self.input_embed = nn.Embedding(num_embeddings=5, embedding_dim=args.hidden_dim)
        self.input_embed = nn.Conv1d(4,512,10)

        self.backbone = DiffTransformerEncoder(args)

        self.classifier = nn.Sequential(
            nn.Linear(args.max_seq_len-9,1),
            nn.Flatten(-2,-1),
            nn.BatchNorm1d(args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 3),
        )

    # @compile_decorator
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

class CNNDiffTransformerReg(RegHyperModel):
    def __init__(self, args):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            EPCOTConvLayer(4, 256, kernel_size=10),
            nn.Dropout(p=args.dropout),
            EPCOTConvLayer(256, 256, kernel_size=10),
            # DiffAttnLayer(256, 1, 200, 4)
            # MultiheadFlashDiff2(embed_dim=256, depth=0, max_seq_len=200, num_heads=4),
        )
        self.conv_block2 = nn.Sequential(
            EPCOTConvLayer(256, 360, kernel_size=8),
            nn.Dropout(p=args.dropout),
            # DiffAttnLayer(360, 3, 40, 4)
            EPCOTConvLayer(360, 512, kernel_size=8),
        )
        # self.conv_block3 = nn.Sequential(
        #     EPCOTConvLayer(360, 512, kernel_size=8),
        #     nn.Dropout(p=args.dropout),
        #     DiffAttnLayer(512, 5, 10, 4)
        #
        #     # EPCOTConvLayer(360, 512, kernel_size=8),
        # )
        # self.conv_block4 = nn.Sequential(
        #     # nn.BatchNorm1d(512),
        #     EPCOTConvLayer(512, 512, kernel_size=8),
        #     # nn.BatchNorm1d(512),
        #     nn.Dropout(p=args.dropout),
        #     DiffAttnLayer(512, 7, 10, 4)
        #     # MultiheadFlashDiff2(embed_dim=512, depth=3, max_seq_len=10, num_heads=4),
        #     # nn.BatchNorm1d(512),
        #     # EPCOTConvLayer(360, 512, kernel_size=8),
        # )

        self.diff_t = nn.Sequential(
            DiffAttnLayer(512, 4, 10, 4),
            DiffAttnLayer(512, 5, 10, 4),
            # DiffTransformerLayer(512, 4, 10, 4, args.dropout),
            # DiffTransformerLayer(512, 4, 10, 5, args.dropout)
        )

        self.pooling_layer1 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.BatchNorm1d(256),
            nn.Dropout(p=args.dropout),
        )
        self.pooling_layer2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.BatchNorm1d(512),
            nn.Dropout(p=args.dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(10, 1),
            nn.Flatten(-2,-1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 3),
        )

        self.lr = args.lr
        self.best_val_pr= 0
        self.weight_decay = args.weight_decay
        self.loss_type = args.loss_type

        if "+" in args.loss_type:
            num = len(args.loss_type.split('+'))
            self.loss_balancer = AutomaticWeightedLoss(num=num)

        if args.compile:
            self.conv_block1 = torch.compile(self.conv_block1, fullgraph=True, dynamic=False, mode='max-autotune')
            self.conv_block2 = torch.compile(self.conv_block2, fullgraph=True, dynamic=False, mode='max-autotune')

            self.classifier = torch.compile(self.classifier, fullgraph=True, dynamic=False, mode='max-autotune')

            # self.diff_t = torch.compile(self.diff_t, fullgraph=False, dynamic=False, mode='max-autotune')
            # self.forward = torch.compile(self.forward, fullgraph=False, dynamic=False, mode='max-autotune')

    def forward(self, x):
        x = seq2onehot(x)
        x = self.conv_block1(x)
        x = self.pooling_layer1(x)
        x = self.conv_block2(x)
        x = self.pooling_layer2(x)
        # x = self.conv_block3(x)
        # x = self.conv_block4(x)
        x = self.diff_t(x)

        output = self.classifier(x)
        return {'output': output}

    def calculate_loss(self, target, output, mask=None, mode='train'):
        if mask is not None:
            target = target[mask]
            output = output[mask]
        loss = []
        if 'mse' in self.loss_type:
            mse = F.mse_loss(output, target)
            if mode!='test':
                self.log(mode+'_mse', mse, on_step=True, on_epoch=True)
            loss.append(mse)
        if 'l1' in self.loss_type:
            l1 = F.l1_loss(output, target)
            if mode!='test':
                self.log(mode+'_l1', l1, on_step=True, on_epoch=True)
            loss.append(l1)
        if 'kl' in self.loss_type:
            kl = F.kl_div(F.log_softmax(output), F.log_softmax(target))
            if mode != 'test':
                self.log(mode + '_kl', kl, on_step=True, on_epoch=True)
            loss.append(kl)

        if len(loss)>1:
            loss = self.loss_balancer(loss)
            return loss
        else:
            return loss[0]

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
            nn.BatchNorm1d(args.hidden_dim),
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
    def __init__(self, hidden_dim, num_heads, max_seq_len, depth, dropout):
        super().__init__()
        self.attn_layer = DiffAttnLayer(hidden_dim=hidden_dim, depth=depth, max_seq_len=max_seq_len, num_heads=num_heads, dropout=dropout)

        self.ffn = SwishGLU(hidden_dim, 2 * hidden_dim, dropout=dropout)
        # self.rms_norm1 = RMSNorm(hidden_dim, eps=1e-5, elementwise_affine=True)
        # self.rms_norm2 = RMSNorm(hidden_dim, eps=1e-5, elementwise_affine=True)
        # self.rms_norm2 = nn.LayerNorm([max_seq_len-9, hidden_dim])

    def forward(self, x):
        x = self.attn_layer(x)

        # x = self.rms_norm1(x)

        x = self.ffn(x)

        # x = self.rms_norm2(x)
        return x
