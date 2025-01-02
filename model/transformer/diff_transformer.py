from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import Tensor

from bytelatent.data.patcher import Patcher, PatcherArgs
from bytelatent.model.blt import get_blt_input, init_embeddings, compute_hash_embeddings, patch_ids_from_lengths, \
    EmbeddingType, ByteLatentTransformerArgs
from bytelatent.model.local_models import LocalEncoder
from model.cnn import EPCOTConvLayer, EPCOTEncoder, EPCOTConvBlock
from model.ffn import SwishGLU
from model.hyper_model import RegHyperModel, ClassHyperModel
from model.transformer.attention import MultiheadFlashAttention
from model.transformer.diff_attention import MultiheadFlashDiff2, DiffAttnLayer
from model.transformer.rms_norm import RMSNorm
from model.transformer.blt import BLTLocalEncoder
from model.utils import seq2onehot, compile_decorator, AutomaticWeightedLoss


class DiffTransformerEncoder(nn.Module):
    def __init__(self, args, norm=None):
        super().__init__()
        encoder = []
        for i in range(args.enc_layers):
            encoder.append(DiffTransformerLayer(hidden_dim=args.hidden_dim, depth=i,
                                                dropout=args.dropout,
                                                max_seq_len=args.max_seq_len,
                                                num_heads=args.num_heads))


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
        self.loss_type = args.loss_type
        self.local_encoder = BLTLocalEncoder()

        # self.input_embed = nn.Conv1d(4,args.hidden_dim,10, padding='same')

        self.backbone = DiffTransformerEncoder(args)

        self.classifier = nn.Sequential(
            nn.AvgPool1d(kernel_size=4, stride=4),
            nn.Flatten(-2, -1),
            nn.Linear(50*args.hidden_dim,128),
            nn.LayerNorm([128]),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    # @compile_decorator
    def forward(self, x):
        x, patch_lengths = x
        x = self.local_encoder(x-7, patch_lengths).transpose(1,2)
        # x = seq2onehot(x)
        # x = self.input_embed(x)
        # x = x.transpose(1,2)
        # x = F.avg_pool1d(x, kernel_size=2, stride=2)
        x = self.backbone(x)
        x = self.classifier(x)
        return {'output': x}

    # def local_encoder_forward(self, local_encoder_tokens, patch_lengths):
    #     local_encoder_embeds = compute_hash_embeddings(
    #         local_encoder_tokens=local_encoder_tokens,
    #         local_encoder=self.local_encoder,
    #         encoder_hash_tok_embedding=self.encoder_hash_tok_embedding,
    #         encoder_hash_byte_group_nb_functions=self.encoder_hash_byte_group_nb_functions,
    #         encoder_hash_byte_group_size=self.encoder_hash_byte_group_size,
    #         encoder_hash_byte_group_vocab=self.encoder_hash_byte_group_vocab,
    #     )
    #     # patch_lengths, _ = self.patcher.patch(local_encoder_tokens)
    #     # patch_start_ids = torch.Tensor([0,1,2,3,4,5,6])
    #     # seq_len=7
    #     # patch_lengths = patch_lengths_from_start_ids(patch_start_ids, seq_len)
    #     patch_ids = patch_ids_from_lengths(
    #         patch_lengths, local_encoder_tokens.shape[-1]
    #     )
    #     (h_encoder, h_cross), cache_encoder = self.local_encoder(
    #         tokens=local_encoder_tokens,
    #         embeds=local_encoder_embeds,
    #         patch_embeds=None,
    #         cross_mask=None,
    #         num_patches=patch_lengths.shape[1],
    #         patch_ids=patch_ids,
    #     )
    #     return h_encoder

    def calculate_loss(self, target, output, mask=None, mode='train'):
        if mask is not None:
            target = target[mask]
            output = output[mask]
        loss = []
        if 'mse' in self.loss_type:
            mse = F.mse_loss(output, target)
            if mode != 'test':
                self.log(mode + '_mse', mse, on_step=True, on_epoch=True)
            loss.append(mse)
        if 'l1' in self.loss_type:
            l1 = F.l1_loss(output, target)
            if mode != 'test':
                self.log(mode + '_l1', l1, on_step=True, on_epoch=True)
            loss.append(l1)
        if 'kl' in self.loss_type:
            kl = F.kl_div(F.log_softmax(output), F.log_softmax(target))
            if mode != 'test':
                self.log(mode + '_kl', kl, on_step=True, on_epoch=True)
            loss.append(kl)

        if len(loss) > 1:
            loss = self.loss_balancer(loss)
            return loss
        else:
            return loss[0]


class CNNDiffTransformerReg(RegHyperModel):
    def __init__(self, args):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            EPCOTConvBlock(4,256, kernel_size=10, dropout=args.dropout, length=200),
            # EPCOTConvBlock(128, 256, kernel_size=10, dropout=args.dropout, length=200),
            # DiffAttnLayer(256, 1, 200, 4, dropout=args.dropout),
            # DiffAttnLayer(256, 2, 200, 4, dropout=args.dropout),
        )
        # self.diff_attn1 = DiffAttnLayer(256, 4, 200, 4, dropout=args.dropout)
        self.diff_attn1 = DiffTransformerLayer(256, 4, 200, 3, args.dropout)
        self.conv_block2 = nn.Sequential(
            EPCOTConvBlock(256, 256, kernel_size=8, dropout=args.dropout, length=100),
            # nn.Dropout(p=args.dropout),
            # DiffAttnLayer(360, 2, 40, 4, dropout=args.dropout),
            # DiffAttnLayer(360, 3, 40, 4, dropout=args.dropout),
            # EPCOTConvBlock(256, 256, kernel_size=8, dropout=args.dropout, length=40),
        )
        # self.diff_attn2 = DiffAttnLayer(256, 9, 40, 4, dropout=args.dropout)
        self.diff_attn2 = DiffTransformerLayer(256, 4, 100, 7, args.dropout)

        self.conv_block3 = nn.Sequential(
            EPCOTConvBlock(256, 256, kernel_size=8, dropout=args.dropout, length=50),
            # DiffAttnLayer(512, 5, 10, 4)
            # EPCOTConvBlock(256, 256, kernel_size=8, dropout=args.dropout, length=10),
        )
        self.diff_attn3 = DiffTransformerLayer(256, 4, 50, 11, args.dropout)
        self.conv_block4 = EPCOTConvBlock(256, 256, kernel_size=8, dropout=args.dropout, length=25)

        # self.diff_t = nn.Sequential(
        #     DiffAttnLayer(512, 4, 10, 4, dropout=args.dropout),
        #     DiffAttnLayer(512, 5, 10, 4, dropout=args.dropout),
        #     # DiffTransformerLayer(512, 4, 10, 4, args.dropout),
        #     # DiffTransformerLayer(512, 4, 10, 5, args.dropout)
        # )

        self.pooling_layer1 = nn.Sequential(
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.LayerNorm([256, 100]),
            nn.Dropout(p=args.dropout),
        )
        self.pooling_layer2 = nn.Sequential(
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.LayerNorm([256,50]),
            nn.Dropout(p=args.dropout),
        )
        self.pooling_layer3 = nn.Sequential(
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.LayerNorm([256, 25]),
            nn.Dropout(p=args.dropout),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(-2, -1),
            nn.Linear(256*25, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 3),
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
            self.conv_block3 = torch.compile(self.conv_block3, fullgraph=True, dynamic=False, mode='max-autotune')
            self.classifier = torch.compile(self.classifier, fullgraph=True, dynamic=False, mode='max-autotune')

            # self.diff_t = torch.compile(self.diff_t, fullgraph=False, dynamic=False, mode='max-autotune')
            # self.forward = torch.compile(self.forward, fullgraph=False, dynamic=False, mode='max-autotune')

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = seq2onehot(x)
        x = self.conv_block1(x)
        x = self.diff_attn1(x)

        x = self.pooling_layer1(x)

        x = self.conv_block2(x)
        x = self.diff_attn2(x)
        x = self.pooling_layer2(x)


        x = self.conv_block3(x)
        x = self.diff_attn3(x)
        x = self.pooling_layer3(x)

        x = self.conv_block4(x)
        # x = self.diff_t(x)

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

        self.ffn = SwishGLU(hidden_dim, 3 * hidden_dim, dropout=dropout)
        # self.rms_norm1 = RMSNorm(hidden_dim, eps=1e-5, elementwise_affine=True)
        # self.rms_norm2 = RMSNorm(hidden_dim, eps=1e-5, elementwise_affine=True)
        # self.rms_norm2 = nn.LayerNorm([max_seq_len-9, hidden_dim])

    def forward(self, x):
        x = self.attn_layer(x)

        # x = self.rms_norm1(x)

        x = self.ffn(x)

        # x = self.rms_norm2(x)
        return x
