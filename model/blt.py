import numpy as np
import torch
from pytorch_lightning import LightningModule

from bytelatent.model.blt import ByteLatentTransformer
from model.hyper_model import NTPHyperModel, RegressionHyperModel
from model.transformer.attention import MultiheadFlashAttention
import torch.nn as nn
import torch.nn.functional as F

from model.transformer.transformer import FlashTransformerLayer
from model.utils import seq2onehot


class BLT(NTPHyperModel):
    def __init__(self, args, lr=0.0001, weight_decay=1e-6):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_length = 200
        self.best_val_loss = 100
        self.eos = 128
        self.pad = 129
        # self.unk=2
        # self.alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-=~!%* ,./?;\':"\\|'
        # self.tokenizer = {chara: i + 3 for i, chara in enumerate(self.alphabet)}

        self.backbone = ByteLatentTransformer(args)
        self.criterion = nn.CrossEntropyLoss()

        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_dim, 32),
        #     nn.ReLU(),
        #     nn.LayerNorm([32]),
        #     nn.Linear(32, output_channels),
        # )# Batch, Length-1, 4

    def forward(self, data):
        seq, patch_lengths = data
        output = self.backbone(seq, patch_lengths)
        return output

    def calculate_loss(self, data, target):
        ## get target mask
        target_mask = target != 129

        ## forward
        pred = self.forward(data)

        ## mask output and target
        pred = pred[target_mask, :]
        target = target[target_mask]

        loss = self.criterion(pred, target)
        return loss

    def tokenize(self, text):
        tokens = []
        for char in text:
            if char in self.tokenizer:
                tokens.append(self.tokenizer[char])
            else:
                tokens.append(self.unk)
        return tokens

    def generate_text(self, prompt, k=1):
        assert len(prompt) < 200, 'Prompt is too long'
        ## tokenize prompt via ascii
        prompt = [ord(ch) for ch in prompt]
        # prompt = self.tokenize(prompt)
        prompt = torch.tensor(prompt, dtype=torch.int64)

        ## pad prompt
        prompt_length = len(prompt)
        if prompt_length < self.max_length:
            pad_length = self.max_length - prompt_length
            pad = torch.ones((pad_length,), dtype=torch.int64)*129
            prompt = torch.hstack((prompt, pad))

        prompt = prompt.to(self.device)

        ## inference
        while True:
            pred = self.forward((prompt.unsqueeze(0), None))[:, prompt_length-1, :].to(torch.int64)

            # get top k token
            _, top_k_token = torch.topk(pred, k, dim=-1)
            pred = top_k_token[:, torch.randint(0, k, (1,))]

            if pred == self.eos:
                break

            prompt[prompt_length] = pred.squeeze()
            prompt_length += 1

            if prompt_length >= self.max_length:
                break

        ## decode output
        prompt = prompt[prompt != self.pad]
        prompt = prompt[prompt != self.eos]
        prompt = prompt.squeeze().cpu().numpy()
        prompt = [chr(token) for token in prompt]
        # prompt = [self.alphabet[token-3] for token in prompt if token > 2]

        return ''.join(prompt)

class BLTDNALM(RegressionHyperModel):
    def __init__(self, args, lr=0.0001, weight_decay=1e-6, output_channels=3, loss_type='mse'):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.best_val_pr = 0
        self.loss_type = loss_type

        self.backbone = ByteLatentTransformer(args)
        self.regressor = nn.Sequential(
            nn.Flatten(-2,-1),
            nn.Linear(args.vocab_size*args.max_length, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_channels)
        )

    def forward(self, data):
        x, patch_lengths = data
        x = self.backbone(x, patch_lengths=patch_lengths)

        x = self.regressor(x)

        return {'output': x}

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