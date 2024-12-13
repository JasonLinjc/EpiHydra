import torch
import yaml
from torch import nn
import torch.nn.functional as F

from .ffn import FFN
from .hyper_model import ClassHyperModel, RegHyperModel
from .utils import seq2onehot, focal_loss, compile_decorator, disable_compile_decorator
from model.stripedhyena import StripedHyena
from .stripedhyena.utils import dotdict


class HyenaBackboneClass(ClassHyperModel):
    def __init__(self, args, config_path="./configs/sh-stem-test.yml"):
        super().__init__()
        config = dotdict(yaml.load(open(config_path), Loader=yaml.FullLoader))
        self.backbone = StripedHyena(config).to(torch.float32).cuda()

        self.classifier = nn.Sequential(
            nn.Linear(5, 1),
            nn.Flatten(-2,-1),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, args.num_class),
            nn.Sigmoid()
        )

        self.dnase = args.dnase
        if not args.dnase:
            self.backbone.conv_net[0] = nn.Conv1d(4, 256, kernel_size=10)

        self.num_class = args.num_class
        self.lr = args.lr
        self.loss_type = args.loss_type
        self.best_val_auprc = 0
        self.weight_decay = args.weight_decay

    def forward(self, x, dnase, target=None, mode='train'):
        x = self.backbone(x-7)

        output = self.classifier(x[0][:, 300:1300, :])
        return {'mode': mode, 'output': output}

    def calculate_loss(self, target, mode, output, mask=None):
        if mask is not None:
            target = target[mask]
            output = output[mask]

        loss = []
        if 'focal' in self.loss_type:
            focal = focal_loss(output, target)
            self.log(mode + '_focal_loss', focal, on_step=True, on_epoch=True, sync_dist=True)
            loss.append(focal)
        if 'bce' in self.loss_type:
            bce = F.binary_cross_entropy(output, target)
            self.log(mode + '_bce_loss', bce, on_step=True, on_epoch=True, sync_dist=True)
            loss.append(bce)

        return loss[0]


class HyenaBackboneReg(RegHyperModel):
    def __init__(self, args, config_path="./configs/sh-stem-test.yml"):
        super().__init__()
        config = dotdict(yaml.load(open(config_path), Loader=yaml.FullLoader))
        self.backbone = StripedHyena(config)

        self.regressor = nn.Sequential(
            nn.Linear(200,1),
            nn.Flatten(-2, -1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256,3)
        )

        self.dnase = args.dnase
        # if not args.dnase:
        #     self.backbone.conv_net[0] = nn.Conv1d(4, 256, kernel_size=10)

        self.num_class = args.num_class
        self.lr = args.lr
        self.loss_type = args.loss_type
        self.best_val_auprc = 0
        self.weight_decay = args.weight_decay

        self.best_val_pr = 0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    # @disable_compile_decorator
    # @torch.compile(fullgraph=False, dynamic=False, mode='max-autotune')
    def forward(self, x):
        x = seq2onehot(x)
        x = self.backbone(x)[0]

        output = self.regressor(x.transpose(1, 2))
        # output = x
        return {'output': output}

    # @compile_decorator
    def calculate_loss(self, target, mode, output):
        # if mask is not None:
        #     target = target[mask]
        #     output = output[mask]
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
