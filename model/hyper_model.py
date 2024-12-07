import numpy as np
import pandas as pd
import torch.optim
from pandas import DataFrame
from pytorch_lightning import LightningModule
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_curve, auc
from torch.nn import functional as F

from .utils import focal_loss, calculate_auprc, calculate_auroc

class ClassHyperModel(LightningModule):
    def __init__(self):
        super(ClassHyperModel, self).__init__()

    def training_step(self, batch, batch_idx):
        x, target, dnase = batch
        output = self(x, dnase, target, mode='train')
        loss = self.calculate_loss(target, **output)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss}

    def on_validation_start(self) -> None:
        self.outputs = []
        self.targets = []

    def validation_step(self, batch, batch_idx):
        x, target, dnase = batch
        output_dict = self(x, dnase, target, mode='eval')
        self.outputs.append(output_dict['output'].detach().cpu().to(torch.float32).numpy())
        self.targets.append(target.detach().cpu().to(torch.float32).numpy())
        loss = self.calculate_loss(target, **output_dict)

        self.log('val_loss', loss, on_step=True, sync_dist=True)

        return {'val_loss': loss}

    def on_validation_epoch_end(self) -> None:
        self.outputs = np.concatenate(self.outputs, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)

        auprcs, mean_auprcs, std_auprcs = calculate_auprc(self.targets, self.outputs)

        DataFrame(auprcs).to_csv('./valid_result.csv')

        self.log('val_mean_auprc', mean_auprcs, prog_bar=True, on_epoch=True)
        self.log('val_std_auprc', std_auprcs, on_epoch=True)
        if self.best_val_auprc < mean_auprcs:
            self.best_val_auprc = mean_auprcs

    def test_step(self, batch, batch_idx):
        data, target, dnase = batch
        output_dict = self(data, dnase, target, mode='eval')
        # loss = self.calculate_loss(target, **output_dict)
        # self.log('test_loss', loss, on_step=True, prog_bar=True, sync_dist=True)
        self.outputs.append(output_dict['output'].detach().cpu().to(torch.float32).numpy())
        self.targets.append(target.detach().cpu().to(torch.float32).numpy())

    def on_test_start(self) -> None:
        self.outputs = []
        self.targets = []

    def on_test_epoch_end(self) -> None:
        self.outputs = np.concatenate(self.outputs, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)

        auprcs, mean_auprcs, std_auprcs = calculate_auprc(self.targets, self.outputs)
        print(f'Test AUPRC: {mean_auprcs:.2f}, {std_auprcs:.2f}')
        aurocs, mean_aurocs, std_aurocs = calculate_auroc(self.targets, self.outputs)
        print(f'Test AUROC: {mean_aurocs:.2f}, {std_aurocs:.2f}')
        DataFrame((auprcs, aurocs)).to_csv('./test_result.csv')

        self.log('auprc', mean_auprcs)
        self.log('auroc', mean_aurocs)
        self.log('best_val_auprc', self.best_val_auprc)
        # print(f'Test Precision: {precision:.2f}')
        # print(f'Test Recall: {recall:.2f}')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

class RegHyperModel(LightningModule):
    def __init__(self):
        super(RegHyperModel, self).__init__()
        # self.criterion=WeightedMSELoss([1,2,2,2])

    def training_step(self, batch, batch_idx):
        x, target = batch
        output = self(x)
        loss = self.calculate_loss(target, mode='train', **output)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss}

    def on_validation_start(self) -> None:
        self.outputs = []
        self.targets = []

    def validation_step(self, batch, batch_idx):
        x, target = batch
        output_dict = self(x)
        self.outputs.append(output_dict['output'].flatten(-2,-1).detach().cpu().to(torch.float32).numpy())
        self.targets.append(target.flatten(-2,-1).detach().cpu().to(torch.float32).numpy())
        loss = self.calculate_loss(target, mode='val', **output_dict)

        self.log('val_loss', loss, on_step=True, sync_dist=True)

        return {'val_loss': loss}

    def on_validation_epoch_end(self) -> None:
        self.outputs = np.concatenate(self.outputs, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)

        pr, _ = pearsonr(self.targets, self.outputs)

        # DataFrame(r).to_csv('./valid_result.csv')

        self.log('val_pr', pr, prog_bar=True, on_epoch=True)

        if self.best_val_pr < pr:
            self.best_val_pr = pr

    def test_step(self, batch, batch_idx):
        data, target = batch
        output_dict = self(data)
        loss = self.calculate_loss(target, mode='test', **output_dict)
        self.log('test_loss', loss, on_step=True, prog_bar=True, sync_dist=True)
        self.outputs.append(output_dict['output'].flatten(-2,-1).detach().cpu().to(torch.float32).numpy())
        self.targets.append(target.flatten(-2,-1).detach().cpu().to(torch.float32).numpy())

    def on_test_start(self) -> None:
        self.outputs = []
        self.targets = []

    def on_test_epoch_end(self) -> None:
        self.outputs = np.concatenate(self.outputs, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)

        pr, p = pearsonr(self.targets, self.outputs)

        self.log('peason_r', pr)
        self.log('best_val_pr', self.best_val_pr)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer