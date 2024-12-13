import torch
from pytorch_lightning.profilers import AdvancedProfiler
from torch import nn
import os
from torch.utils.data import DataLoader, Subset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

from model import utils
from model.cnn import ExperimentArgs, EPCOTBackboneReg
from model.striped_hyena import HyenaBackboneReg
from model.transformer.diff_transformer import DiffTransformerReg, CNNDiffTransformerReg

import glob

os.environ['WANDB_MODE']='offline'

cell='MPRA'
project_name=cell+'-EPCOT_DiffT'
# project_name='test'
dec_layers = 3
enc_layers = 12
hidden_dim = 256
loss_type = 'mse'
freeze_backbone = False
num_class = 12
dnase = False
max_seq_len = 200

num_heads=8

lr = 0.0001
weight_decay = 1e-6

# experiment_name = f'lr{lr}-alpha{alpha}-beta{beta}-factor-contra{positive_threshold}-d_bn{d_bottle_neck}-ch{d_encoder}-{loss_type}-{d_model}_{n_layer}'
# experiment_name = f'lr{lr}-{loss_type}-3striped-abs-dp0.1'
experiment_name = f'{loss_type}-12/256-diff_t_cnn_embed'
epochs = 50
if project_name == 'test':
    epochs = 1

batch_size=32
args = ExperimentArgs(loss_type=loss_type,
                      # from_ckpt='models/pretrain_dnase.pt',
                      dnase=dnase,
                      # load_backbone='weight/4cells-EPCOT-mask_label/lr0.0005-bce-dnase-H3K-pretraine/epoch=09-val_mean_auprc=0.68.ckpt',
                      freeze_backbone=freeze_backbone,
                      num_class=num_class,
                      enc_layers=enc_layers,
                      max_seq_len=max_seq_len,
                      num_heads=num_heads,
                      compile=False,
                      dropout=0.1,
                      hidden_dim=hidden_dim)



# model = EPCOTBackboneReg(args)
# model = DiffTransformerReg(args)
# model = HyenaBackboneReg(args)
model = CNNDiffTransformerReg(args)

trainset = utils.MPRADataset('../EPCOT/Data/Table_S2__MPRA_dataset.txt', set_type='train')
validset = utils.MPRADataset('../EPCOT/Data/Table_S2__MPRA_dataset.txt', set_type='valid')
testset = utils.MPRADataset('../EPCOT/Data/Table_S2__MPRA_dataset.txt', set_type='test')

if project_name=='test':
    trainset=Subset(trainset,range(20))
    validset=Subset(validset,range(20))
    # testset=Subset(testset,range(20))
    wandb_logger = None
else:
    wandb.init(settings=wandb.Settings(_disable_stats=True), name=experiment_name, project=project_name, dir='./weight', mode='offline')
    wandb_logger=WandbLogger(name=experiment_name, project=project_name, save_dir='./weight',)
    # wandb_logger.watch(model, log='all', log_freq=100)
    # wandb_logger = None

checkpoint_callback = ModelCheckpoint(
        monitor='val_pr',  # 监控验证集损失
        dirpath=f'./weight/{project_name}/{experiment_name}/',
        filename='{epoch:02d}-{val_pr:.2f}',
        save_top_k=1,
        mode='max'
    )
early_stopping = EarlyStopping(monitor='val_pr', patience=5, mode='max')
checkpoint_callback_latest = ModelCheckpoint(
    monitor=None,  # 不监控特定指标，只保存最新模型
    save_top_k=1,  # 保存1个模型
    dirpath=f'./weight/{project_name}/{experiment_name}/',  # 保存路径
    filename=f'{experiment_name}-latest'  # 文件名格式
)
# profiler = AdvancedProfiler(dirpath='profile.txt')
trainer = Trainer(
    callbacks=[
        early_stopping,
        checkpoint_callback,
        checkpoint_callback_latest
    ],
    max_epochs=epochs,accelerator='gpu',
    logger=wandb_logger, default_root_dir=f'./weight/{project_name}',
    log_every_n_steps=1,
    # accumulate_grad_batches=2,
    precision='bf16-mixed',
    # profiler=profiler
)

# DataLoader
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16)

trainer.fit(
    model, trainloader, validloader,
    # ckpt_path='weight/MPRA-EPCOT_DiffT/lr0.0001-mse-3striped-abs-dp0.1/lr0.0001-mse-3striped-abs-dp0.1-latest.ckpt'
)

model.test_length=len(testset)
testloader = DataLoader(testset,batch_size=batch_size, drop_last=False, num_workers=1)
test_ckpt = glob.glob(f'weight/{project_name}/{experiment_name}/epoch=*-val_pr=*.ckpt')

if len(test_ckpt)==0:
    test_ckpt = None
else:
    test_ckpt = test_ckpt[0]

trainer.test(
    model, testloader,
    ckpt_path=test_ckpt,
)


print()
# ws.Beep(1, 3)
