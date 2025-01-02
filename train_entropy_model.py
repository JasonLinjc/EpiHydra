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
from model.entropy_model import EntropyModel

import glob

os.environ['WANDB_MODE']='offline'

cell='MPRA'
# project_name=cell+'-EntropyModel'
project_name='test'
dec_layers = 3

enc_layers = 6
hidden_dim = 256
loss_type = 'mse'
freeze_backbone = False
num_class = 12
dnase = False
max_seq_len = 199

num_heads=4

lr = 0.0001
weight_decay = 1e-6

# experiment_name = f'lr{lr}-alpha{alpha}-beta{beta}-factor-contra{positive_threshold}-d_bn{d_bottle_neck}-ch{d_encoder}-{loss_type}-{d_model}_{n_layer}'
# experiment_name = f'lr{lr}-{loss_type}-3striped-abs-dp0.1'
experiment_name = f'entropy_model-{hidden_dim}-wikitext'
epochs = 200
if project_name == 'test':
    epochs = 1

batch_size=32

model = EntropyModel(hidden_dim, num_heads, max_seq_len)

# trainset = utils.DHSDataset('./data/MPR.csv', set_type='training')
# validset = utils.DHSDataset('./data/DHS.csv', set_type='validation')
# testset = utils.DHSDataset('./data/DHS.csv', set_type='testing')
# trainset = utils.ByteLevelTextDataset(file_path='./data/wikitext-2/train.csv', seq_length=200)
# validset = utils.ByteLevelTextDataset(file_path='./data/wikitext-2/test.txt', seq_length=200)
# testset = utils.ByteLevelTextDataset(file_path='./data/wikitext-2/test.txt', seq_length=200)
trainset = utils.MPRADataset('./data/MPRADataset_200.txt', set_type='train')
validset = utils.MPRADataset('./data/MPRADataset_200.txt', set_type='valid')
testset = utils.MPRADataset('./data/MPRADataset_200.txt', set_type='train')

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
callbacks= utils.get_callbacks(monitor='val_par', monitor_mode='max', project_name=project_name, experiment_name=experiment_name)

trainer = Trainer(
    callbacks=callbacks,
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
    # ckpt_path='weight/MPRA-EntropyModel/entropy_model-256/epoch=04-val_loss=3.584178998039533e-08.ckpt'
)

model.test_length=len(testset)
testloader = DataLoader(testset, batch_size=batch_size, drop_last=False, num_workers=1)
test_ckpt = glob.glob(f'weight/{project_name}/{experiment_name}/epoch=*-val_loss=*.ckpt')

if len(test_ckpt)==0:
    test_ckpt = None
else:
    test_ckpt = test_ckpt[0]

trainer.test(
    model, testloader,
    ckpt_path=test_ckpt,
    # ckpt_path='weight/MPRA-EntropyModel/entropy_model-256/epoch=55-val_loss=1.119520902633667.ckpt'
)


print()
# ws.Beep(1, 3)
