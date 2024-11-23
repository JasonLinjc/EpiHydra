import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from model import utils
from model.cnn import ExperimentArgs, EPCOTBackboneReg
from model.transformer.diff_transformer import DiffTransformerReg

cell='Reg'
project_name=cell+'-EPCOT-200bp'
# project_name='test'

dec_layers = 3
enc_layers = 6

loss_type = 'bce'
freeze_backbone = False
num_class = 12
dnase = False
max_seq_len = 200

num_heads=8


lr = 0.0001
weight_decay = 1e-6

# experiment_name = f'lr{lr}-alpha{alpha}-beta{beta}-factor-contra{positive_threshold}-d_bn{d_bottle_neck}-ch{d_encoder}-{loss_type}-{d_model}_{n_layer}'
experiment_name = f'lr{lr}-{loss_type}-EPCOT_DiffT_cnn_embed'
epochs = 50
if project_name == 'test':
    epochs = 1

batch_size=16
args = ExperimentArgs(loss_type=loss_type,
                      # from_ckpt='models/pretrain_dnase.pt',
                      dnase=dnase,
                      # load_backbone='weight/4cells-EPCOT-mask_label/lr0.0005-bce-dnase-H3K-pretraine/epoch=09-val_mean_auprc=0.68.ckpt',
                      freeze_backbone=freeze_backbone,
                      num_class=num_class,
                      enc_layers=enc_layers,
                      max_seq_len=max_seq_len,
                      num_heads=num_heads)

# model = EPCOTBackboneReg(args)
model = DiffTransformerReg(args)

trainset = utils.MPRADataset('../EPCOT/Data/Table_S2__MPRA_dataset.txt', set_type='train')
validset = utils.MPRADataset('../EPCOT/Data/Table_S2__MPRA_dataset.txt', set_type='valid')
testset = utils.MPRADataset('../EPCOT/Data/Table_S2__MPRA_dataset.txt', set_type='test')

if project_name=='test':
    trainset=Subset(trainset,range(20))
    validset=Subset(validset,range(20))
    # testset=Subset(testset,range(20))
    wandb_logger = None
else:
    wandb_logger=WandbLogger(name=experiment_name, project=project_name, save_dir='./weight/')
    wandb_logger.watch(model, log='all', log_freq=100)

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
trainer = Trainer(
    callbacks=[
        early_stopping,
        checkpoint_callback,
        checkpoint_callback_latest
    ],
    max_epochs=epochs,accelerator='gpu',
    logger=wandb_logger, default_root_dir=f'./weight/{project_name}',
    log_every_n_steps=1,
    accumulate_grad_batches=2,
    precision='bf16-mixed',
)

# DataLoader
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=16)
validloader = DataLoader(validset, batch_size=batch_size,drop_last=False, num_workers=16)

trainer.fit(
    model, trainloader, validloader,
    # ckpt_path='weight/Reg-EPCOT-200bp/lr0.0001-bce-dnase-EPCOT_DiffT_no_linear/lr0.0001-bce-dnase-EPCOT_DiffT_no_linear-latest.ckpt'
)

model.test_length=len(testset)
testloader= DataLoader(testset,batch_size=batch_size, drop_last=False, num_workers=16)
trainer.test(
    model, testloader,
    # ckpt_path='weight/Reg-EPCOT-200bp/lr0.0001-bce-dnase-EPCOT_DiffT_no_linear/lr0.0001-bce-dnase-EPCOT_DiffT_no_linear-latest.ckpt'
)

print()
# ws.Beep(1, 3)
