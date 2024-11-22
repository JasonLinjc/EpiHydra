import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from model import utils
from model.cnn import ExperimentArgs, EPCOTBackboneClass
from model.transformer.diff_transformer import DiffTransformerClass

cell='4cells'
# project_name=cell+'-EPCOT-DiffT'
project_name='test'
positive_threshold = 0.3
n_decoder_layers = 3
loss_type = 'bce'

num_class = 12
dnase = True

lr = 0.0001
weight_decay = 1e-6

# experiment_name = f'lr{lr}-alpha{alpha}-beta{beta}-factor-contra{positive_threshold}-d_bn{d_bottle_neck}-ch{d_encoder}-{loss_type}-{d_model}_{n_layer}'
experiment_name = f'lr{lr}-{loss_type}-DiffT'
epochs = 50
if project_name == 'test':
    epochs = 1

batch_size=16
args = ExperimentArgs(loss_type=loss_type,
                      # from_ckpt='models/pretrain_dnase.pt',
                      dnase=dnase,
                      # load_backbone='weight/4cells-EPCOT-mask_label/lr0.0005-bce-dnase-H3K-pretraine/epoch=09-val_mean_auprc=0.68.ckpt',
                      num_class=num_class)

# model = EPCOTBackboneClass(args)
model = DiffTransformerClass(args)


trainset = utils.EPCOTDataset('../EPCOT/data/4cells245_train.h5')
validset = utils.EPCOTDataset('../EPCOT/data/4cells245_valid.h5')
testset = utils.EPCOTDataset('../EPCOT/data/4cells245_test.h5')

if project_name=='test':
    trainset=Subset(trainset,range(20))
    validset=Subset(validset,range(20))
    # testset=Subset(testset,range(20))
    wandb_logger = None
else:
    wandb_logger=WandbLogger(name=experiment_name, project=project_name, save_dir='./weight/')
    wandb_logger.watch(model, log='all', log_freq=100)

checkpoint_callback = ModelCheckpoint(
        monitor='val_mean_auprc',  # 监控验证集损失
        dirpath=f'./weight/{project_name}/{experiment_name}/',
        filename='{epoch:02d}-{val_mean_auprc:.2f}',
        save_top_k=1,
        mode='max'
    )
early_stopping = EarlyStopping(monitor='val_mean_auprc', patience=5, mode='max')
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
