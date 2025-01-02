import os
from torch.utils.data import DataLoader, Subset

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb

from model import utils
from model.cnn import ExperimentArgs, DeepCNNBLT
from model.utils import get_callbacks

import glob

os.environ['WANDB_MODE']='offline'

cell='MPRA'
project_name=cell+'-EPCOT_DiffT'
# project_name='test'
dec_layers = 3

enc_layers = 3
hidden_dim = 256
loss_type = 'mse'
freeze_backbone = False
num_class = 12
dnase = False
max_seq_len = 200

num_heads=8

lr = 0.0001
weight_decay = 1e-6

threshold=0.1
experiment_name = f'{loss_type}-{threshold}-blt_deep_cnn-cross-rela'
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
                      dropout=0.3,
                      hidden_dim=hidden_dim)


# model = EPCOTBackboneReg(args)
model = DeepCNNBLT(args)
# model = BassetBranched(args=args)
# model = DiffTransformerReg(args)
# model = HyenaBackboneReg(args)
# model = CNNDiffTransformerReg(args)
# model = LegNet(args)

# trainset = utils.MPRADataset('./data/MPRA_dataset1.txt', set_type='train')
# validset = utils.MPRADataset('./data/MPRA_dataset1.txt', set_type='valid')
# testset = utils.MPRADataset('./data/MPRA_dataset1.txt', set_type='test')

trainset = utils.MPRADataset('./data/MPRADataset_200.txt', set_type='train', patches_path='./0.1train_patching_mono.pkl')
validset = utils.MPRADataset('./data/MPRADataset_200.txt', set_type='valid', patches_path='./0.1valid_patching_mono.pkl')
testset = utils.MPRADataset('./data/MPRADataset_200.txt', set_type='test', patches_path='./0.1test_patching_mono.pkl')

if project_name=='test':
    trainset=Subset(trainset,range(20))
    validset=Subset(validset,range(20))
    # testset=Subset(testset,range(20))
    wandb_logger = None
else:
    wandb.init(settings=wandb.Settings(_disable_stats=True), name=experiment_name, project=project_name, dir='./weight', mode='offline')
    wandb_logger=WandbLogger(name=experiment_name, project=project_name, save_dir='./weight',)

callbacks= get_callbacks(monitor='val_par', monitor_mode='max', project_name=project_name, experiment_name=experiment_name)
trainer = Trainer(
    callbacks=callbacks,
    max_epochs=epochs, accelerator='cuda',
    logger=wandb_logger, default_root_dir=f'./weight/{project_name}',
    log_every_n_steps=1,
    # accumulate_grad_batches=2,
    precision='bf16-mixed',
)

# DataLoader
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16)

trainer.fit(
    model, trainloader, validloader,
    # ckpt_path='weight/MPRA-EPCOT_DiffT/lr0.0001-mse-3striped-abs-dp0.1/lr0.0001-mse-3striped-abs-dp0.1-latest.ckpt'
)

model.test_length=len(testset)
testloader = DataLoader(testset, batch_size=1, drop_last=False, num_workers=1)
test_ckpt = glob.glob(f'weight/{project_name}/{experiment_name}/epoch=*-val_pr=*.ckpt')

if len(test_ckpt)==0:
    test_ckpt = None
else:
    test_ckpt = test_ckpt[0]

trainer.test(
    model, testloader,
    ckpt_path=test_ckpt,
    # ckpt_path='weight/MPRA-EPCOT_DiffT/mse-12/256-diff_t_cnn_embed/epoch=18-val_pr=0.79.ckpt'
)

print()
# ws.Beep(1, 3)
