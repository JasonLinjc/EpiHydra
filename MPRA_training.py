import os
from torch.utils.data import DataLoader, Subset

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb

from bytelatent.model.blt import ByteLatentTransformerArgs
from model import utils
from model.blt import BLT, BLTDNALM
from model.cnn import ExperimentArgs, DeepCNNBLT
from model.utils import get_callbacks, ByteLevelTextDataset
import hydra

import glob

# os.environ['WANDB_MODE']='offline'

cell='MPRA'
project_name=cell+'-EPCOT_DiffT'
# project_name='test'
threshold=0.1
experiment_name = f'{threshold}-blt-mpra-100patch-14m'
if project_name == 'test':
    epochs = 1

trainset = utils.MPRADataset('./data/MPRADataset_200.txt', set_type='train', patches_path='./data/0.1train_patching_mono.pkl')
validset = utils.MPRADataset('./data/MPRADataset_200.txt', set_type='valid', patches_path='./data/0.1valid_patching_mono.pkl')
testset = utils.MPRADataset('./data/MPRADataset_200.txt', set_type='test', patches_path='./data/0.1test_patching_mono.pkl')

if project_name == 'test':
    trainset = Subset(trainset, range(20))
    validset = Subset(validset, range(20))
    # testset=Subset(testset,range(20))
    wandb_logger = None
else:
    wandb.init(settings=wandb.Settings(_disable_stats=True), name=experiment_name, project=project_name,
               dir='./weight', mode='offline')
    wandb_logger = WandbLogger(name=experiment_name, project=project_name, save_dir='./weight', )
    # wandb_logger = None
@hydra.main(config_name='blt', config_path='./configs')
def train(conf, ):
    os.chdir('/home/zapravdu/EpiHydra')
    trainloader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, drop_last=True, num_workers=16)
    validloader = DataLoader(validset, batch_size=conf.batch_size, shuffle=True, drop_last=True, num_workers=16)
    testloader = DataLoader(testset, batch_size=1, drop_last=False, num_workers=1)

    args = ByteLatentTransformerArgs(**conf.blt)
    # model = EPCOTBackboneReg(args)
    # model = DeepCNNBLT(args)
    model = BLTDNALM(args)
    # model = BLT(args, lr=conf.lr, weight_decay=conf.weight_decay)
    # model = DiffTransformerReg(args)
    # model = HyenaBackboneReg(args)
    # model = CNNDiffTransformerReg(args)

    callbacks = get_callbacks(monitor=conf.monitor, monitor_mode=conf.monitor_mode, project_name=project_name,
                              experiment_name=experiment_name)
    trainer = Trainer(
        callbacks=callbacks,
        max_epochs=conf.max_epochs, accelerator=conf.device,
        logger=wandb_logger, default_root_dir=f'/home/zapravdu/EpiHydra/weight/{project_name}',
        # accumulate_grad_batches=2,
        precision=conf.precision,
    )

    trainer.fit(
        model, trainloader, validloader,
        # ckpt_path='weight/MPRA-EPCOT_DiffT/lr0.0001-mse-3striped-abs-dp0.1/lr0.0001-mse-3striped-abs-dp0.1-latest.ckpt'
    )
    model.test_length = len(trainloader)
    test_ckpt = glob.glob(f'weight/{project_name}/{experiment_name}/epoch=*-val_pr=*.ckpt')
    if len(test_ckpt) == 0:
        test_ckpt = None
    else:
        test_ckpt = test_ckpt[0]
    trainer.test(
        model, testloader,
        ckpt_path=test_ckpt,
        # ckpt_path='weight/MPRA-EPCOT_DiffT/mse-12/256-diff_t_cnn_embed/epoch=18-val_pr=0.79.ckpt'
    )

train()
print()
# ws.Beep(1, 3)
