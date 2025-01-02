import pickle

import torch
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import numpy as np

from bytelatent.data.patcher import Patcher, PatcherArgs, calculate_entropies
from model import utils
from model.entropy_model import EntropyModel

threshold=0.01
# threshold=1.33
set_type='train'
entropy_ckpt = 'weight/MPRA-EntropyModel/entropy_model-256-full_set/epoch=19-val_loss=1.1159042119979858.ckpt'

### dataset
train_set = utils.MPRADataset('./data/MPRADataset_200.txt', set_type=set_type)
train_loader = DataLoader(dataset=train_set, batch_size=256, shuffle=False, drop_last=False)
# train_set = Subset(train_set, range(10))

### load model
args = PatcherArgs(realtime_patching=True, threshold=threshold, monotonicity=True, entropy_model_checkpoint_dir=entropy_ckpt)
patcher = Patcher(args)

### patch data
patch_lengths = []
for data, _ in tqdm(train_loader):
    data = data.cuda()
    batch_patch_length, _ = patcher.patch((data))
    batch_patch_length = batch_patch_length.cpu().numpy()
    patch_lengths+=[patch_length for patch_length in batch_patch_length]

### save patches
with open(f'./{threshold}{set_type}_patching_mono.pkl', 'wb') as f:
    pickle.dump(patch_lengths, f)