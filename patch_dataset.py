import pickle

import pandas as pd
import torch
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import numpy as np

from bytelatent.data.patcher import Patcher, PatcherArgs, calculate_entropies
from model import utils
from model.entropy_model import EntropyModel

threshold=0.1
# threshold=1.33
set_type='test'
device = 'cuda'
entropy_ckpt = 'weight/MPRA-EntropyModel/entropy_model-256-MPRA-RC-199/epoch=28-val_loss=1.11.ckpt'

## dataset
train_set = utils.MPRADataset('./data/MPRADataset_200.txt', set_type=set_type, reverse_complement=False)
train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=False, drop_last=False, num_workers=16)
# train_loader = pd.read_csv(f'./data/wikitext-2/{set_type}_text.csv')
# train_set = Subset(train_set, range(10))

## load model
args = PatcherArgs(realtime_patching=True, threshold=threshold, monotonicity=True,
                   entropy_model_checkpoint_dir=entropy_ckpt, output_channels=5, device=device, max_length=199)
patcher = Patcher(args)

## patch data
patch_lengths = []
for i, (data, _) in enumerate(tqdm(train_loader)):
    data = data.to(device)
    seq = data[:, :-1]
    # seq = torch.ones_like(data) * 5
    # seq[:, 1:] = data[:, :-1]
    batch_patch_length, _ = patcher.patch((seq))
    batch_patch_length = batch_patch_length.cpu().numpy()
    patch_lengths += [patch_length for patch_length in batch_patch_length]
# train_loader['patch_lengths']=None
# train_loader['patch_lengths'].astype('object')
# for i in tqdm(range(len(train_loader))):
#     ### get seq
#     sequence = train_loader.loc[i, 'seq']
#     token = [ord(ch) for ch in sequence]
#     token = torch.tensor(token, dtype=torch.int64)

    # ### padding
    # if len(token) < 200:
    #     pad = torch.ones((200 - len(token),), dtype=torch.int64) * 129
    #     token = torch.hstack((token, pad))

    ### inference
    # patch_length, _ = patcher.patch(token.unsqueeze(0).cuda())
    # patch_length = list(patch_length.squeeze().cpu().numpy())
    # train_loader.at[i, 'patch_lengths'] = patch_length
    # patch_lengths += [patch_length for patch_length in batch_patch_length]

### save patches
# train_loader.to_csv(f'./{set_type}_text.csv')
with open(f'./{threshold}{set_type}_patching_mono_new.pkl', 'wb') as f:
    pickle.dump(patch_lengths, f)