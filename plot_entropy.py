import numpy as np
import torch

from bytelatent.data.patcher import PatcherArgs, Patcher
from model import utils
import matplotlib.pyplot as plt
import torch.nn.functional as F

from model.entropy_model import EntropyModel
from model.utils import CaduceusTokenizer, ByteLevelTextDataset

threshold = 0.1
device='cuda'
entropy_ckpt = 'weight/MPRA-EntropyModel/entropy_model-256-MPRA-RC-199/epoch=28-val_loss=1.11.ckpt'

## fetch a sample
trainset = utils.MPRADataset('./seq_50.csv', set_type='train', patches_path='./patch_50.pkl')
# trainset = ByteLevelTextDataset(file_path='./data/wikitext-2/test_text.csv', seq_length=200)
# trainset = utils.DHSDataset('./data/DHS.csv', set_type='testing')
(seq, _), _ = trainset[1]
seq = seq.unsqueeze(0).cuda()

## load models
model = EntropyModel.load_from_checkpoint(
    checkpoint_path=entropy_ckpt,
    hidden_dim=256, num_heads=4, max_length=199, output_channels=5
)
args = PatcherArgs(realtime_patching=True, threshold=threshold, monotonicity=True,
                   entropy_model_checkpoint_dir=entropy_ckpt, output_channels=5, device=device, max_length=199)
patcher = Patcher(args)
model = model.to(torch.bfloat16)

base_sequence = seq[:, :-1]

## inference
scores = model(base_sequence)
log_probs = F.log_softmax(scores, dim=-1)
probs = torch.exp(log_probs)
p_log_p = log_probs * probs
entropy = -p_log_p.sum(dim=-1)
entropy = entropy.squeeze().detach().cpu().to(torch.float).numpy()
entropy = np.hstack((None, entropy))
# entropy = np.hstack((entropy, None))

## calculate start id
batch_patch_length, _ = patcher.patch(base_sequence, include_next_token=True)
batch_patch_length = batch_patch_length.squeeze().cpu().numpy()
length_sum = 0
start_ids = []
for i in batch_patch_length:
    start_ids.append(length_sum)
    length_sum += i

## ascii to string
# base_sequence=np.char.mod('%c', base_sequence.squeeze().cpu().numpy())
## token2dna
seq_dict = np.array(("A", "C", "G", "T", "N"))
base_sequence = seq_dict[seq.squeeze().cpu().numpy()]
# base_sequence = np.hstack(('SOS', base_sequence))
# base_sequence[0]='SOS'

## plot
plt.figure(figsize=(10, 6))
# plt.grid()
plt.plot(range(len(base_sequence)), entropy, marker='o', linestyle='-', color='b')
plt.xticks(range(len(base_sequence)), list(base_sequence))
plt.xlabel('Base Sequence')
plt.ylabel('Entropy')
plt.title('Entropy of Each Base in the Sequence')

# plot start id
plt.tight_layout()
for start_id in start_ids:
    plt.axvline(x=start_id, color='r', linestyle='--')

plt.show()