import numpy as np
import torch

from bytelatent.data.patcher import PatcherArgs, Patcher
from model import utils
import matplotlib.pyplot as plt
import torch.nn.functional as F

from model.entropy_model import EntropyModel
from model.utils import CaduceusTokenizer, ByteLevelTextDataset

threshold=0.01
entropy_ckpt = 'weight/MPRA-EntropyModel/entropy_model-256-full_set/epoch=19-val_loss=1.1159042119979858.ckpt'

### fetch a sample
trainset = utils.MPRADataset('./data/MPRADataset_200.txt', set_type='train')
# trainset = ByteLevelTextDataset(file_path='./data/wikitext-2/train.csv', seq_length=200)
# trainset = utils.DHSDataset('./data/DHS.csv', set_type='testing')
base_sequence, _ = trainset[1]
base_sequence = base_sequence.unsqueeze(0).cuda()

### load models
model = EntropyModel.load_from_checkpoint(
    checkpoint_path=entropy_ckpt,
    hidden_dim=256, num_heads=4, max_seq_len=199, output_channel=5
)
args = PatcherArgs(realtime_patching=True, threshold=threshold, monotonicity=True, entropy_model_checkpoint_dir=entropy_ckpt)
patcher = Patcher(args)
model = model.to(torch.bfloat16)

### inference
scores = model(base_sequence[:, :-1])
log_probs = F.log_softmax(scores, dim=-1)
probs = torch.exp(log_probs)
p_log_p = log_probs * probs
entropy = -p_log_p.sum(dim=-1)
entropy = entropy.squeeze().detach().cpu().to(torch.float).numpy()
# entropy = np.hstack((None, entropy, None))
entropy = np.hstack((None, entropy))

### calculate start id
batch_patch_length, _ = patcher.patch(base_sequence)
batch_patch_length = batch_patch_length.squeeze().cpu().numpy()
length_sum=0
start_ids=[]
for i in batch_patch_length:
    length_sum+=i
    start_ids.append(length_sum)

### ascii to string
# base_sequence=np.char.mod('%c', base_sequence.squeeze().cpu().numpy())
### token2dna
seq = np.array(("A", "C", "G", "T", "N"))
base_sequence = seq[base_sequence.squeeze().cpu().numpy()]
# base_sequence = np.hstack((base_sequence, 'EOS'))
# base_sequence[0]='SOS'

### plot
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
    plt.axvline(x=start_id-1, color='r', linestyle='--')

plt.show()