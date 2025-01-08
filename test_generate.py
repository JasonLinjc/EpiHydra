from torch.utils.data import DataLoader

from bytelatent.data.patcher import PatcherArgs
from bytelatent.model.blt import ByteLatentTransformerArgs, LocalEncoderArgs, get_blt_input, init_embeddings, \
    EmbeddingType, compute_hash_embeddings, patch_ids_from_lengths, ByteLatentTransformer

from model import utils
from model.blt import BLT
from model.utils import MPRADataset, ByteLevelTextDataset

device = 'cuda'
threshold = 1
entropy_ckpt = 'weight/MPRA-EntropyModel/entropy_model-256-wikitext-new/epoch=44-val_loss=1.20.ckpt'
trainset = utils.MPRADataset('./data/MPRADataset_200.txt', set_type='train', patches_path='./0.1train_patching_mono.pkl')
### load models
# patch_args = PatcherArgs(realtime_patching=True, threshold=threshold, output_channels=130, monotonicity=True,
#                          entropy_model_checkpoint_dir=entropy_ckpt)
# patcher = Patcher(patch_args)

args = ByteLatentTransformerArgs(vocab_size=130, data_loader_patching=False, realtime_patching=True,
                                 patching_threshold=threshold, monotonicity=True, entropy_model_checkpoint_dir=entropy_ckpt)
model = BLT.load_from_checkpoint(checkpoint_path='weight/MPRA-EPCOT_DiffT/mse-0.1-blt_text-test/epoch=03-val_loss=1.27.ckpt', args=args).to(device)

while True:
    prompt = input()
    print(model.generate_text(prompt))


