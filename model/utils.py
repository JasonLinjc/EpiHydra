import torch
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from scipy.stats import pearsonr
from torch.nn import functional as F
from transformers import AutoTokenizer
# from src.dataloaders.utils.mlm import mlm_getitem
import pandas as pd
import numpy as np
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import random
import h5py
from Bio.Seq import Seq
import torch.nn.functional as F

class MPRADataset(torch.utils.data.Dataset):
    def __init__(self, path, set_type):
        self.set_type = set_type
        self.tokenizer = Tokenizer(200)

        activity_columns = ['K562_log2FC', 'HepG2_log2FC', 'SKNSH_log2FC', 'sequence', 'type']
        self.data = pd.read_csv(path, usecols=activity_columns)
        self.data = self.data[self.data['type'] == set_type]
        self.length = len(self.data)

    def __getitem__(self, item):
        seq = self.data.iloc[item]['sequence']
        while len(seq)<200:
            item = random.randint(0, self.length-1)
            seq = self.data.iloc[item]['sequence']

        seq = self.tokenizer(seq, truncation=True, add_special_tokens=False)['input_ids']
        seq = torch.LongTensor(seq)
        if len(seq) > 200:
            seq = seq[:200]

        target = np.array(self.data.iloc[item].loc[['K562_log2FC', 'HepG2_log2FC', 'SKNSH_log2FC']],dtype=np.float32)
        return seq, target

    def __len__(self):
        return self.length

class EPCOTDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            path,
            tokenizer=None,
            max_length=1600,
            one_hot=False,
            data_augment=None,
            # contrastive = False,
    ):
        self.one_hot = one_hot
        # self.contrastive = contrastive
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            # self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer = Tokenizer(max_length)

        self.data_augment = data_augment
        self.path = path

        with h5py.File(path, 'r') as f:
            targets = f['targets'][:]
            self.length = len(targets)
            # seqs = f['seqs']
            # dnases = f['dnases']
            # # 将数据集内容读取为NumPy数组
            # self.targets = targets[:]
            # self.seqs = seqs[()]
            # self.dnases = dnases[:]

        # self.H3K_mask = [236, 237, 238, 239, 240, 242, 243, 244]
        self.H3K_mask = [0, 56, 149, 236, 237, 238, 239, 240, 241, 242, 243, 244]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""
        with h5py.File(self.path, 'r') as f:
            target = f['targets'][idx]
            seq = f['seqs'][idx].decode()
            dnase = f['dnases'][idx]

        target = torch.Tensor(target)
        dnase = torch.Tensor(dnase)

        if self.data_augment:
            if random.random() < self.data_augment:
                seq = str(Seq(seq).reverse_complement())
                target = target.flip([1])

        seq = self.tokenizer(seq, truncation=True, add_special_tokens=False)

        # convert to tensor
        seq = torch.LongTensor(seq['input_ids'])

        return seq, target[self.H3K_mask], dnase

    def calculate_hamming_disstance(self, x, y):
        positive_count = (x > 0).sum(-1)
        if positive_count <= 1:
            threshold = 0
        else:
            threshold = positive_count // 10

        hamming_distance = np.sum(x != y)
        return hamming_distance <= threshold


def focal_loss(
        p: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "mean",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    # p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy(p, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


from typing import List, Sequence, Optional, Dict, Tuple
from transformers import PreTrainedTokenizer


class Tokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids"]

    def __init__(self,
                 model_max_length: int,
                 characters: Sequence[str] = ("A", "C", "G", "T", "N"),
                 complement_map=None,
                 bos_token="[BOS]",
                 eos_token="[SEP]",
                 sep_token="[SEP]",
                 cls_token="[CLS]",
                 pad_token="[PAD]",
                 mask_token="[MASK]",
                 unk_token="[UNK]",
                 **kwargs):
        """Character tokenizer for Hugging Face transformers.

        Adapted from https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen-hf/blob/main/tokenization_hyena.py
        Args:
            model_max_length (int): Model maximum sequence length.
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                [UNK] with id=6. Following is a list of the special tokens with
                their corresponding ids:
                    "[CLS]": 0
                    "[SEP]": 1
                    "[BOS]": 2
                    "[MASK]": 3
                    "[PAD]": 4
                    "[RESERVED]": 5
                    "[UNK]": 6
                an id (starting at 7) will be assigned to each character.
            complement_map (Optional[Dict[str, str]]): Dictionary with string complements for each character.
        """
        if complement_map is None:
            complement_map = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
        self.characters = characters
        self.model_max_length = model_max_length

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        add_prefix_space = kwargs.pop("add_prefix_space", False)
        padding_side = kwargs.pop("padding_side", "left")

        self._complement_map = {}
        for k, v in self._vocab_str_to_int.items():
            complement_id = self._vocab_str_to_int[complement_map[k]] if k in complement_map.keys() else v
            self._complement_map[self._vocab_str_to_int[k]] = complement_id

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=add_prefix_space,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    @property
    def complement_map(self) -> Dict[int, int]:
        return self._complement_map

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return list(text.upper())  # Convert all base pairs to uppercase

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)  # Note: this operation has lost info about which base pairs were originally lowercase

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        # cls = [self.cls_token_id]
        result = token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int

    # Fixed vocabulary with no vocab file
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple:
        return ()

def seq2onehot(x):
    x -= 7
    x = F.one_hot(x, num_classes=5)
    x = x[..., :4]
    return x.float().transpose(1, 2)


def calculate_auprc(targets, outputs):
    auprcs = []
    num_classes = targets.shape[1]
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(targets[:, i], outputs[:, i])
        auprc = auc(recall, precision)
        auprcs.append(auprc)
    auprcs = np.array(auprcs)
    mean_auprcs = auprcs.mean()
    std_auprcs = auprcs.std()
    return auprcs, mean_auprcs, std_auprcs


def calculate_auroc(targets, outputs):
    aurocs = []
    num_classes = targets.shape[1]
    for i in range(num_classes):
        # 计算AUROC
        auroc = roc_auc_score(targets[:, i], outputs[:, i])
        aurocs.append(auroc)

    aurocs = np.array(aurocs)
    mean_aurocs = aurocs.mean()
    std_aurocs = aurocs.std()

    return aurocs, mean_aurocs, std_aurocs

