import math

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from torch import nn

from .rms_norm import RMSNorm
from .rotary import apply_rotary_emb


class MultiheadFlashDiff2(nn.Module):
    """
    DiffAttn implemented with FlashAttention, for packages that does not support different qk/v dimensions
    e.g., flash-attention (https://github.com/Dao-AILab/flash-attention)
    """

    def __init__(
            self,
            # args,
            embed_dim,
            depth,
            num_heads,
            max_seq_len,
            rotary_embedding=True,
            output_project=True,
    ):
        super().__init__()
        # self.args = args
        self.embed_dim = embed_dim
        # self._precomputed_freqs_cis = None

        self.max_seq_len = max_seq_len
        # num_heads set to half of Transformer's #heads
        self.num_heads = num_heads  # // args.model_parallel_size
        self.num_kv_heads = num_heads  # args.decoder_kv_attention_heads // args.model_parallel_size if args.decoder_kv_attention_heads is not None else num_heads // args.model_parallel_size
        self.n_rep = self.num_heads // self.num_kv_heads

        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5

        angle = 1.0 / (10000 ** torch.linspace(0, 1, self.head_dim // 2, dtype=torch.float))
        index = torch.arange(self.max_seq_len).to(angle)
        self._precomputed_freqs_cis = index[:, None] * angle
        self._precomputed_freqs_cis = self._precomputed_freqs_cis.cuda()

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)

        if output_project:
            self.out_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim, bias=False),
                # nn.ReLU(),
            )
        else:
            self.out_proj = None

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
        self.rotary_embedding = rotary_embedding
        # self.subln1 = RMSNorm(embed_dim, eps=1e-5, elementwise_affine=True)
        # self.subln2 = RMSNorm(embed_dim, eps=1e-5, elementwise_affine=True)

    def forward(
            self,
            x,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2, self.head_dim)

        if self.rotary_embedding:
            rel_pos = self.build_rel_pos(q, 0)
            q = apply_rotary_emb(q, *rel_pos, interleaved=True)
            k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        q = q.reshape(bsz, tgt_len, self.num_heads, 2, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_kv_heads, 2, self.head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        v1, v2 = v[:, :, :, 0], v[:, :, :, 1]

        attn11 = flash_attn_func(q1, k1, v1, causal=False)
        attn12 = flash_attn_func(q1, k1, v2, causal=False)
        attn1 = torch.cat([attn11, attn12], dim=-1)

        attn21 = flash_attn_func(q2, k2, v1, causal=False)
        attn22 = flash_attn_func(q2, k2, v2, causal=False)
        attn2 = torch.cat([attn21, attn22], dim=-1)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        if self.out_proj is not None:
            attn = self.out_proj(attn)
        # if self.out_proj is not None:
        #     short_cut = attn
        #     attn = self.out_proj(attn)
        #
        #     attn = attn + short_cut
        #     attn = self.subln2(attn)
        return attn

    def build_rel_pos(self, x, start_pos):
        cos = torch.cos(self._precomputed_freqs_cis[start_pos:start_pos + x.size(1)])
        sin = torch.sin(self._precomputed_freqs_cis[start_pos:start_pos + x.size(1)])
        rel_pos = (cos.to(x.dtype), sin.to(x.dtype))
        return rel_pos

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

