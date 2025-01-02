import torch
from flash_attn import flash_attn_func
from torch import nn

from model.transformer.rms_norm import RMSNorm
from model.transformer.rotary import apply_rotary_emb


class AttnLayer(nn.Module):
    def  __init__(self, hidden_dim, max_seq_len, num_heads, causal=False, output_project=False, dropout=0.1):
        super().__init__()
        # self.attn_layer = MultiheadFlashDiff2(embed_dim=hidden_dim, depth=depth, max_seq_len=max_seq_len, num_heads=num_heads, output_project=output_project)
        self.attn_layer = MultiheadFlashAttention(embed_dim=hidden_dim, max_seq_len=max_seq_len, num_heads=num_heads, output_project=output_project, causal=causal)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = RMSNorm(hidden_dim, eps=1e-5, elementwise_affine=True)
        self.norm2 = RMSNorm(hidden_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x, inference_params=None, padding_mask=None, *args):
        x = x.transpose(1,2)
        shortcut = x

        x = self.norm1(x)
        x = self.attn_layer(x)
        x = self.dropout(x)

        x = x+shortcut
        x = self.norm2(x)

        return x.transpose(1,2)

class MultiheadFlashAttention(nn.Module):
    """
    DiffAttn implemented with FlashAttention, for packages that does not support different qk/v dimensions
    e.g., flash-attention (https://github.com/Dao-AILab/flash-attention)
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            max_seq_len,
            rotary_embedding=True,
            output_project=True,
            causal=False,
    ):
        super().__init__()
        # self.args = args
        self.embed_dim = embed_dim
        # self._precomputed_freqs_cis = None

        self.max_seq_len = max_seq_len
        self.causal = causal
        # num_heads set to half of Transformer's #heads
        self.num_heads = num_heads  # // args.model_parallel_size
        self.num_kv_heads = num_heads  # args.decoder_kv_attention_heads // args.model_parallel_size if args.decoder_kv_attention_heads is not None else num_heads // args.model_parallel_size
        self.n_rep = self.num_heads // self.num_kv_heads

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        angle = 1.0 / (10000 ** torch.linspace(0, 1, self.head_dim // 2, dtype=torch.float))
        index = torch.arange(self.max_seq_len).to(angle)
        self._precomputed_freqs_cis = index[:, None] * angle
        self._precomputed_freqs_cis = self._precomputed_freqs_cis.cuda()

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        torch.nn.init.xavier_uniform_(self.q_proj.weight)
        torch.nn.init.xavier_uniform_(self.k_proj.weight)
        torch.nn.init.xavier_uniform_(self.v_proj.weight)

        if output_project:
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

            torch.nn.init.xavier_uniform_(self.out_proj.weight)
        else:
            self.out_proj = None

        self.subln = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
        self.rotary_embedding = rotary_embedding

    def forward(
            self,
            x,
    ):
        bsz, tgt_len, embed_dim = x.size()
        dtype = x.dtype
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, self.num_kv_heads, self.head_dim)

        v = self.v_proj(x)
        v = v.view(bsz, src_len, self.num_kv_heads, self.head_dim)
        dtype = q.dtype

        if self.rotary_embedding:

            rel_pos = self.build_rel_pos(dtype, tgt_len, 0)
            q = apply_rotary_emb(q, *rel_pos, interleaved=True)
            k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        q = q.reshape(bsz, tgt_len, self.num_heads, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_kv_heads, self.head_dim)

        # attn = torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
        attn = flash_attn_func(q, k, v, causal=self.causal)

        attn = self.subln(attn)

        attn = attn.reshape(bsz, tgt_len, self.num_heads * self.head_dim)

        if self.out_proj is not None:
            attn = self.out_proj(attn)

        return attn

    def build_rel_pos(self, dtype, length, start_pos):
        cos = torch.cos(self._precomputed_freqs_cis[start_pos:start_pos + length])
        sin = torch.sin(self._precomputed_freqs_cis[start_pos:start_pos + length])
        rel_pos = (cos.to(dtype), sin.to(dtype))
        return rel_pos