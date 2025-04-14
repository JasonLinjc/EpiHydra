from torch import nn

from bytelatent.data.patcher import PatcherArgs, Patcher
from bytelatent.model.blt import ByteLatentTransformerArgs, init_embeddings, EmbeddingType, compute_hash_embeddings, \
    patch_ids_from_lengths, cross_attn_mask
from bytelatent.model.local_models import LocalEncoder
from bytelatent.model.utils import downsample
from model.utils import seq2onehot
import torch.nn.functional as F


class BLTLocalEncoder(nn.Module):
    def __init__(self, encoder_args):
        super().__init__()
        # patch_args = PatcherArgs()
        self.cross_attn_encoder = encoder_args.cross_attn_encoder
        self.local_encoder = LocalEncoder(encoder_args)
        self.downsampling_by_pooling = encoder_args.downsampling_by_pooling
        self.patch_size = 10
        self.use_cross_attn_mask = encoder_args.use_cross_attn_mask
        self.patches_as_queries = encoder_args.patches_as_queries
        # self.patcher = Patcher(patch_args)

        self.encoder_hash_tok_embedding = init_embeddings(
            encoder_args,
            EmbeddingType.HASH_TOK,
            local_encoder_dim=self.local_encoder.dim,
            encoder_hash_byte_group_size=encoder_args.encoder_hash_byte_group_size,
        )
        self.input_embed = nn.Sequential(
            nn.Conv1d(4, 256, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )
        self.encoder_hash_byte_group_nb_functions = encoder_args.encoder_hash_byte_group_nb_functions
        self.encoder_hash_byte_group_size = encoder_args.encoder_hash_byte_group_size
        self.encoder_hash_byte_group_vocab = encoder_args.encoder_hash_byte_group_vocab

        # self.cross_linear = nn.Linear(256,256)

    def forward(self, local_encoder_tokens, patch_lengths):
        # hash_embeds = compute_hash_embeddings(
        #     local_encoder_tokens=local_encoder_tokens,
        #     local_encoder=self.local_encoder,
        #     encoder_hash_tok_embedding=self.encoder_hash_tok_embedding,
        #     encoder_hash_byte_group_nb_functions=self.encoder_hash_byte_group_nb_functions,
        #     encoder_hash_byte_group_size=self.encoder_hash_byte_group_size,
        #     encoder_hash_byte_group_vocab=self.encoder_hash_byte_group_vocab,
        # )

        local_encoder_embeds = self.input_embed(seq2onehot(local_encoder_tokens)).transpose(1,2)
        # local_encoder_embeds = local_encoder_embeds + hash_embeds
        patch_ids = patch_ids_from_lengths(
            patch_lengths, local_encoder_tokens.shape[-1]
        )

        if self.use_cross_attn_mask:
            cross_attn_mask_enc = cross_attn_mask(
                patch_ids,
                patch_lengths,
                199,
                patches_as_queries=self.patches_as_queries,
            )
        else:
            cross_attn_mask_enc = None

        (h_encoder, h_cross), cache_encoder = self.local_encoder(
            tokens=local_encoder_tokens,
            embeds=local_encoder_embeds,
            patch_embeds=None,
            cross_mask=cross_attn_mask_enc,
            num_patches=patch_lengths.shape[1],
            patch_ids=patch_ids,
        )
        if not self.cross_attn_encoder and self.downsampling_by_pooling is not None:
            assert (
                patch_ids.shape[1] == h_encoder.shape[1]
            ), f"{patch_ids.shape[1]} != {h_encoder.shape[1]}"
            ## modified
            h_cross = downsample(
                h_encoder,
                patch_lengths.shape[1],
                patch_lengths,
                patch_ids,
                downsampling_by_pooling=self.downsampling_by_pooling,
                patch_size=self.patch_size,
            )

        # h_cross = F.softmax(self.cross_linear(h_cross) + h_cross, dim=-1)
            # h = h_encoder
        # else:
        #     # Reshape h_cross
        #     h = h_cross.view(bs, patch_lengths.shape[1], -1)
        # return h_cross
        return h_encoder + h_cross if h_cross is not None else h_encoder
