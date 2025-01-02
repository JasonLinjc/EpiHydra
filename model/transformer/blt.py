from torch import nn

from bytelatent.data.patcher import PatcherArgs, Patcher
from bytelatent.model.blt import ByteLatentTransformerArgs, init_embeddings, EmbeddingType, compute_hash_embeddings, \
    patch_ids_from_lengths
from bytelatent.model.local_models import LocalEncoder


class BLTLocalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_args = ByteLatentTransformerArgs()
        # patch_args = PatcherArgs()
        self.local_encoder = LocalEncoder(encoder_args)
        # self.patcher = Patcher(patch_args)
        self.encoder_hash_tok_embedding = init_embeddings(
            encoder_args,
            EmbeddingType.HASH_TOK,
            local_encoder_dim=self.local_encoder.dim,
            encoder_hash_byte_group_size=encoder_args.encoder_hash_byte_group_size,
        )
        # self.input_embed = nn.Sequential(
        #     nn.Embedding(num_embeddings=4096, embedding_dim=args.hidden_dim),
        #     nn.LayerNorm([48,256]),
        # )
        self.encoder_hash_byte_group_nb_functions = encoder_args.encoder_hash_byte_group_nb_functions
        self.encoder_hash_byte_group_size = encoder_args.encoder_hash_byte_group_size
        self.encoder_hash_byte_group_vocab = encoder_args.encoder_hash_byte_group_vocab

    def forward(self, local_encoder_tokens, patch_lengths):

        local_encoder_embeds = compute_hash_embeddings(
            local_encoder_tokens=local_encoder_tokens,
            local_encoder=self.local_encoder,
            encoder_hash_tok_embedding=self.encoder_hash_tok_embedding,
            encoder_hash_byte_group_nb_functions=self.encoder_hash_byte_group_nb_functions,
            encoder_hash_byte_group_size=self.encoder_hash_byte_group_size,
            encoder_hash_byte_group_vocab=self.encoder_hash_byte_group_vocab,
        )
        # patch_lengths, _ = self.patcher.patch(local_encoder_tokens)
        # patch_start_ids = torch.Tensor([0,1,2,3,4,5,6])
        # seq_len=7
        # patch_lengths = patch_lengths_from_start_ids(patch_start_ids, seq_len)
        patch_ids = patch_ids_from_lengths(
            patch_lengths, local_encoder_tokens.shape[-1]
        )
        (h_encoder, h_cross), cache_encoder = self.local_encoder(
            tokens=local_encoder_tokens,
            embeds=local_encoder_embeds,
            patch_embeds=None,
            cross_mask=None,
            num_patches=patch_lengths.shape[1],
            patch_ids=patch_ids,
        )
        return h_cross
