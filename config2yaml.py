from bytelatent.model.blt import ByteLatentTransformerArgs

enc_layers = 3
hidden_dim = 256
loss_type = 'mse'
freeze_backbone = False
num_class = 12
dnase = False
max_seq_len = 1800

args = ByteLatentTransformerArgs(vocab_size=5, max_length=max_seq_len, max_encoder_seq_length=max_seq_len,
                                 max_patch_length=max_seq_len, max_seqlen=max_seq_len )