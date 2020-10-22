import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyAttentionModelWithPooling(nn.Module):
    """My Attention model, based on the Transformer encoder."""

    def __init__(self, vocab_size, embedding_dim, max_seq_len, num_heads, dim_feedforward, num_layers, dropout=0.5):
        super(MyAttentionModelWithPooling, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout=dropout, max_len=max_seq_len)
        encoder_layers = TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward, dropout)
        # output shape (batch_size, max_seq_len, embedding_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers) 
        self.encoder = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.decoder = nn.Linear(self.embedding_dim, 1)
        self.batch_normer = torch.nn.BatchNorm1d(self.embedding_dim)
        self.decoder_act = nn.Sigmoid()

        self.pooler = nn.AvgPool1d(max_seq_len, stride=1)

        self.init_weights()

    def _generate_square_both_ways_mask(self, sz):
        mask = torch.ones(sz, sz)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, word_position_mask, has_mask=True, padding_mask=None):
        # S: source sequence length
        # N: batch size
        # E: encoding
        # wanted src shape to transformer encoder: (S, N, E)

        src = src.transpose(0, 1) # to (S, N)
        word_position_mask = word_position_mask.transpose(0, 1) # to (S, N)

        if has_mask: # [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked positions
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_both_ways_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        #print(f"Input shape: {src.shape}")
        #print(f"Mask shape: {word_position_mask.shape}")
        src = self.encoder(src.transpose(0,1)).transpose(0, 1) * math.sqrt(self.embedding_dim)
        #print(f"Embedded src shape: {src.shape}")
        skip_src = torch.masked_select(src, word_position_mask.unsqueeze(-1).expand(src.shape)).view(-1, self.embedding_dim)
        #print(f"Embedded src of interest: {src_of_interest}")
        #print(f"Alternative src of interest shape: {alternative_src.shape}")
        #print(f"Alternative src of interest: {alternative_src/math.sqrt(self.embedding_dim)}")
        #print(h)
        src = self.pos_encoder(src)
        #print(f"Positional encoding shape: {src.shape}")
        #print(f"src mask shape: {self.src_mask.shape}")
        output = self.transformer_encoder(src, mask=self.src_mask, src_key_padding_mask=padding_mask)
        #print(f"Transformer encoder output shape: {output.shape}")
        ##output = torch.masked_select(output, word_position_mask.unsqueeze(-1).expand(output.shape)).view(-1, self.embedding_dim)
        #print(f"Masked output shape: {output.shape}")
        ##output = torch.add(output, skip_src)
        #print(f"Added output shape: {output.shape}")
        ##output = self.batch_normer(output)
        output = self.pooler(output.transpose(0, 1).transpose(1, 2)).squeeze(2)
        #print(f"Pooler output shape: {output.shape}")
        output = self.decoder(output)
        #print(f"Decoder output shape: {output.shape}")
        return self.decoder_act(output)