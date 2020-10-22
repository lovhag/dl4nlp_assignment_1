import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MyAttentionModel(nn.Module):
    """My Attention model, based on the Transformer encoder."""

    def __init__(self, vocab_size, embedding_dim, max_seq_len, num_heads, dim_feedforward, num_layers, dropout=0.5):
        super(MyAttentionModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout=0.1, max_len=max_seq_len)
        encoder_layers = TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward, dropout)
        # output shape (batch_size, max_seq_len, embedding_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers) 
        self.encoder = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.decoder = nn.Linear(embedding_dim*max_seq_len, 1)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        #print(f"Input shape: {src.shape}")
        src = self.encoder(src) * math.sqrt(self.embedding_dim)
        #print(f"Embedded shape: {src.shape}")
        src = self.pos_encoder(src)
        #print(f"Positional encoding shape: {src.shape}")
        output = self.transformer_encoder(src, self.src_mask)
        #print(f"Transformer encoder output shape: {output.shape}")
        # (batch_size, n_tokens, emb_dim) -> (batch_size, 1, emb_dim)
        output = output.view(output.shape[0], -1)
        #print(f"Reshaped output shape: {output.shape}")
        output = self.decoder(output)
        #print(f"Decoder output shape: {output.shape}")
        #print(h)
        return F.log_softmax(output, dim=-1)


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
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout=0.1, max_len=max_seq_len)
        encoder_layers = TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward, dropout)
        # output shape (batch_size, max_seq_len, embedding_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers) 
        self.encoder = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.pooler = nn.AvgPool1d(max_seq_len, stride=1)
        self.decoder = nn.Linear(embedding_dim, 1)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        #print(f"Input shape: {src.shape}")
        src = self.encoder(src) * math.sqrt(self.embedding_dim)
        #print(f"Embedded shape: {src.shape}")
        src = self.pos_encoder(src)
        #print(f"Positional encoding shape: {src.shape}")
        output = self.transformer_encoder(src, self.src_mask)
        #print(f"Transformer encoder output shape: {output.shape}")
        # (batch_size, n_tokens, emb_dim) -> (batch_size, 1, emb_dim)
        output = self.pooler(output.permute(0,2,1))
        #print(f"Pooled output shape: {output.shape}")
        output = self.decoder(output.view(-1,output.shape[1]))
        #print(f"Decoder output shape: {output.shape}")
        #print(h)
        return F.log_softmax(output, dim=-1)

class MyAttentionModelWithMaskOnWordPosition(nn.Module):
    """My Attention model, based on the Transformer encoder."""

    def __init__(self, vocab_size, embedding_dim, max_seq_len, num_heads, dim_feedforward, num_layers, dropout=0.5):
        super(MyAttentionModelWithMaskOnWordPosition, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout=0.1, max_len=max_seq_len)
        encoder_layers = TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward, dropout)
        # output shape (batch_size, max_seq_len, embedding_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers) 
        self.encoder = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.decoder = nn.Linear(embedding_dim, 1)
        self.decoder_act = nn.Sigmoid()

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, word_position_mask, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        #print(f"Input shape: {src.shape}")
        #print(f"Mask shape: {word_position_mask.shape}")
        src = self.encoder(src) * math.sqrt(self.embedding_dim)
        #print(f"Embedded shape: {src.shape}")
        src = self.pos_encoder(src)
        #print(f"Positional encoding shape: {src.shape}")
        #print(f"Word position mask shape after unsqueeze: {word_position_mask.unsqueeze(-1).expand(src.shape).shape}")
        src = torch.masked_select(src, word_position_mask.unsqueeze(-1).expand(src.shape)).view(-1, 1, self.embedding_dim)
        #print(f"Masked output shape: {src.shape}")
        output = self.transformer_encoder(src, self.src_mask)
        #print(f"Transformer encoder output shape: {output.shape}")
        output = self.decoder(output)
        #print(f"Decoder output shape: {output.shape}")
        return self.decoder_act(output.squeeze(2))

class MyAttentionModelWithMaskOnWordPositionAndSkip(nn.Module):
    """My Attention model, based on the Transformer encoder."""

    def __init__(self, vocab_size, embedding_dim, max_seq_len, num_heads, dim_feedforward, num_layers, dropout=0.5):
        super(MyAttentionModelWithMaskOnWordPositionAndSkip, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout=0.1, max_len=max_seq_len)
        encoder_layers = TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward, dropout)
        # output shape (batch_size, max_seq_len, embedding_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers) 
        self.encoder = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.decoder = nn.Linear(embedding_dim, 1)
        self.batch_normer = torch.nn.BatchNorm1d(embedding_dim)
        self.decoder_act = nn.Sigmoid()

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, word_position_mask, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        #print(f"Input shape: {src.shape}")
        #print(f"Mask shape: {word_position_mask.shape}")
        src = self.encoder(src) * math.sqrt(self.embedding_dim)
        #print(f"Embedded src shape: {src.shape}")
        skip_src = torch.masked_select(src, word_position_mask.unsqueeze(-1).expand(src.shape)).view(-1, 1, self.embedding_dim)
        #print(f"Skip src shape: {skip_src.shape}")
        src = self.pos_encoder(src)
        #print(f"Positional encoding shape: {src.shape}")
        #print(f"Word position mask shape after unsqueeze: {word_position_mask.unsqueeze(-1).expand(src.shape).shape}")
        src = torch.masked_select(src, word_position_mask.unsqueeze(-1).expand(src.shape)).view(-1, 1, self.embedding_dim)
        print(f"Masked src shape: {src.shape}")
        output = self.transformer_encoder(src, self.src_mask)
        #print(f"Transformer encoder output shape: {output.shape}")
        output = torch.add(output, skip_src).squeeze(1)
        #print(f"Added output shape: {output.shape}")
        output = self.batch_normer(output)
        output = self.decoder(output)
        #print(f"Decoder output shape: {output.shape}")
        return self.decoder_act(output)

class MyAttentionModelWithMaskOnWordPositionOutputAndSkip(nn.Module):
    """My Attention model, based on the Transformer encoder."""

    def __init__(self, vocab_size, embedding_dim, max_seq_len, num_heads, dim_feedforward, num_layers, dropout=0.5):
        super(MyAttentionModelWithMaskOnWordPositionOutputAndSkip, self).__init__()
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
        self.decoder = nn.Linear(embedding_dim, 1)
        self.batch_normer = torch.nn.BatchNorm1d(max_seq_len)
        self.decoder_act = nn.Sigmoid()

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, word_position_mask, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        #print(f"Input shape: {src.shape}")
        #print(f"Mask shape: {word_position_mask.shape}")
        src = self.encoder(src) * math.sqrt(self.embedding_dim)
        #print(f"Embedded src shape: {src.shape}")
        skip_src = src
        #print(f"Skip src shape: {skip_src.shape}")
        src = self.pos_encoder(src)
        #print(f"Positional encoding shape: {src.shape}")
        output = self.transformer_encoder(src, self.src_mask)
        #print(f"Transformer encoder output shape: {output.shape}")
        output = torch.add(output, skip_src)
        #print(f"Added output shape: {output.shape}")
        output = self.batch_normer(output)
        #print(f"Mask shape: {word_position_mask.unsqueeze(-1).expand(output.shape).shape}")
        output = torch.masked_select(output, word_position_mask.unsqueeze(-1).expand(output.shape)).view(-1, 1, self.embedding_dim)
        #print(f"Masked output shape: {output.shape}")
        output = self.decoder(output).squeeze(2)
        #print(f"Decoder output shape: {output.shape}")
        return self.decoder_act(output)

class MyAttentionModelWithMaskOnWordPositionOutputAndMaskedSkip(nn.Module):
    """My Attention model, based on the Transformer encoder."""

    def __init__(self, vocab_size, embedding_dim, max_seq_len, num_heads, dim_feedforward, num_layers, dropout=0.5):
        super(MyAttentionModelWithMaskOnWordPositionOutputAndMaskedSkip, self).__init__()
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
        self.decoder = nn.Linear(embedding_dim, 1)
        self.batch_normer = torch.nn.BatchNorm1d(1)
        self.decoder_act = nn.Sigmoid()

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, word_position_mask, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        #print(f"Input shape: {src.shape}")
        #print(f"Mask shape: {word_position_mask.shape}")
        src = self.encoder(src) * math.sqrt(self.embedding_dim)
        #print(f"Embedded src shape: {src.shape}")
        skip_src = torch.masked_select(src, word_position_mask.unsqueeze(-1).expand(src.shape)).view(-1, 1, self.embedding_dim)
        #print(f"Skip src shape: {skip_src.shape}")
        src = self.pos_encoder(src)
        #print(f"Positional encoding shape: {src.shape}")
        output = self.transformer_encoder(src, self.src_mask)
        #print(f"Transformer encoder output shape: {output.shape}")
        output = torch.masked_select(output, word_position_mask.unsqueeze(-1).expand(output.shape)).view(-1, 1, self.embedding_dim)
        #print(f"Masked output shape: {output.shape}")
        output = torch.add(output, skip_src)
        #print(f"Added output shape: {output.shape}")
        output = self.batch_normer(output)
        #print(f"Mask shape: {word_position_mask.unsqueeze(-1).expand(output.shape).shape}")
        output = self.decoder(output).squeeze(2)
        #print(f"Decoder output shape: {output.shape}")
        return self.decoder_act(output)

class MyAttentionModelWithMaskOnWordPositionOutputAndMaskedSkipCORRECTED(nn.Module):
    """My Attention model, based on the Transformer encoder."""

    def __init__(self, vocab_size, embedding_dim, max_seq_len, num_heads, dim_feedforward, num_layers, dropout=0.5):
        super(MyAttentionModelWithMaskOnWordPositionOutputAndMaskedSkipCORRECTED, self).__init__()
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
        self.decoder = nn.Linear(embedding_dim, 1)
        self.batch_normer = torch.nn.BatchNorm1d(1)
        self.decoder_act = nn.Sigmoid()

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, word_position_mask, has_mask=True):
        # S: source sequence length
        # N: batch size
        # E: encoding
        # wanted src shape to transformer encoder: (S, N, E)

        src = src.view(self.max_seq_len, -1) # to (S, N)
        word_position_mask = word_position_mask.view(self.max_seq_len, -1) # to (S, N)

        if has_mask: # [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked positions
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        #print(f"Input shape: {src.shape}")
        #print(f"Mask shape: {word_position_mask.shape}")
        src = self.encoder(src) * math.sqrt(self.embedding_dim)
        #print(f"Embedded src shape: {src.shape}")
        skip_src = torch.masked_select(src, word_position_mask.unsqueeze(-1).expand(src.shape)).view(1, -1, self.embedding_dim)
        #print(f"Skip src shape: {skip_src.shape}")
        src = self.pos_encoder(src)
        #print(f"Positional encoding shape: {src.shape}")
        #print(f"src mask shape: {self.src_mask.shape}")
        output = self.transformer_encoder(src, mask=self.src_mask) # will only look at previous tokens for current token 
        #print(f"Transformer encoder output shape: {output.shape}")
        output = torch.masked_select(output, word_position_mask.unsqueeze(-1).expand(output.shape)).view(1, -1, self.embedding_dim)
        #print(f"Masked output shape: {output.shape}")
        output = torch.add(output, skip_src)
        #print(f"Added output shape: {output.shape}")
        output = self.batch_normer(output.view(-1, 1, self.embedding_dim))
        output = self.decoder(output.view(-1, 1, self.embedding_dim)).squeeze(2)
        #print(f"Decoder output shape: {output.shape}")
        return self.decoder_act(output)

class MyAttentionModelWithMaskOnWordPositionOutputAndMaskedSkipCORRECTEDwithMaskOnPadding(nn.Module):
    """My Attention model, based on the Transformer encoder."""

    def __init__(self, vocab_size, embedding_dim, max_seq_len, num_heads, dim_feedforward, num_layers, dropout=0.5):
        super(MyAttentionModelWithMaskOnWordPositionOutputAndMaskedSkipCORRECTEDwithMaskOnPadding, self).__init__()
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
        self.decoder = nn.Linear(embedding_dim, 1)
        self.batch_normer = torch.nn.BatchNorm1d(1)
        self.decoder_act = nn.Sigmoid()

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
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

        src = src.view(self.max_seq_len, -1) # to (S, N)
        word_position_mask = word_position_mask.view(self.max_seq_len, -1) # to (S, N)

        if has_mask: # [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked positions
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        #print(f"Input shape: {src.shape}")
        #print(f"Mask shape: {word_position_mask.shape}")
        src = self.encoder(src) * math.sqrt(self.embedding_dim)
        #print(f"Embedded src shape: {src.shape}")
        skip_src = torch.masked_select(src, word_position_mask.unsqueeze(-1).expand(src.shape)).view(1, -1, self.embedding_dim)
        #print(f"Skip src shape: {skip_src.shape}")
        src = self.pos_encoder(src)
        #print(f"Positional encoding shape: {src.shape}")
        #print(f"src mask shape: {self.src_mask.shape}")
        output = self.transformer_encoder(src, mask=self.src_mask, src_key_padding_mask=padding_mask) # will only look at previous tokens for current token 
        #print(f"Transformer encoder output shape: {output.shape}")
        output = torch.masked_select(output, word_position_mask.unsqueeze(-1).expand(output.shape)).view(1, -1, self.embedding_dim)
        #print(f"Masked output shape: {output.shape}")
        output = torch.add(output, skip_src)
        #print(f"Added output shape: {output.shape}")
        output = self.batch_normer(output.view(-1, 1, self.embedding_dim))
        output = self.decoder(output.view(-1, 1, self.embedding_dim)).squeeze(2)
        #print(f"Decoder output shape: {output.shape}")
        return self.decoder_act(output)

class MyAttentionModelWithMaskOnWordPositionOutputAndMaskedSkipPlusMaskOnPadding(nn.Module):
    """My Attention model, based on the Transformer encoder."""

    def __init__(self, vocab_size, embedding_dim, max_seq_len, num_heads, dim_feedforward, num_layers, dropout=0.5):
        super(MyAttentionModelWithMaskOnWordPositionOutputAndMaskedSkipPlusMaskOnPadding, self).__init__()
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
        self.decoder = nn.Linear(embedding_dim, 1)
        self.batch_normer = torch.nn.BatchNorm1d(1)
        self.decoder_act = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, word_position_mask, padding_mask=None):

        #print(f"Input shape: {src.shape}")
        #print(f"Mask shape: {word_position_mask.shape}")
        src = self.encoder(src) * math.sqrt(self.embedding_dim)
        #print(f"Embedded src shape: {src.shape}")
        skip_src = torch.masked_select(src, word_position_mask.unsqueeze(-1).expand(src.shape)).view(-1, 1, self.embedding_dim)
        #print(f"Skip src shape: {skip_src.shape}")
        src = self.pos_encoder(src)
        print(f"Positional encoding shape: {src.shape}")
        #padding_mask = padding_mask.unsqueeze(-1).expand(src.shape)
        print(f"Padding mask shape: {padding_mask.shape}")
        output = self.transformer_encoder(src, src_key_padding_mask=padding_mask)
        #print(f"Transformer encoder output shape: {output.shape}")
        output = torch.masked_select(output, word_position_mask.unsqueeze(-1).expand(output.shape)).view(-1, 1, self.embedding_dim)
        #print(f"Masked output shape: {output.shape}")
        output = torch.add(output, skip_src)
        #print(f"Added output shape: {output.shape}")
        output = self.batch_normer(output)
        #print(f"Mask shape: {word_position_mask.unsqueeze(-1).expand(output.shape).shape}")
        output = self.decoder(output).squeeze(2)
        #print(f"Decoder output shape: {output.shape}")
        return self.decoder_act(output)

class MyAttentionModelWithPoolingAndSkip(nn.Module):
    """My Attention model, based on the Transformer encoder."""

    def __init__(self, vocab_size, embedding_dim, max_seq_len, num_heads, dim_feedforward, num_layers, dropout=0.5):
        super(MyAttentionModelWithPoolingAndSkip, self).__init__()
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
        self.decoder = nn.Linear(embedding_dim, 1)
        self.batch_normer = torch.nn.BatchNorm1d(max_seq_len)
        self.pooler = nn.AvgPool1d(max_seq_len, stride=1)
        self.decoder_act = nn.Sigmoid()

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, word_position_mask, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        #print(f"Input shape: {src.shape}")
        #print(f"Mask shape: {word_position_mask.shape}")
        src = self.encoder(src) * math.sqrt(self.embedding_dim)
        #print(f"Embedded src shape: {src.shape}")
        skip_src = src
        #print(f"Skip src shape: {skip_src.shape}")
        src = self.pos_encoder(src)
        #print(f"Positional encoding shape: {src.shape}")
        output = self.transformer_encoder(src, self.src_mask)
        #print(f"Transformer encoder output shape: {output.shape}")
        output = torch.add(output, skip_src)
        #print(f"Added output shape: {output.shape}")
        output = self.batch_normer(output)
        #print(f"Mask shape: {word_position_mask.unsqueeze(-1).expand(output.shape).shape}")
        output = self.pooler(output.permute(0,2,1)).squeeze(2)
        #print(f"Pooled output shape: {output.shape}")
        output = self.decoder(output)
        #print(f"Decoder output shape: {output.shape}")
        return self.decoder_act(output)

class MyAttentionModelWithPoolingAndSkipOnWordPosition(nn.Module):
    """My Attention model, based on the Transformer encoder."""

    def __init__(self, vocab_size, embedding_dim, max_seq_len, num_heads, dim_feedforward, num_layers, dropout=0.5):
        super(MyAttentionModelWithPoolingAndSkipOnWordPosition, self).__init__()
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
        self.decoder = nn.Linear(embedding_dim, 1)
        self.batch_normer = torch.nn.BatchNorm1d(max_seq_len)
        self.pooler = nn.AvgPool1d(max_seq_len, stride=1)
        self.decoder_act = nn.Sigmoid()

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, word_position_mask, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        #print(f"Input shape: {src.shape}")
        #print(f"Mask shape: {word_position_mask.shape}")
        src = self.encoder(src) * math.sqrt(self.embedding_dim)
        #print(f"Embedded src shape: {src.shape}")
        skip_src_masked = torch.masked_select(src, word_position_mask.unsqueeze(-1).expand(src.shape)).view(-1, 1, self.embedding_dim).squeeze(1)
        #print(f"Skip src masked shape: {skip_src_masked.shape}")
        skip_src = src
        #print(f"Skip src shape: {skip_src.shape}")
        src = self.pos_encoder(src)
        #print(f"Positional encoding shape: {src.shape}")
        output = self.transformer_encoder(src, self.src_mask)
        #print(f"Transformer encoder output shape: {output.shape}")
        output = torch.add(output, skip_src)
        #print(f"Added output shape: {output.shape}")
        output = self.batch_normer(output)
        #print(f"Mask shape: {word_position_mask.unsqueeze(-1).expand(output.shape).shape}")
        output = self.pooler(output.permute(0,2,1)).squeeze(2)
        #print(f"Pooled output shape: {output.shape}")
        output = torch.add(output, skip_src_masked)
        output = self.decoder(output)
        #print(f"Decoder output shape: {output.shape}")
        return self.decoder_act(output)

class MyAttentionModelWithPoolingAndSkipOnWordPositionTwoLayers(nn.Module):
    """My Attention model, based on the Transformer encoder."""

    def __init__(self, vocab_size, embedding_dim, max_seq_len, num_heads, dim_feedforward, num_layers, dropout=0.5, dim_hidden_decoder=8):
        super(MyAttentionModelWithPoolingAndSkipOnWordPositionTwoLayers, self).__init__()
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
        self.decoder = nn.Linear(embedding_dim, dim_hidden_decoder)
        self.decoder_2 = nn.Linear(dim_hidden_decoder, 1)
        self.batch_normer = torch.nn.BatchNorm1d(max_seq_len)
        self.pooler = nn.AvgPool1d(max_seq_len, stride=1)
        self.decoder_hidden_act = nn.Tanh()
        self.decoder_act = nn.Sigmoid()

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, word_position_mask, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        #print(f"Input shape: {src.shape}")
        #print(f"Mask shape: {word_position_mask.shape}")
        src = self.encoder(src) * math.sqrt(self.embedding_dim)
        #print(f"Embedded src shape: {src.shape}")
        skip_src_masked = torch.masked_select(src, word_position_mask.unsqueeze(-1).expand(src.shape)).view(-1, 1, self.embedding_dim).squeeze(1)
        #print(f"Skip src masked shape: {skip_src_masked.shape}")
        skip_src = src
        #print(f"Skip src shape: {skip_src.shape}")
        src = self.pos_encoder(src)
        #print(f"Positional encoding shape: {src.shape}")
        output = self.transformer_encoder(src, self.src_mask)
        #print(f"Transformer encoder output shape: {output.shape}")
        output = torch.add(output, skip_src)
        #print(f"Added output shape: {output.shape}")
        output = self.batch_normer(output)
        #print(f"Mask shape: {word_position_mask.unsqueeze(-1).expand(output.shape).shape}")
        output = self.pooler(output.permute(0,2,1)).squeeze(2)
        #print(f"Pooled output shape: {output.shape}")
        output = torch.add(output, skip_src_masked)
        output = self.decoder(output)
        output = self.decoder_hidden_act(output)
        output = self.decoder_2(output)
        #print(f"Decoder output shape: {output.shape}")
        return self.decoder_act(output)