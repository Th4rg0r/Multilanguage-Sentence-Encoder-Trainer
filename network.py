import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()

        # Init matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # calc sinosudial values
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)

        # Add batch dimension
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Add positional encoding
        x + self.pe[:, : x.size(1)]
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_head=8,
        num_layers=6,
        dim_feed_forward=2048,
        dropout=0.15,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.num_layers = num_layers
        self.dim_feed_forward = 2048
        self.dropout = dropout

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=self.dim_feed_forward,
            dropout=self.dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=self.num_layers,
            norm=self.layer_norm,
        )

    def forward(self, x, mask):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        return x


class MissingFinder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_head=8,
        num_layers=6,
        dim_feed_forward=2048,
        dropout=0.15,
    ):
        super().__init__()
        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_head=n_head,
            num_layers=num_layers,
            dim_feed_forward=dim_feed_forward,
            dropout=dropout,
        )
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask):
        x = self.encoder(x, mask)
        x = self.output_layer(x)
        return x
