import torch
import torch.nn as nn
import math

def mean_pooling(model_output, attention_mask):
    """
    Performs mean pooling on the token embeddings, ignoring padding tokens.
    
    Args:
        model_output (torch.Tensor): The output of the transformer model.
                                     Shape: (batch_size, sequence_length, embedding_dim)
        attention_mask (torch.Tensor) : The attention mask from the tokenizer.
                                        Shape: (batch_size, sequence_length)
    Returns:
        torch.Tensor: The Sentence embeddings.
                      Shape: (batch_size, embedding_dim)
    """
    # Expand the attention mask,  to match the shape of  the token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()

    # sum the embeddings but only for non-padding tokens (where mask is  1)
    sum_embeddings =  torch.sum(model_output * input_mask_expanded, dim=1)

    # Sum the mask, to get the actual length of each sentence (ignoring padding)
    # Add a small elipson (1e-9) to avoid division by zero for empty sequences
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    #Divide the sum of embeddings by the actual length to get the mean
    return sum_embeddings / sum_mask
    
     

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=10000):
        super(PositionalEncoding, self).__init__()

        # Init matrix
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # calc sinosudial values
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
        )

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, embedding_dim)

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
        embedding_dim=512,
        num_attention_heads=8,
        num_encoder_layers=6,
        feed_forward_dim=2048,
        dropout=0.15,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.num_encoder_layers = num_encoder_layers
        self.feed_forward_dim = 2048
        self.dropout = dropout

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.pos_encoder = PositionalEncoding(self.embedding_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_attention_heads,
            dim_feedforward=self.feed_forward_dim,
            dropout=self.dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=self.num_encoder_layers,
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
        embedding_dim=512,
        num_attention_heads=8,
        num_encoder_layers=6,
        feed_forward_dim=2048,
        dropout=0.15,
    ):
        super().__init__()
        self.encoder = Encoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_attention_heads=num_attention_heads,
            num_encoder_layers=num_encoder_layers,
            feed_forward_dim=feed_forward_dim,
            dropout=dropout,
        )
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, mask):
        x = self.encoder(x, mask)
        x = self.output_layer(x)
        return x


class SentenceEncoder(nn.Module):
    """
    A final, production-ready sentence encoder model.
    Wraps a pre-trained Encoder and a pooling layer to directly output
    a single sentence embedding.
    """
    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        """
        Takes tokenized input and returns a single sentence embedding.
        """
        token_embeddings = self.encoder(input_ids, attention_mask)
        sentence_embedding = mean_pooling(token_embeddings, attention_mask)
        return sentence_embedding
