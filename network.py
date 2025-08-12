import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
     

# MPNet model class (Song et al., 2020):contentReference[oaicite:0]{index=0}: Masked & Permuted language modeling
class MPNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim=768, num_attention_heads=12, num_encoder_layers=12, max_position_embeddings=512, dropout=0.1):
        super(MPNet, self).__init__()
        self.embedding_dim = embedding_dim
        # Token and (absolute) position embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(max_position_embeddings, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        # Shared relative attention bias embedding (T5-style buckets)
        self.relative_attention_bias = nn.Embedding(32 * 2, num_attention_heads)  # using 64 buckets (32 positive, 32 negative)
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            MPNetLayer(embedding_dim, num_attention_heads, dropout) for _ in range(num_encoder_layers)
        ])
        # Final language modeling head (tie with embeddings could be done for efficiency)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [batch_size, seq_len] token indices
        attention_mask: [batch_size, seq_len] with 1 for real tokens and 0 for padding
        """
        batch_size, seq_len = input_ids.size()
        # Create position ids [0..seq_len-1] for each element in batch
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        # Embed tokens and positions
        inputs_embeds = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        hidden_states = self.layer_norm(inputs_embeds)
        hidden_states = self.dropout(hidden_states)

        # Compute relative position bias once (shared across layers)
        # Here we use T5-like bucketed relative positions for efficiency
        # Create context and memory position matrices
        context_position = position_ids.unsqueeze(-1)  # [batch, seq, 1]
        memory_position = position_ids.unsqueeze(-2)   # [batch, 1, seq]
        relative_position = memory_position - context_position  # [batch, seq, seq]
        # Bucketed relative position as in T5/MPNet
        rp_bucket = self._relative_position_bucket(relative_position, num_buckets=32)
        # [batch, seq, seq, num_attention_heads] -> combine head and bucket
        # Embed and reshape to [batch, num_attention_heads, seq, seq]
        values = self.relative_attention_bias(rp_bucket)  # [batch, seq, seq, heads]
        values = values.permute(0, 3, 1, 2)  # [batch, heads, seq, seq]

        # Apply each transformer layer
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask, position_bias=values)

        # Final layer norm (optional, as used in BERT)
        # (here layer outputs already normalized)
        sequence_output = hidden_states

        # Compute token logits for masked positions
        logits = self.lm_head(sequence_output)  # [batch, seq, vocab_size]
        return logits

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        """
        Translate relative position matrix to bucket ids.
        Follows MPNet/T5 implementation: roughly half buckets for negative, half for positive.
        """
        # relative_position: [batch, seq_len, seq_len]
        # We only need per-element calculation, so remove batch dimension
        rp = relative_position.clone().detach()
        rp = rp.clamp(-max_distance, max_distance)
        rp_bucket = torch.zeros_like(rp)
        # Negative and positive buckets
        neg_mask = rp < 0
        pos_mask = rp > 0
        num_buckets //= 2
        # Use log scale for larger distances
        if num_buckets > 0:
            # For positive distances
            rp_pos = rp[pos_mask].float()
            # Small linear region and larger log region
            max_exact = num_buckets // 2
            is_small = rp_pos < max_exact
            val_if_large = max_exact + (
                (torch.log(rp_pos / max_exact) / torch.log(torch.tensor(max_distance / max_exact))) * (num_buckets - max_exact)
            ).long()
            val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
            bucket_pos = torch.where(is_small, rp_pos.long(), val_if_large)
            # Fill buckets (positive side)
            rp_bucket[pos_mask] = bucket_pos + num_buckets
            # For negative distances
            rp_neg = (-rp[neg_mask]).float()
            is_small = rp_neg < max_exact
            val_if_large = max_exact + (
                (torch.log(rp_neg / max_exact) / torch.log(torch.tensor(max_distance / max_exact))) * (num_buckets - max_exact)
            ).long()
            val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
            bucket_neg = torch.where(is_small, rp_neg.long(), val_if_large)
            rp_bucket[neg_mask] = bucket_neg
        return rp_bucket.long()

class MPNetLayer(nn.Module):
    """Single Transformer encoder layer with MPNet (relative-bias) attention."""
    def __init__(self, embedding_dim, num_attention_heads, dropout):
        super(MPNetLayer, self).__init__()
        self.self_attn = MPNetSelfAttention(embedding_dim, num_attention_heads)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        # Feed-forward network (intermediate size usually 4*hidden)
        self.fc1 = nn.Linear(embedding_dim, embedding_dim * 4)
        self.fc2 = nn.Linear(embedding_dim * 4, embedding_dim)

    def forward(self, hidden_states, attention_mask=None, position_bias=None):
        # Self-attention with (optional) relative positional bias
        attn_output = self.self_attn(hidden_states, attention_mask=attention_mask, position_bias=position_bias)
        attn_output = self.dropout(attn_output)
        # First residual + layer norm
        hidden_states = self.layer_norm1(hidden_states + attn_output)
        # Feed-forward (GELU activation)
        ffn_output = F.gelu(self.fc1(hidden_states))
        ffn_output = self.dropout(self.fc2(ffn_output))
        # Second residual + layer norm
        hidden_states = self.layer_norm2(hidden_states + ffn_output)
        return hidden_states

class MPNetSelfAttention(nn.Module):
    """Multi-head self-attention with support for adding a precomputed position bias."""
    def __init__(self, embedding_dim, num_attention_heads):
        super(MPNetSelfAttention, self).__init__()
        if embedding_dim % num_attention_heads != 0:
            raise ValueError("Hidden size must be divisible by num_attention_heads.")
        self.num_attention_heads = num_attention_heads
        self.head_size = embedding_dim // num_attention_heads
        self.all_head_size = embedding_dim
        # Query, Key, Value projections
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key   = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.out   = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, attention_mask=None, position_bias=None):
        # hidden_states: [batch, seq_len, embedding_dim]
        batch_size, seq_len, _ = hidden_states.size()
        # Linear projections
        query_layer = self.query(hidden_states)  # [batch, seq, hidden]
        key_layer   = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        # Reshape for multi-head: [batch, num_attention_heads, seq_len, head_size]
        def reshape_for_heads(x):
            return x.view(batch_size, seq_len, self.num_attention_heads, self.head_size).permute(0, 2, 1, 3)
        query_layer = reshape_for_heads(query_layer)
        key_layer   = reshape_for_heads(key_layer)
        value_layer = reshape_for_heads(value_layer)
        # Compute scaled dot-product attention scores
        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [batch, heads, seq, seq]
        scores = scores / (self.head_size ** 0.5)
        # Add relative position bias if provided
        if position_bias is not None:
            scores = scores + position_bias.unsqueeze(0)  # broadcast over batch
        # Apply attention mask (if any)
        if attention_mask is not None:
            # attention_mask: [batch, seq] -> [batch, 1, 1, seq] with 0 or -inf
            scores = scores + attention_mask.unsqueeze(1).unsqueeze(2)
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        # Attention output
        context = torch.matmul(attn_probs, value_layer)  # [batch, heads, seq, head_size]
        # Reshape back to [batch, seq, hidden]
        context = context.squeeze().permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.all_head_size)
        # Final linear layer
        attn_output = self.out(context)
        return attn_output


class SentenceEncoder(nn.Module):
    """
    A final, production-ready sentence encoder model.
    Wraps a pre-trained Encoder and a pooling layer to directly output
    a single sentence embedding.
    """
    def __init__(self, encoder: MPNet):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        """
        Takes tokenized input and returns a single sentence embedding.
        """
        token_embeddings = self.encoder(input_ids, attention_mask)
        sentence_embedding = mean_pooling(token_embeddings, attention_mask)
        return sentence_embedding
