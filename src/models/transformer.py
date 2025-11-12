"""
Transformer model components including positional embeddings, attention, and decoder blocks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEmbedding(nn.Module):
    """
    Positional Embedding layer that adds positional information to input embeddings.

    Args:
        max_seq_len (int): Maximum sequence length
        embed_model_dim (int): Dimension of embedding model
    """
    def __init__(self, max_seq_len, embed_model_dim):
        super().__init__()
        self.embed_dim = embed_model_dim
        pe = torch.zeros(max_seq_len, embed_model_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_model_dim, 2).float() *
            (-math.log(10000.0) / embed_model_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """Add positional encoding to input."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Args:
        embed_dim (int): Embedding dimension
        n_heads (int): Number of attention heads
    """
    def __init__(self, embed_dim=512, n_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = embed_dim // n_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, key, query, value, mask=None):
        """
        Apply multi-head attention.

        Args:
            key: Key tensor
            query: Query tensor
            value: Value tensor
            mask: Optional attention mask

        Returns:
            Attention output tensor
        """
        batch_size, seq_length, _ = query.shape
        qkv = self.qkv_proj(query).reshape(
            batch_size, seq_length, 3, self.n_heads, self.single_head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.single_head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = (
            torch.matmul(attn_weights, v)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, seq_length, self.embed_dim)
        )
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed-forward layers.

    Args:
        embed_dim (int): Embedding dimension
        expansion_factor (int): Expansion factor for feed-forward layer
        n_heads (int): Number of attention heads
    """
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim)
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, mask=None):
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor
            mask: Optional attention mask

        Returns:
            Output tensor
        """
        attn_output = self.dropout(self.attention(x, x, x, mask))
        x = self.norm1(x + attn_output)
        ff_output = self.dropout(self.feed_forward(x))
        return self.norm2(x + ff_output)


class TransformerDecoderOnly(nn.Module):
    """
    Decoder-only Transformer model for text generation.

    Args:
        vocab_size (int): Size of vocabulary
        embed_dim (int): Embedding dimension
        seq_len (int): Sequence length
        num_layers (int): Number of transformer layers
        n_heads (int): Number of attention heads
    """
    def __init__(self, vocab_size, embed_dim, seq_len, num_layers=6, n_heads=8):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(seq_len, embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads=n_heads)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def make_mask(self, seq):
        """
        Create causal mask for autoregressive generation.

        Args:
            seq: Input sequence tensor

        Returns:
            Causal mask tensor
        """
        batch_size, seq_len = seq.shape
        mask = (
            torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, 1, seq_len, seq_len)
        )
        return mask.to(seq.device)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Input token IDs

        Returns:
            Logits for next token prediction
        """
        mask = self.make_mask(x)
        x = self.position_embedding(self.word_embedding(x))
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc_out(self.norm_out(x))

    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens autoregressively.

        Args:
            idx: Starting token IDs
            max_new_tokens: Number of tokens to generate

        Returns:
            Generated sequence of token IDs
        """
        for _ in range(max_new_tokens):
            logits = self(idx)[:, -1, :]
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
