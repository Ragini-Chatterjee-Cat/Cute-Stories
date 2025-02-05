import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Positional Encoding Class
class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim):
        super().__init__()
        self.embed_dim = embed_model_dim
        pe = torch.zeros(max_seq_len, embed_model_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_model_dim, 2).float() * (-math.log(10000.0) / embed_model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

# Multi-Head Attention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = embed_dim // n_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, key, query, value, mask=None):
        batch_size, seq_length, _ = query.shape
        qkv = self.qkv_proj(query).reshape(batch_size, seq_length, 3, self.n_heads, self.single_head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.single_head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v).permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.embed_dim)
        return self.out_proj(attn_output)

# Transformer Block
class TransformerBlock(nn.Module):
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
        attn_output = self.dropout(self.attention(x, x, x, mask))
        x = self.norm1(x + attn_output)
        ff_output = self.dropout(self.feed_forward(x))
        return self.norm2(x + ff_output)