class TransformerDecoderOnly(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len, num_layers=6, n_heads=8):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(seq_len, embed_dim)
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, n_heads=n_heads) for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def make_mask(self, seq):
        batch_size, seq_len = seq.shape
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
        return mask.to(seq.device)

    def forward(self, x):
        mask = self.make_mask(x)
        x = self.position_embedding(self.word_embedding(x))
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc_out(self.norm_out(x))

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(idx)[:, -1, :]
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx