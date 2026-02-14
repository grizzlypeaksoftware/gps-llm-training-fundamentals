"""
Part 4: MiniGPT Model
=======================
A minimal but complete GPT-style transformer. It uses real multi-head
self-attention, positional embeddings, and layer normalization — the same
architecture (in miniature) as production LLMs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, embed_dim, num_heads, seq_length, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: prevent attending to future tokens
        mask = torch.tril(torch.ones(seq_length, seq_length))
        self.register_buffer("mask", mask.view(1, 1, seq_length, seq_length))

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # Compute Q, K, V in one projection
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention: (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # Weighted sum of values
        out = weights @ v  # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.out_proj(out))


class TransformerBlock(nn.Module):
    """A single transformer block: attention + feed-forward with residuals."""

    def __init__(self, embed_dim, num_heads, seq_length, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, seq_length, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # residual connection
        x = x + self.ffn(self.ln2(x))   # residual connection
        return x


class MiniGPT(nn.Module):
    """A minimal GPT-style language model."""

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, seq_length, dropout=0.1):
        super().__init__()
        self.seq_length = seq_length

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(seq_length, embed_dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, seq_length, dropout)
            for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: share token embedding and output projection
        self.head.weight = self.token_embed.weight

        # Initialize weights
        self.apply(self._init_weights)
        print(f"Model parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.seq_length, f"Sequence length {T} exceeds max {self.seq_length}"

        # Token + positional embeddings
        tok_emb = self.token_embed(idx)
        pos_emb = self.pos_embed(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        x = self.blocks(x)
        x = self.ln_final(x)

        # Project to vocabulary
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        """Autoregressive generation with temperature and top-k sampling."""
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            idx_cond = idx[:, -self.seq_length:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx
