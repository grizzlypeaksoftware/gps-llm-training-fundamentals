"""
Part 5: Attention Mechanism Deep Dive
=======================================
A standalone implementation of scaled dot-product attention for experimentation.
"""

import torch
import torch.nn.functional as F
import math


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        query: (batch, heads, seq_len, head_dim)
        key:   (batch, heads, seq_len, head_dim)
        value: (batch, heads, seq_len, head_dim)
        mask:  (1, 1, seq_len, seq_len) causal mask

    Returns:
        output: weighted sum of values
        weights: attention weights (useful for visualization)
    """
    d_k = query.size(-1)

    # Q @ K^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply causal mask (prevent looking at future tokens)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax over the key dimension
    weights = F.softmax(scores, dim=-1)

    # Weighted sum of values
    output = torch.matmul(weights, value)

    return output, weights


# Demo
seq_len, d_model, num_heads = 6, 16, 4
head_dim = d_model // num_heads

# Random input representing 6 tokens
x = torch.randn(1, num_heads, seq_len, head_dim)

# Causal mask
mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)

output, weights = scaled_dot_product_attention(x, x, x, mask)

print(f"Input shape:   {x.shape}")
print(f"Output shape:  {output.shape}")
print(f"Attention weights (head 0):\n{weights[0, 0].detach()}")
