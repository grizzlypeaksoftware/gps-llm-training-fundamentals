"""
Part 3: Cross-Entropy Loss
===========================
Training an LLM minimizes cross-entropy loss — the difference between the model's
predicted probability distribution over the vocabulary and the actual next token.
"""

import torch
import torch.nn.functional as F


def cross_entropy_loss_manual(predicted_logits, actual_token_index):
    """Manual implementation for understanding."""
    probs = F.softmax(predicted_logits, dim=-1)
    return -torch.log(probs[actual_token_index] + 1e-10)


# Example: vocabulary of 5 tokens
logits = torch.tensor([2.0, 0.5, 3.1, 0.1, -1.0])
actual_token = 2  # the correct next token was index 2

# Manual
loss_manual = cross_entropy_loss_manual(logits, actual_token)
print(f"Manual loss: {loss_manual.item():.4f}")

# PyTorch built-in (preferred — numerically stable)
loss_builtin = F.cross_entropy(logits.unsqueeze(0), torch.tensor([actual_token]))
print(f"PyTorch loss: {loss_builtin.item():.4f}")
