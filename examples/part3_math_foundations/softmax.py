"""
Part 3: Softmax
================
Converts raw model outputs (logits) into a probability distribution.
"""

import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.5, -1.0, 3.0])


def softmax_manual(x):
    """Manual softmax implementation."""
    exps = torch.exp(x - x.max())  # subtract max for numerical stability
    return exps / exps.sum()


probs_manual = softmax_manual(logits)
probs_pytorch = F.softmax(logits, dim=-1)

print(f"Manual:  {probs_manual}")
print(f"PyTorch: {probs_pytorch}")
# Both produce: tensor([0.2312, 0.0850, 0.0516, 0.0115, 0.6276])
