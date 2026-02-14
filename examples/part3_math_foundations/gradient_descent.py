"""
Part 3: Gradient Descent
=========================
The model improves by computing how much each parameter contributed to the error
(via backpropagation) and nudging it in the opposite direction.
"""

import torch

# Simple gradient descent demonstration
weights = torch.tensor([0.5, -0.3], requires_grad=True)
learning_rate = 0.01

# Simulate a loss (normally computed by the model)
loss = (weights * torch.tensor([1.0, -1.0])).sum()

# Backpropagation: compute gradients
loss.backward()

print(f"Gradients: {weights.grad}")

# Manual update step
with torch.no_grad():
    weights -= learning_rate * weights.grad
    print(f"Updated weights: {weights}")

# In practice, you use an optimizer:
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# optimizer.step()
