"""
Load a trained MiniGPT checkpoint and generate text from a prompt.

Usage:
    python generate.py                          # uses default prompt
    python generate.py "Python is"              # custom prompt
    python generate.py "The transformer" 200    # custom prompt + length
"""

import sys
import torch
from tokenizer import CharTokenizer
from model import MiniGPT

# The same training text — needed to rebuild the tokenizer vocabulary
TRAINING_TEXT = """
    Python is a high-level programming language known for its simplicity and readability.
    It supports multiple paradigms including procedural, object-oriented, and functional programming.
    PyTorch is a deep learning framework developed by Meta's AI research lab.
    It provides dynamic computational graphs and GPU acceleration for tensor operations.
    Large language models are neural networks trained to predict the next token in a sequence.
    The transformer architecture uses self-attention to process sequences in parallel.
    Training involves minimizing the cross-entropy loss between predicted and actual tokens.
    Gradient descent optimizes the model parameters by following the gradient of the loss function.
    Backpropagation computes gradients efficiently by applying the chain rule of calculus.
    The attention mechanism allows each token to attend to every other token in the sequence.
    Multi-head attention runs several attention operations in parallel for richer representations.
    Positional embeddings encode the order of tokens since transformers have no inherent sequence order.
    Layer normalization stabilizes training by normalizing activations within each layer.
    Residual connections allow gradients to flow directly through the network during backpropagation.
    The feed-forward network in each transformer block applies two linear transformations with a nonlinearity.
    Weight tying shares parameters between the token embedding and the output projection layer.
    """


def main():
    # Load checkpoint
    checkpoint = torch.load("mini_gpt_checkpoint.pt", weights_only=False)
    config = checkpoint["config"]

    # Rebuild tokenizer from the same text
    tokenizer = CharTokenizer(TRAINING_TEXT)

    # Rebuild model with saved config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniGPT(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        seq_length=config["seq_length"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Get prompt from command line or use default
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Python is"
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 150

    # Generate
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)

    output_ids = model.generate(input_ids, max_new_tokens=max_tokens, temperature=0.8, top_k=20)
    generated = tokenizer.decode(output_ids[0].tolist())

    print(f"\nPrompt: '{prompt}'")
    print(f"Output: {generated}")


if __name__ == "__main__":
    main()
