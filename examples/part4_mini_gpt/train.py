"""
Part 4: Training Loop
======================
Trains the MiniGPT model on a small text corpus and generates sample output.

Usage:
    python train.py
"""

import torch
import time
from tokenizer import CharTokenizer
from model import MiniGPT
from dataset import create_dataloader


def train():
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    SEQ_LENGTH = 64
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 4
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 3e-4
    DROPOUT = 0.1

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    text = """
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

    tokenizer = CharTokenizer(text)
    token_ids = tokenizer.encode(text)

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Total tokens: {len(token_ids)}")

    dataloader = create_dataloader(token_ids, SEQ_LENGTH, BATCH_SIZE)
    print(f"Training batches per epoch: {len(dataloader)}")

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        seq_length=SEQ_LENGTH,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Cosine annealing learning rate schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE / 10
    )

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    print("\n--- Training ---\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        start_time = time.time()

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits, loss = model(x, targets=y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(dataloader)
        elapsed = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(
                f"Epoch {epoch:>3d}/{EPOCHS} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.2f}s"
            )

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------
    print("\n--- Generating Text ---\n")
    model.eval()

    prompts = ["Python is", "The transformer", "Training involves"]

    for prompt in prompts:
        token_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(DEVICE)

        output_ids = model.generate(input_ids, max_new_tokens=150, temperature=0.8, top_k=20)
        generated = tokenizer.decode(output_ids[0].tolist())

        print(f"Prompt: '{prompt}'")
        print(f"Output: {generated[:200]}")
        print()

    # -------------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------------
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab_size': tokenizer.vocab_size,
        'config': {
            'embed_dim': EMBED_DIM,
            'num_heads': NUM_HEADS,
            'num_layers': NUM_LAYERS,
            'seq_length': SEQ_LENGTH,
        }
    }, 'mini_gpt_checkpoint.pt')
    print("Model saved to mini_gpt_checkpoint.pt")


if __name__ == "__main__":
    train()
