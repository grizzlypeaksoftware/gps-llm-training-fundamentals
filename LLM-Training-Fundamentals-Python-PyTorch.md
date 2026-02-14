# The Fundamentals of Training an LLM — A Python & PyTorch Guide

## Introduction

Large Language Models (LLMs) like GPT, Claude, and LLaMA have transformed software development, but the process of actually *training* one remains a black box for most developers. You interact with these models every day — asking them to write code, summarize documents, or answer questions — but what's actually happening under the hood? How does a pile of random numbers turn into something that can write poetry or debug your Python script?

This guide breaks down the core concepts behind LLM training and walks you through each stage using Python and PyTorch — the industry-standard stack for building and training neural networks. We start with the theory, move into working code you can run on your own machine, and finish with a complete picture of how the entire pipeline fits together.

Python with PyTorch is the dominant choice for LLM training at every scale, from toy models on a laptop to frontier models on thousand-GPU clusters. This guide covers both the theory and practical implementations you can run today. No PhD required — just familiarity with Python and a willingness to think in matrices.

All code examples are available as runnable scripts in the [companion repository](https://github.com/grizzlypeaksoftware/gps-llm-training-fundamentals).

---

## Part 1: What Is an LLM, Really?

At its core, an LLM is a neural network — specifically a **transformer** — trained to predict the next token (word or sub-word) in a sequence. That's it. Everything impressive about LLMs flows from this deceptively simple objective.

Think about what "predict the next word" actually requires. To predict what comes after "The capital of France is", the model needs to have learned geography. To predict the next token in a Python function, it needs to have learned programming syntax and logic. By training on enough text, the model implicitly learns an enormous amount about the world — all through this one objective.

### Key Components

**Tokens:** LLMs don't see words the way you do. They see numerical IDs representing sub-word chunks. The sentence "Python is awesome" might become tokens like `[31380, 374, 12738]` (using GPT-4's tokenizer). This process of converting text to numbers is called *tokenization*, and it's the very first step in any LLM pipeline. Common words get their own token, while rare words get split into smaller pieces — that's why you'll sometimes see a model "think" about a word one syllable at a time.

**Embeddings:** Each token ID maps to a high-dimensional vector (e.g., 768 or 4096 numbers). These vectors capture semantic meaning — similar words end up near each other in this vector space. For example, the vectors for "king" and "queen" will be closer together than "king" and "bicycle." The model doesn't start with this knowledge — it *learns* these meaningful representations during training. Initially, embeddings are just random numbers.

**Transformer Architecture:** The model processes token sequences through layers of *self-attention* (which tokens should pay attention to which other tokens?) and *feed-forward networks* (what transformations should be applied?). Self-attention is what makes transformers special — it allows every token in a sequence to directly interact with every other token, regardless of how far apart they are. This is a major advantage over older architectures (like RNNs) that had to process tokens one at a time and struggled with long-range dependencies.

**Parameters (Weights):** The millions or billions of numbers inside the model that get adjusted during training. When someone says "a 7B model," they mean 7 billion trainable parameters. These parameters are organized into matrices that perform the attention and feed-forward computations. Before training, they're random noise. After training, they encode everything the model has learned about language. The entire goal of training is to find the right values for these parameters.

---

## Part 2: The Three Phases of LLM Training

Training an LLM isn't a single step — it's a pipeline with distinct phases, each with different goals, data requirements, and costs. Understanding these phases is key to understanding why LLMs behave the way they do.

### Phase 1: Pre-training

This is the most expensive and foundational phase. The model learns language itself from massive text corpora (books, websites, code, etc.). During pre-training, the model sees enormous amounts of raw text and learns the statistical patterns of language — grammar, facts, reasoning patterns, coding conventions, and much more.

**Objective:** Given a sequence of tokens, predict the next one (causal language modeling). The model reads "The cat sat on the" and tries to predict "mat." It does this billions of times across the entire training corpus, gradually getting better at predicting what comes next in any context.

**Scale:** Trillions of tokens, thousands of GPUs, weeks to months of compute, millions of dollars. This is why most practitioners don't pre-train their own models — they start from one that someone else has already pre-trained.

**What it produces:** A "base model" that can complete text but isn't yet useful for conversations or instructions. If you prompt a base model with "What is 2+2?", it might continue with "What is 3+3? What is 4+4?" because it learned that questions often come in sequences. It hasn't yet learned to *answer* questions — just to predict text.

### Phase 2: Fine-Tuning

The pre-trained model is adapted to specific tasks or behaviors using curated, higher-quality datasets. This is where the model goes from "can complete text" to "can follow instructions and be useful."

**Supervised Fine-Tuning (SFT):** Train on prompt/response pairs to teach the model to follow instructions. The training data looks like conversations: a user asks a question, and there's a high-quality answer. After seeing thousands of these examples, the model learns the pattern of "when given a question, provide a helpful answer" rather than "continue generating more questions."

**Domain Fine-Tuning:** Train on specialized data (legal documents, medical text, code) to improve performance in a specific area. A model fine-tuned on medical literature will give much better answers about diagnoses than a general-purpose model, even if the base model saw some medical text during pre-training.

**Scale:** Thousands to millions of examples, hours to days on a few GPUs. This is dramatically cheaper than pre-training, which is why fine-tuning is the most common way for developers and companies to customize LLMs for their needs.

### Phase 3: Alignment (RLHF / DPO)

The model is further refined to be helpful, harmless, and honest using human feedback. Fine-tuning teaches the model *what* to do, but alignment teaches it *how well* to do it — and what to avoid.

**RLHF (Reinforcement Learning from Human Feedback):** Humans rank model outputs from best to worst. A separate "reward model" is trained on those rankings to predict which outputs humans prefer. The LLM is then optimized to produce outputs that score highly according to the reward model. This is how models learn nuanced behaviors like being polite, admitting uncertainty, and refusing harmful requests.

**DPO (Direct Preference Optimization):** A simpler alternative that skips the reward model and directly optimizes from preference pairs. Instead of training a separate reward model, DPO reformulates the problem so the LLM can learn directly from "this response is better than that response" comparisons. It's becoming increasingly popular because it's simpler to implement and often produces comparable results.

---

## Part 3: The Math You Need to Know

You don't need a math degree to understand LLM training, but three concepts come up repeatedly. Let's demystify each one with code you can run.

### Loss Function

Training an LLM minimizes **cross-entropy loss** — the difference between the model's predicted probability distribution over the vocabulary and the actual next token. In plain English: the model outputs a probability for every possible next token, and the loss measures how much probability it assigned to the *correct* token. If the model was confident about the right answer (high probability), the loss is low. If it was wrong or uncertain, the loss is high.

The entire training process is about making this number go down.

```python
import torch
import torch.nn.functional as F

# Cross-entropy loss for a single prediction
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
```

### Gradient Descent

Once we know how wrong the model is (the loss), we need to figure out *how to fix it*. Gradient descent is the algorithm that does this. It computes how much each parameter contributed to the error (via **backpropagation**) and nudges it in the opposite direction.

Imagine you're standing on a hilly landscape in the dark and you want to find the lowest valley. You can feel the slope under your feet — that's the gradient. You take a step downhill — that's gradient descent. The size of your step is the *learning rate*. Too big and you overshoot; too small and you'll take forever.

Backpropagation is the clever mathematical trick (using the chain rule of calculus) that lets us efficiently compute the gradient for every single parameter in the model, even when there are billions of them.

```python
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
```

### Softmax

Converts raw model outputs (logits) into a probability distribution. The model's final layer outputs raw scores (logits) for every token in the vocabulary — these can be any real number, positive or negative. Softmax squashes them into values between 0 and 1 that sum to 1, turning them into proper probabilities. Higher logits become higher probabilities, but even the lowest logit gets a small (non-zero) probability.

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.5, -1.0, 3.0])

# Manual implementation
def softmax_manual(x):
    exps = torch.exp(x - x.max())  # subtract max for numerical stability
    return exps / exps.sum()

probs_manual = softmax_manual(logits)
probs_pytorch = F.softmax(logits, dim=-1)

print(f"Manual:  {probs_manual}")
print(f"PyTorch: {probs_pytorch}")
# Both produce: tensor([0.2294, 0.0844, 0.0512, 0.0114, 0.6236])
```

---

## Part 4: Building a GPT From Scratch in PyTorch

Now we put theory into practice. This is a minimal but complete GPT-style transformer. It uses real multi-head self-attention, positional embeddings, and layer normalization — the same architecture (in miniature) as production LLMs.

Our model will have around 800,000 parameters (compared to billions in production models), but the architecture is identical. If you understand how this model works, you understand the core of GPT-2, LLaMA, and every other decoder-only transformer.

### Setup

```bash
pip install torch numpy
```

### Tokenizer (Character-Level)

For simplicity, our tokenizer works at the character level — each character (letter, space, punctuation) becomes its own token. This means our vocabulary is tiny (around 40-50 unique characters), which makes training fast. Production LLMs use sub-word tokenizers like BPE (covered in Part 6) with vocabularies of 32,000-100,000+ tokens.

```python
# tokenizer.py

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids):
        return ''.join(self.id_to_char[i] for i in ids)
```

### The Full Transformer Model

This is the heart of the implementation. It's split into three classes that build on each other:

- **`CausalSelfAttention`** — The attention mechanism. Each token creates a query ("what am I looking for?"), a key ("what do I contain?"), and a value ("what information do I provide?"). Attention scores are computed between all query-key pairs, and the results are used to create a weighted combination of values. The "causal" mask ensures tokens can only attend to earlier tokens in the sequence — the model can't cheat by looking ahead.

- **`TransformerBlock`** — One complete transformer layer. It applies attention, then a feed-forward network, with layer normalization and residual connections wrapping each step. Residual connections (the `x = x + ...` pattern) are critical — they allow gradients to flow directly through the network during training, which is what makes training deep networks possible.

- **`MiniGPT`** — The full model. It converts token IDs to embeddings, adds positional information (so the model knows token order), passes everything through a stack of transformer blocks, and projects the result back to vocabulary-sized logits for next-token prediction.

```python
# model.py

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
```

### Dataset and DataLoader

The dataset class creates training examples using a sliding window approach. For a sequence length of 64, it takes 64 consecutive tokens as the input (`x`) and the next 64 tokens (shifted by one position) as the target (`y`). The model learns to predict each token from the tokens before it.

The `drop_last=True` in the DataLoader discards the last incomplete batch, which prevents issues with varying batch sizes during training.

```python
# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """Sliding window over tokenized text (stride=1) for next-token prediction."""

    def __init__(self, token_ids, seq_length):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.token_ids) - self.seq_length

    def __getitem__(self, idx):
        x = self.token_ids[idx : idx + self.seq_length]
        y = self.token_ids[idx + 1 : idx + self.seq_length + 1]
        return x, y


def create_dataloader(token_ids, seq_length, batch_size, shuffle=True):
    dataset = TextDataset(token_ids, seq_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
```

### The Training Loop

The training loop is where everything comes together. For each batch, the model makes predictions, we compute the loss, calculate gradients via backpropagation, and update the weights. A few important details:

- **`optimizer.zero_grad()`** clears the gradients from the previous step (PyTorch accumulates gradients by default).
- **`loss.backward()`** computes gradients for every parameter via backpropagation.
- **`clip_grad_norm_`** prevents exploding gradients — a common problem in deep networks where gradients can become astronomically large and destabilize training.
- **`scheduler.step()`** adjusts the learning rate over time using cosine annealing, which starts high and gradually decays. This helps the model make big adjustments early in training and fine-grained adjustments later.

```python
# train.py

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
```

---

## Part 5: Attention Mechanism Deep Dive

Attention is the single most important innovation in the transformer architecture, and it's worth understanding in isolation. The self-attention implementation in Part 4 is production-style (optimized for efficiency). Here's a standalone version that prioritizes clarity for experimentation.

The core idea is simple: for each token in the sequence, attention computes a weighted average of all other tokens' representations. The weights are determined by how "relevant" each token is to the current one. The term "self-attention" means the tokens attend to each other within the same sequence (as opposed to cross-attention, where tokens from one sequence attend to tokens from another).

Here's how the math works step by step:

1. Each token produces three vectors: a **Query** (Q), a **Key** (K), and a **Value** (V).
2. The attention score between two tokens is the dot product of one token's Query with another's Key, divided by the square root of the dimension (for numerical stability).
3. These scores are passed through softmax to become weights that sum to 1.
4. The output for each token is the weighted sum of all Value vectors, using those weights.

The "multi-head" part means we run several independent attention operations in parallel (each with its own Q, K, V projections), then concatenate the results. Different heads can learn to attend to different types of relationships — one head might focus on syntactic relationships, another on semantic ones.

```python
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
```

---

## Part 6: Tokenization with BPE

In Part 4, we used a character-level tokenizer for simplicity. Production LLMs use **Byte Pair Encoding (BPE)**, which is much more efficient. BPE finds the optimal middle ground between character-level (tiny vocabulary, long sequences) and word-level (huge vocabulary, short sequences) tokenization.

The algorithm is surprisingly intuitive: start with individual characters, find the most frequently occurring pair of adjacent tokens, merge them into a new token, and repeat. After thousands of merges, common words like "the" become single tokens, while rare words get split into sub-word pieces. This means the tokenizer can handle any input text — even words it's never seen before — by falling back to smaller pieces.

### BPE From Scratch

This implementation shows the core algorithm. Watch how it discovers common patterns — "th" + "e" merge into "the", then "the" + " " merge because "the " appears so often in English text.

```python
from collections import Counter


def learn_bpe(text, num_merges):
    """Learn BPE merge rules from text."""
    # Start with character-level tokens
    tokens = list(text)
    merge_rules = []

    for i in range(num_merges):
        # Count adjacent pairs
        pairs = Counter()
        for j in range(len(tokens) - 1):
            pairs[(tokens[j], tokens[j + 1])] += 1

        if not pairs:
            break

        # Find the most frequent pair
        best_pair = pairs.most_common(1)[0]
        (left, right), count = best_pair
        merged = left + right
        merge_rules.append((left, right, merged, count))

        print(f"Merge {i+1}: '{left}' + '{right}' → '{merged}' ({count} occurrences)")

        # Apply the merge
        new_tokens = []
        j = 0
        while j < len(tokens):
            if j < len(tokens) - 1 and tokens[j] == left and tokens[j + 1] == right:
                new_tokens.append(merged)
                j += 2
            else:
                new_tokens.append(tokens[j])
                j += 1
        tokens = new_tokens

    return merge_rules, tokens


text = "the cat sat on the mat the cat the cat sat"
rules, final_tokens = learn_bpe(text, num_merges=10)
print(f"\nFinal tokens: {final_tokens}")
```

### Using the `tiktoken` Library (OpenAI's Tokenizer)

In practice, you don't implement BPE yourself — you use a battle-tested library. `tiktoken` is OpenAI's fast tokenizer implementation. The example below shows how a real tokenizer breaks text into sub-word tokens. Notice how common words get their own token, and the leading space is often included as part of the token (e.g., `" language"` not `"language"`).

```bash
pip install tiktoken
```

```python
import tiktoken

# GPT-4 uses the cl100k_base encoding — pinning to the encoding for stability
enc = tiktoken.get_encoding("cl100k_base")

text = "Large language models predict the next token."
tokens = enc.encode(text)

print(f"Text: {text}")
print(f"Token IDs: {tokens}")
print(f"Token count: {len(tokens)}")
print(f"Decoded tokens: {[enc.decode([t]) for t in tokens]}")

# Output:
# Text: Large language models predict the next token.
# Token IDs: [35353, 4221, 4211, 7168, 279, 1828, 4037, 13]
# Token count: 8
# Decoded tokens: ['Large', ' language', ' models', ' predict', ' the', ' next', ' token', '.']
```

---

## Part 7: Fine-Tuning with Hugging Face

Building a model from scratch (Part 4) is great for learning, but in practice, you'll usually start from a pre-trained model and fine-tune it for your use case. This is dramatically more efficient — someone else has already spent millions of dollars on pre-training, and you can adapt the result to your needs in hours for a few dollars.

For fine-tuning a real pre-trained model, Hugging Face's `transformers` and `trl` libraries are the standard tools. The example below uses **LoRA (Low-Rank Adaptation)**, which is the most popular fine-tuning technique because it lets you train a tiny fraction of the model's parameters (often less than 1%) while achieving results comparable to full fine-tuning.

LoRA works by freezing all the original model weights and injecting small trainable matrices into the attention layers. Instead of updating a 4096x4096 weight matrix (16 million parameters), LoRA decomposes the update into two small matrices — for example, 4096x16 and 16x4096 (131,000 parameters). This is where the "low-rank" in the name comes from.

### LoRA Fine-Tuning (Parameter-Efficient)

```bash
pip install transformers datasets peft trl accelerate bitsandbytes
```

```python
# finetune_lora.py

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


def main():
    # -------------------------------------------------------------------------
    # Load base model
    # -------------------------------------------------------------------------
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # small enough for a single GPU

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    print(f"Base model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -------------------------------------------------------------------------
    # LoRA configuration — only train a small number of adapter weights
    # -------------------------------------------------------------------------
    lora_config = LoraConfig(
        r=16,                       # rank of the low-rank matrices
        lora_alpha=32,              # scaling factor
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}%)")

    # -------------------------------------------------------------------------
    # Training data (replace with your own dataset)
    # -------------------------------------------------------------------------
    training_data = [
        {"text": "<|user|>\nWhat is gradient descent?\n<|assistant|>\nGradient descent is an optimization algorithm that iteratively adjusts parameters by moving in the direction of steepest decrease of the loss function. The learning rate controls the step size."},
        {"text": "<|user|>\nExplain backpropagation.\n<|assistant|>\nBackpropagation computes gradients of the loss with respect to each parameter by applying the chain rule of calculus backwards through the network, from output layer to input layer."},
        {"text": "<|user|>\nWhat is a transformer?\n<|assistant|>\nA transformer is a neural network architecture that uses self-attention mechanisms to process sequences in parallel. It consists of encoder and/or decoder blocks with multi-head attention and feed-forward layers."},
        # Add hundreds or thousands more examples for real fine-tuning...
    ]

    dataset = Dataset.from_list(training_data)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir="./lora_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,  # set to False if not using NVIDIA GPU; use bf16=True for Ampere+ GPUs
        optim="adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    # Save the LoRA adapter
    model.save_pretrained("./lora_adapter")
    print("LoRA adapter saved to ./lora_adapter")


if __name__ == "__main__":
    main()
```

### Loading and Using the Fine-Tuned Model

One of LoRA's best features is that the adapter is tiny — often just a few megabytes — compared to the multi-gigabyte base model. You can share adapters easily, and load different adapters onto the same base model for different tasks.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model + LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = PeftModel.from_pretrained(base_model, "./lora_adapter")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Generate
prompt = "<|user|>\nWhat is attention in transformers?\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Part 8: Key Concepts and Vocabulary

This section serves as a reference for the terminology and techniques you'll encounter when reading LLM papers, blog posts, and documentation. Bookmark this and come back to it as needed.

### Hyperparameters

Hyperparameters are the settings you choose *before* training begins — they're not learned by the model, but they heavily influence how well training goes. Choosing good hyperparameters is part science, part art, and part trial-and-error.

| Parameter | What It Controls | Typical Values |
|---|---|---|
| Learning Rate | How big each update step is. Too high = unstable training. Too low = slow convergence. | 1e-5 to 3e-4 |
| Batch Size | Samples per gradient update. Larger = smoother gradients but more memory. | 4 to 2048 (with gradient accumulation) |
| Epochs | Passes through the data. More epochs = more chances to learn, but risk overfitting. | 1–5 for fine-tuning, 1 for pre-training |
| Context Length | Max tokens the model sees at once. Longer = more context but quadratic memory cost. | 512 to 128k+ |
| Temperature | Randomness during generation. 0 = always pick the top token. Higher = more creative/random. | 0.0–2.0 |
| Dropout | Regularization (randomly zeroing neurons). Prevents overfitting by forcing redundancy. | 0.1–0.3 |
| Warmup Steps | Gradually increase LR at start. Prevents early instability when weights are still random. | 100–2000 |
| Weight Decay | L2 regularization on parameters. Keeps weights small to prevent overfitting. | 0.01–0.1 |
| Gradient Clipping | Cap gradient magnitude. Prevents exploding gradients from destabilizing training. | 1.0 (max norm) |

### Training Techniques

These are the practical techniques that make large-scale training feasible. Without them, training a billion-parameter model would require impossibly large GPUs or take impossibly long.

**Mixed Precision (FP16/BF16):** Train with 16-bit floats instead of 32-bit. Halves memory usage, speeds up training on modern GPUs (which have specialized hardware for 16-bit math), with minimal quality loss. BF16 (bfloat16) is preferred when available because it has a larger exponent range than FP16, making it less prone to overflow.

**Gradient Accumulation:** Simulate larger batch sizes by accumulating gradients over multiple forward passes before updating. If your GPU can only fit a batch of 4, but you want an effective batch size of 32, you accumulate gradients over 8 steps before calling `optimizer.step()`. This gives you the benefits of large-batch training without needing the memory.

**Gradient Checkpointing:** Trade compute for memory by recomputing activations during backprop instead of storing them. During the forward pass, a normal model stores every intermediate result (activation) so it can compute gradients later. Gradient checkpointing discards most of these and recomputes them when needed, reducing memory usage by 60-70% at the cost of about 30% more compute time.

**LoRA (Low-Rank Adaptation):** Fine-tune by adding small trainable matrices to frozen model weights. Reduces trainable parameters by 99%+. See Part 7 for a hands-on example.

**QLoRA:** Combine 4-bit quantization with LoRA for fine-tuning large models on consumer GPUs. The base model weights are compressed to 4 bits (reducing memory by ~8x), while the LoRA adapters are trained in higher precision. This lets you fine-tune a 70B model on a single 48GB GPU.

**DeepSpeed / FSDP:** Distributed training frameworks that shard model parameters, gradients, and optimizer states across multiple GPUs. When a model is too large to fit on a single GPU, these frameworks split it across many GPUs and handle the communication between them. FSDP is built into PyTorch; DeepSpeed is a separate library from Microsoft with additional optimizations.

---

## Part 9: Python Libraries and Tools for LLM Training

The LLM ecosystem is large, but you don't need to learn everything. Here's a practical map of the tools you'll encounter, organized by what they do and when you'd use them.

| Library | Use Case | Scale |
|---|---|---|
| **PyTorch** | Core training framework — tensors, autograd, neural network modules. Everything else builds on top of this. | All scales |
| **Hugging Face Transformers** | Pre-trained models, tokenizers, and training utilities. The "hub" for sharing and downloading models. | Fine-tuning to medium pre-training |
| **PEFT** | Parameter-efficient fine-tuning (LoRA, QLoRA, etc.). Use when you want to fine-tune without modifying all model weights. | Fine-tuning |
| **TRL** | RLHF, DPO, SFT trainers. Specialized training loops for alignment and instruction tuning. | Alignment |
| **DeepSpeed** | Distributed training, ZeRO optimization. Use when your model doesn't fit on a single GPU. | Large-scale pre-training |
| **FSDP (PyTorch)** | Built-in distributed training. Similar to DeepSpeed but integrated directly into PyTorch. | Large-scale pre-training |
| **vLLM** | Fast inference serving. Optimized for serving models in production with high throughput. | Production inference |
| **Axolotl** | Opinionated fine-tuning framework. Simplifies the fine-tuning workflow with YAML config files. | Fine-tuning |
| **LitGPT** | Clean GPT implementations. Great for learning and research — readable code without heavy abstractions. | Learning and research |
| **tiktoken / sentencepiece** | Tokenization. Convert text to token IDs and back. | Data preparation |
| **Weights & Biases** | Experiment tracking. Log metrics, visualize training curves, compare runs. Essential for any serious training. | All scales |

---

## Part 10: Cost and Compute Reference

One of the most common questions about LLM training is "how much does it cost?" The answer varies enormously depending on scale. Here's a rough guide to set expectations.

| Scale | Parameters | Training Cost | Hardware | Time |
|---|---|---|---|---|
| Toy model | 1K–1M | Free | CPU | Minutes |
| Small model | 10M–100M | $10–$100 | Single GPU | Hours |
| Medium model | 1B–10B | $10K–$500K | 8–64 GPUs | Days–weeks |
| Large model | 70B | $1M–$10M | 256–1024 GPUs | Weeks |
| Frontier model | 400B+ | $50M–$500M+ | Thousands of GPUs | Months |

**Fine-tuning costs are dramatically lower:** LoRA fine-tuning a 7B model can cost as little as $5–$50 on cloud GPUs, or run on a single consumer GPU with 24GB VRAM (RTX 4090, A5000). This is why fine-tuning has democratized LLM customization — you don't need a massive budget to adapt a model to your domain.

The MiniGPT model in Part 4 of this guide falls into the "toy model" category — it trains in a couple of minutes on a CPU with no special hardware required. This makes it ideal for experimenting and building intuition before scaling up.

---

## Part 11: End-to-End Workflow Summary

Here's how all the pieces fit together in a complete LLM development pipeline, from raw data to deployed model. Each step corresponds to concepts covered earlier in this guide.

```
1. DATA COLLECTION
   └── Scrape, license, or generate text data
       └── Clean, deduplicate, filter for quality

2. TOKENIZATION
   └── Train a BPE tokenizer on your corpus (or use an existing one)
       └── Convert all text to token ID sequences

3. PRE-TRAINING
   └── Initialize transformer weights randomly
       └── Train with next-token prediction (causal LM objective)
           └── Monitor loss, adjust learning rate schedule
               └── Save checkpoints regularly

4. FINE-TUNING
   └── Prepare instruction/response pairs
       └── Apply LoRA or full fine-tuning
           └── Train for 1-5 epochs on curated data

5. ALIGNMENT
   └── Collect human preference data (chosen vs rejected outputs)
       └── Train with DPO or RLHF
           └── Evaluate on safety and helpfulness benchmarks

6. EVALUATION
   └── Perplexity on held-out data
       └── Task-specific benchmarks (MMLU, HumanEval, etc.)
           └── Human evaluation for quality and safety

7. DEPLOYMENT
   └── Quantize model (INT8, INT4) for efficient inference
       └── Serve with vLLM, TGI, or similar
           └── Monitor performance and collect feedback
```

---

## Conclusion

Python with PyTorch is the definitive stack for LLM training. The fundamentals — tokenization, embeddings, attention, loss functions, gradient descent — are all grounded in straightforward mathematical operations that PyTorch makes accessible and GPU-accelerated.

The MiniGPT implementation in this guide uses the same architectural components as GPT-2, LLaMA, and other open-source LLMs — just at a smaller scale. Scaling up is primarily an engineering challenge of distributed compute, data pipelines, and training stability, not a fundamental change in approach. The concepts you've learned here apply directly, whether you're training a toy model on your laptop or orchestrating a thousand-GPU cluster.

**Recommended Next Steps:**

1. Run the MiniGPT training example and experiment with hyperparameters (try increasing layers, heads, or embedding dimension).
2. Fine-tune TinyLlama or a similar small model with LoRA on your own domain data.
3. Read Andrej Karpathy's "Let's Build GPT" — the definitive walkthrough of building a transformer from scratch.
4. Study the original "Attention Is All You Need" paper (2017).
5. Explore the Hugging Face model hub and try different base models for fine-tuning.
6. Set up Weights & Biases to track your training experiments.
