# LLM Training Fundamentals — Python & PyTorch

A hands-on guide to understanding how Large Language Models are trained, from the math foundations through building a GPT from scratch to fine-tuning with Hugging Face.

No PhD required — just familiarity with Python and a willingness to think in matrices. Every code example in this guide is runnable on your own machine, most of them without a GPU.

## What's Inside

The [main guide](LLM-Training-Fundamentals-Python-PyTorch.md) covers:

| Part | Topic | Runnable Code |
|------|-------|---------------|
| 1 | What is an LLM? (tokens, embeddings, transformers, parameters) | — |
| 2 | The three phases of training (pre-training, fine-tuning, alignment) | — |
| 3 | Core math (cross-entropy loss, gradient descent, softmax) | `examples/part3_math_foundations/` |
| 4 | Building a GPT from scratch in PyTorch | `examples/part4_mini_gpt/` |
| 5 | Attention mechanism deep dive | `examples/part5_attention/` |
| 6 | BPE tokenization (from scratch + tiktoken) | `examples/part6_tokenization/` |
| 7 | Fine-tuning with Hugging Face and LoRA | `examples/part7_finetuning/` |
| 8 | Key concepts and hyperparameter reference | — |
| 9 | Python libraries and tools for LLM training | — |
| 10 | Cost and compute reference | — |
| 11 | End-to-end workflow summary | — |

## Quick Start

### Prerequisites

- Python 3.9+
- pip

### Install Dependencies

```bash
# Core dependencies (Parts 3-5)
pip install torch numpy

# Tokenization demo (Part 6)
pip install tiktoken

# Fine-tuning (Part 7) — requires a GPU with 8GB+ VRAM
pip install transformers datasets peft trl accelerate bitsandbytes
```

Or install everything at once:

```bash
pip install -r requirements.txt
```

### Run the Examples

**Math foundations (Part 3):**

```bash
python examples/part3_math_foundations/cross_entropy_loss.py
python examples/part3_math_foundations/gradient_descent.py
python examples/part3_math_foundations/softmax.py
```

**Train a GPT from scratch (Part 4):**

```bash
cd examples/part4_mini_gpt
python train.py
```

This trains a ~800K parameter transformer on a small text corpus. Takes a few minutes on CPU, much faster with a GPU. You'll see the loss drop from ~2.9 to ~0.09 over 100 epochs, and the model will generate coherent text from prompts.

**Generate text from a trained model (Part 4):**

After training, you can generate text any time using the saved checkpoint:

```bash
cd examples/part4_mini_gpt
python generate.py                        # default prompt
python generate.py "Python is"            # custom prompt
python generate.py "The transformer" 200  # custom prompt + token count
```

**Attention mechanism demo (Part 5):**

```bash
python examples/part5_attention/attention_demo.py
```

**BPE tokenization (Part 6):**

```bash
python examples/part6_tokenization/bpe_from_scratch.py
python examples/part6_tokenization/tiktoken_demo.py
```

**LoRA fine-tuning (Part 7):**

```bash
cd examples/part7_finetuning
python finetune_lora.py    # Train LoRA adapter
python inference.py         # Generate with fine-tuned model
```

> Note: Part 7 requires a GPU and will download TinyLlama (~2GB).

## Project Structure

```
.
├── LLM-Training-Fundamentals-Python-PyTorch.md   # The full guide
├── README.md
├── requirements.txt
└── examples/
    ├── part3_math_foundations/
    │   ├── cross_entropy_loss.py
    │   ├── gradient_descent.py
    │   └── softmax.py
    ├── part4_mini_gpt/
    │   ├── tokenizer.py      # Character-level tokenizer
    │   ├── model.py           # MiniGPT transformer model
    │   ├── dataset.py         # Dataset and DataLoader
    │   ├── train.py           # Training loop + generation
    │   └── generate.py        # Load checkpoint and generate text
    ├── part5_attention/
    │   └── attention_demo.py  # Standalone attention implementation
    ├── part6_tokenization/
    │   ├── bpe_from_scratch.py  # BPE algorithm from scratch
    │   └── tiktoken_demo.py     # OpenAI's production tokenizer
    └── part7_finetuning/
        ├── finetune_lora.py   # LoRA fine-tuning with Hugging Face
        └── inference.py       # Load and use fine-tuned model
```

## License

MIT
