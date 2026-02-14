"""
Part 6: Using tiktoken (OpenAI's Tokenizer)
=============================================
Demonstrates how production tokenizers work using OpenAI's tiktoken library.

Requires: pip install tiktoken
"""

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
