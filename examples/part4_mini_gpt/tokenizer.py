"""
Part 4: Character-Level Tokenizer
===================================
A simple character-level tokenizer for the MiniGPT model.
Production LLMs use BPE (see Part 6), but character-level
tokenization keeps things simple for learning.
"""


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
