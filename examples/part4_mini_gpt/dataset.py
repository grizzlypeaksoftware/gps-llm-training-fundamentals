"""
Part 4: Dataset and DataLoader
================================
Chunks tokenized text into overlapping sequences for training.
"""

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
