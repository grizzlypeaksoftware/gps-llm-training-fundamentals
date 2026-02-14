"""
Part 6: BPE Tokenization From Scratch
=======================================
A from-scratch implementation of Byte Pair Encoding (BPE),
the tokenization algorithm used by production LLMs.
"""

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

        print(f"Merge {i+1}: '{left}' + '{right}' -> '{merged}' ({count} occurrences)")

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
