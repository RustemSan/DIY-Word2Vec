import numpy as np
from collections import Counter


class Vocab:
    def __init__(self, min_count: int = 5, subsample_threshold: float = 1e-3):
        """
        Initialize the Vocabulary handler.
        :param min_count: Minimum frequency of a word to be kept in the vocabulary.
        :param subsample_threshold: Threshold for dropping frequent words (e.g., 'the', 'is').
        """
        self.min_count = min_count
        self.threshold = subsample_threshold
        self.word2id = {}  # Map: word string -> unique integer ID
        self.id2word = {}  # Map: unique integer ID -> word string
        self.word_counts = Counter()
        self.vocab_size = 0

    def build_vocab(self, tokens: list[str]):
        """
        Create word-to-ID and ID-to-word mappings from a list of tokens.
        """
        # Count frequency of each word in the provided token list
        self.word_counts.update(tokens)

        # Filter out words that appear less than min_count times to reduce noise
        valid_words = [w for w, c in self.word_counts.items() if c >= self.min_count]

        # Sort words by frequency (optional, but keeps the mapping stable)
        valid_words.sort(key=lambda w: self.word_counts[w], reverse=True)

        # Create bidirectional mappings
        for i, word in enumerate(valid_words):
            self.word2id[word] = i
            self.id2word[i] = word

        self.vocab_size = len(self.word2id)
        print(f"Vocabulary built successfully. Total unique words: {self.vocab_size}")

    def subsample(self, token_ids: list[int]) -> list[int]:
        """
        Perform subsampling of frequent words to improve training speed and quality.
        Words like 'the' or 'a' provide less information than rare words like 'algorithm'.
        """
        if not self.word_counts:
            return token_ids

        total_count = sum(self.word_counts.values())
        kept_tokens = []

        for tid in token_ids:
            word = self.id2word[tid]
            # Calculate word frequency relative to the entire dataset
            freq = self.word_counts[word] / total_count

            # Mikolov's formula for the probability of keeping a word:
            # P(w) = sqrt(threshold / freq) + (threshold / freq)
            # We simplify it here using a standard version:
            prob_keep = (np.sqrt(freq / self.threshold) + 1) * (self.threshold / freq)

            # If a random number is less than the probability, we keep the word
            if np.random.random() < prob_keep:
                kept_tokens.append(tid)

        return kept_tokens