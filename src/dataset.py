import numpy as np


class Word2VecDataset:
    def __init__(self, tokens: list[int], vocab, window_size: int = 5, n_negatives: int = 5):
        """
        Initialize Word2Vec dataset.

        :param tokens: List of word IDs from the corpus.
        :param vocab: The Vocab object containing word counts and mappings.
        :param window_size: Maximum distance between target and context word.
        :param n_negatives: Number of negative samples per positive pair.
        """
        self.tokens = tokens
        self.vocab = vocab
        self.window_size = window_size
        self.n_negatives = n_negatives
        # Pre-compute the negative sampling table for efficiency
        self.neg_sampling_table = self._create_neg_table()

    def _create_neg_table(self, table_size: int = 10_000_000):
        """
        Creates a table for fast negative sampling using the 3/4 power rule.
        This pre-computed table allows efficient sampling without calling
        np.random.choice repeatedly, which would be slow in the main loop.
        """
        # Get counts for all words in the vocabulary
        counts = np.array([self.vocab.word_counts[self.vocab.id2word[i]] for i in range(self.vocab.vocab_size)])

        # Apply 3/4 power rule as suggested in the original Word2Vec paper
        # This gives medium-frequency words higher probability
        pow_counts = counts ** 0.75
        probs = pow_counts / pow_counts.sum()

        return np.random.choice(self.vocab.vocab_size, size=table_size, p=probs)

    def get_batches(self, batch_size: int):
        """
        Generator that yields batches of (target, context, label).
        Uses bulk negative sampling for high performance in NumPy.
        """
        targets, contexts, labels = [], [], []

        # Optimization: Pre-generate a large pool of negative indices
        neg_idx = 0

        for i, target in enumerate(self.tokens):
            # Dynamic window size (standard Word2Vec trick)
            current_window = np.random.randint(1, self.window_size + 1)

            # Define window range
            start = max(0, i - current_window)
            end = min(len(self.tokens), i + current_window + 1)

            for j in range(start, end):
                if i == j:
                    continue

                # 1. Add Positive Pair
                context = self.tokens[j]
                targets.append(target)
                contexts.append(context)
                labels.append(1)

                # 2. Add Negative Pairs
                # PERFORMANCE TIP: Sampling in bulk is much faster than np.random.choice inside a loop
                for _ in range(self.n_negatives):
                    # Rotate through the pre-computed table
                    neg_context = self.neg_sampling_table[neg_idx % len(self.neg_sampling_table)]
                    neg_idx += 1

                    targets.append(target)
                    contexts.append(neg_context)
                    labels.append(0)

                # If batch is ready, yield it
                if len(targets) >= batch_size:
                    yield (np.array(targets[:batch_size]),
                           np.array(contexts[:batch_size]),
                           np.array(labels[:batch_size]))
                    # Clear lists for the next batch
                    targets, contexts, labels = [], [], []