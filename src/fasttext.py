"""
FastText Word2Vec: Word embeddings with subword information (character n-grams).

Core Idea:
──────────
Word2Vec treats each word as an atomic unit.
Problem: Rare words -> poor embeddings, OOV words -> no representation

FastText Solution:
Represents each word as the sum of its character n-grams embeddings.
Example for word "apple":
  - char 3-grams: <ap, app, ppl, ple, le>
  - Word embedding = mean(embeddings of all n-grams)

Advantages:
✅ Handling OOV words (unknown words = sum of known n-grams)
✅ Better for morphologically complex languages (Russian, German, etc.)
✅ Improves embeddings for rare words
✅ Uses morphological information

Mathematics:
────────────
Word vector: v_word = mean(v_ngram_1, v_ngram_2, ..., v_ngram_k)

Loss remains the same, but updating works differently:
∂L/∂v_ngram = ∂L/∂v_word * ∂v_word/∂v_ngram = ∂L/∂v_word / k

where k = number of n-grams in the word
"""

import numpy as np
from typing import Dict, List, Set


class CharacterNGramExtractor:
    """Extract character n-grams from words."""

    def __init__(self, ngram_size: int = 3, min_ngrams: int = 3):
        """
        :param ngram_size: Size of n-grams (usually 3 for FastText)
        :param min_ngrams: Minimum n-grams in a word (for very short words)
        """
        self.ngram_size = ngram_size
        self.min_ngrams = min_ngrams
        self.ngram_to_id = {}
        self.next_ngram_id = 0

    def extract_ngrams(self, word: str) -> List[int]:
        """
        Extract n-grams from word and return their IDs.

        Example for word="apple", ngram_size=3:
        word_with_bounds = "<apple>"
        n-grams: <ap, app, ppl, ple, le>

        :param word: Input word
        :return: List of n-gram IDs
        """
        # Add boundary symbols (important!)
        word = f"<{word}>"

        ngrams = []
        # Extract all n-grams
        for i in range(len(word) - self.ngram_size + 1):
            ngram = word[i:i+self.ngram_size]

            # Return ID for this n-gram (create if needed)
            if ngram not in self.ngram_to_id:
                self.ngram_to_id[ngram] = self.next_ngram_id
                self.next_ngram_id += 1

            ngrams.append(self.ngram_to_id[ngram])

        # If word is shorter than ngram_size, include whole word
        if len(ngrams) < self.min_ngrams:
            whole_word = f"<{word}>"
            if whole_word not in self.ngram_to_id:
                self.ngram_to_id[whole_word] = self.next_ngram_id
                self.next_ngram_id += 1
            ngrams.append(self.ngram_to_id[whole_word])

        return ngrams

    @property
    def vocab_size(self) -> int:
        """Number of unique n-grams."""
        return len(self.ngram_to_id)


class Word2VecFastText:
    """
    Word2Vec with FastText subword information.

    Key Differences:
    - Standard Word2Vec: embeddings[word_id] = one vector
    - FastText: embeddings[word_id] = mean(embeddings[ngram_1, ngram_2, ...])

    This allows:
    1. Handling OOV words
    2. Using morphological information
    3. Better training of rare words
    """

    def __init__(self,
                 vocab_size: int,
                 vocab_dict: Dict[int, str],  # id2word mapping
                 embedding_dim: int = 100,
                 ngram_size: int = 3,
                 learning_rate: float = 0.025,
                 min_learning_rate: float = 0.0001):
        """
        Initialize FastText Word2Vec.

        :param vocab_size: Number of words in vocabulary
        :param vocab_dict: Mapping from word_id to word (string)
        :param embedding_dim: Embedding dimensionality
        :param ngram_size: Character n-gram size
        :param learning_rate: Learning rate
        :param min_learning_rate: Minimum LR
        """
        self.vocab_size = vocab_size
        self.vocab_dict = vocab_dict
        self.embedding_dim = embedding_dim

        # N-gram extractor
        self.ngram_extractor = CharacterNGramExtractor(ngram_size=ngram_size)

        # Extract n-grams for each word in vocabulary
        self.word_ngrams = {}  # word_id -> list of ngram_ids
        for word_id, word in vocab_dict.items():
            self.word_ngrams[word_id] = self.ngram_extractor.extract_ngrams(word)

        print(f"✅ FastText: {self.ngram_extractor.vocab_size} unique n-grams extracted")

        # Embedding matrices
        # W_ngram: for n-grams (not for words!)
        # C: for context words (usually words are used, not n-grams)
        self.W_ngram = np.random.randn(self.ngram_extractor.vocab_size, embedding_dim) * 0.01
        self.C = np.random.randn(vocab_size, embedding_dim) * 0.01

        # Learning rate scheduling
        self.initial_lr = learning_rate
        self.learning_rate = learning_rate
        self.min_lr = min_learning_rate

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation with clipping for stability."""
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def get_word_embedding(self, word_id: int) -> np.ndarray:
        """
        Get embedding for a word.

        This is the main difference from standard Word2Vec!

        Instead of: embedding = W[word_id]
        FastText: embedding = mean(W_ngram[ngram_ids])

        :param word_id: ID of word
        :return: (embedding_dim,) embedding vector
        """
        ngram_ids = self.word_ngrams[word_id]
        ngram_embeddings = self.W_ngram[ngram_ids]  # (n_ngrams, embedding_dim)
        # Average all n-grams -> one embedding for word
        return np.mean(ngram_embeddings, axis=0)

    def get_word_embedding_batch(self, word_ids: np.ndarray) -> np.ndarray:
        """
        Get embeddings for a batch of words.

        Production version (faster for batch operations).

        :param word_ids: Array of word IDs (batch_size,)
        :return: (batch_size, embedding_dim) embeddings
        """
        batch_size = len(word_ids)
        embeddings = np.zeros((batch_size, self.embedding_dim))

        for i, wid in enumerate(word_ids):
            embeddings[i] = self.get_word_embedding(wid)

        return embeddings

    def train_batch(self, targets: np.ndarray, contexts: np.ndarray,
                   labels: np.ndarray, weight_decay: float = 1e-5) -> float:
        """
        Train batch using FastText embeddings for targets.

        :param targets: Target word IDs (batch_size,)
        :param contexts: Context word IDs (batch_size,)
        :param labels: Labels 1 (positive) or 0 (negative) (batch_size,)
        :param weight_decay: L2 regularization strength
        :return: Loss value
        """
        # 1. Get embeddings
        # For target words we use n-grams, for context use standard embeddings
        v_targets = self.get_word_embedding_batch(targets)  # (B, d)
        v_contexts = self.C[contexts]  # (B, d)

        # 2. Forward pass
        scores = np.sum(v_targets * v_contexts, axis=1)  # (B,)
        predictions = self.sigmoid(scores)  # (B,)

        # 3. Compute loss
        eps = 1e-10
        bce_loss = -np.mean(labels * np.log(predictions + eps) +
                           (1 - labels) * np.log(1 - predictions + eps))

        # L2 regularization (for n-gram embeddings)
        l2_loss = weight_decay * 0.5 * (np.sum(self.W_ngram ** 2) + np.sum(self.C ** 2))
        total_loss = bce_loss + l2_loss

        # 4. Backward pass
        errors = (predictions - labels).reshape(-1, 1)  # (B, 1)

        # Gradients
        grad_v_targets = errors * v_contexts  # (B, d)
        grad_C = errors * v_targets + weight_decay * v_contexts  # (B, d)

        # 5. Update context embeddings (as usual)
        np.add.at(self.C, contexts, -self.learning_rate * grad_C)

        # 6. Update n-gram embeddings (KEY MOMENT)
        # Each n-gram should be updated proportionally to its contribution to loss
        for batch_idx, target_id in enumerate(targets):
            ngram_ids = self.word_ngrams[target_id]
            grad_ngram = grad_v_targets[batch_idx] / len(ngram_ids)  # Spread gradient

            # Update all n-grams in this word
            np.add.at(self.W_ngram, ngram_ids, -self.learning_rate * grad_ngram)

        # L2 regularization for n-grams
        self.W_ngram -= self.learning_rate * weight_decay * self.W_ngram

        return total_loss

    def train_epoch(self, batch_generator, epoch: int = 0,
                   total_epochs: int = 1, weight_decay: float = 1e-5) -> float:
        """
        Train for one epoch with learning rate scheduling.

        :param batch_generator: Generator yielding (targets, contexts, labels)
        :param epoch: Current epoch number
        :param total_epochs: Total number of epochs
        :param weight_decay: L2 regularization strength
        :return: Average loss for epoch
        """
        total_loss = 0.0
        batch_count = 0

        # Learning rate decay
        progress = epoch / total_epochs if total_epochs > 0 else 0
        self.learning_rate = self.initial_lr * (1 - progress) + self.min_lr * progress

        for targets, contexts, labels in batch_generator:
            batch_loss = self.train_batch(targets, contexts, labels, weight_decay=weight_decay)
            total_loss += batch_loss
            batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        return avg_loss

    def most_similar(self, word_id: int, topk: int = 5) -> List[tuple]:
        """
        Find most similar words based on FastText embeddings.

        :param word_id: Query word ID
        :param topk: Number of similar words to return
        :return: List of (word_id, similarity) tuples
        """
        # Get query embedding (average its n-grams)
        query_vec = self.get_word_embedding(word_id)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)

        # Get all word embeddings
        all_embeddings = np.array([self.get_word_embedding(i) for i in range(self.vocab_size)])
        all_embeddings = all_embeddings / (np.linalg.norm(all_embeddings, axis=1, keepdims=True) + 1e-10)

        # Cosine similarity
        similarities = np.dot(all_embeddings, query_vec)

        # Top-k excluding the query word itself
        top_indices = np.argsort(-similarities)[1:topk+1]
        results = [(idx, float(similarities[idx])) for idx in top_indices]

        return results

    def get_vector_for_oov_word(self, word: str) -> np.ndarray:
        """
        Get embedding for an unknown word!

        This is the MAIN advantage of FastText:
        - Standard Word2Vec: OOV word = no embedding
        - FastText: OOV word = mean(embeddings of its n-grams)

        Example:
        "apple" -> unknown word
        but its n-grams <ap, app, ppl, ple, le> can be known
        (they appeared in other words)

        :param word: Unknown word
        :return: Embedding vector
        """
        ngram_ids = self.ngram_extractor.extract_ngrams(word)
        ngram_embeddings = self.W_ngram[ngram_ids]
        return np.mean(ngram_embeddings, axis=0)


def comparison_fasttext_vs_word2vec():
    """
    Comparison of FastText and standard Word2Vec.
    """
    print("=" * 70)
    print("FastText vs Word2Vec Comparison")
    print("=" * 70)

    comparison = {
        "Feature": ["OOV words", "Rare words", "Morphology", "Memory", "Speed", "Quality"],
        "Word2Vec": ["❌ None", "⚠️ Poor", "❌ No", "✅ Low", "✅ Fast", "⭐⭐⭐"],
        "FastText": ["✅ Yes!", "✅ Better", "✅ Uses", "⚠️ Medium", "⚠️ Slower", "⭐⭐⭐⭐"],
    }

    print("\n")
    print(f"{'Feature':<20} {'Word2Vec':<20} {'FastText':<20}")
    print("─" * 60)
    for i in range(len(comparison["Feature"])):
        feature = comparison["Feature"][i]
        w2v = comparison["Word2Vec"][i]
        ft = comparison["FastText"][i]
        print(f"{feature:<20} {w2v:<20} {ft:<20}")

    print("\n✅ Recommendation: Use FastText if:")
    print("  • There are OOV words in test")
    print("  • Language is morphologically complex (Russian, German, etc.)")
    print("  • You want better handling of rare words")
    print("  • You have enough memory")

    print("\n✅ Use standard Word2Vec if:")
    print("  • Closed vocabulary (no OOV)")
    print("  • Need maximum speed")
    print("  • Memory is limited")

