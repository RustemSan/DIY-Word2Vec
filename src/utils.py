"""
Advanced utilities and analysis tools for Word2Vec embeddings.
Use these for evaluation and visualization.
"""

import numpy as np
from typing import List, Tuple, Dict


class EmbeddingAnalyzer:
    """Analyze and evaluate Word2Vec embeddings."""

    def __init__(self, embeddings: np.ndarray, id2word: Dict[int, str]):
        """
        Initialize analyzer.

        :param embeddings: Embedding matrix (vocab_size, dim)
        :param id2word: Mapping from word ID to word string
        """
        self.embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        self.id2word = id2word
        self.vocab_size = len(id2word)
        self.embedding_dim = embeddings.shape[1]

    def cosine_similarity(self, word1_vec: np.ndarray, word2_vec: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        word1_norm = word1_vec / (np.linalg.norm(word1_vec) + 1e-10)
        word2_norm = word2_vec / (np.linalg.norm(word2_vec) + 1e-10)
        return float(np.dot(word1_norm, word2_norm))

    def most_similar(self, word_id: int, topk: int = 5) -> List[Tuple[str, float]]:
        """Find most similar words to a given word."""
        word_vec = self.embeddings[word_id]
        similarities = np.dot(self.embeddings, word_vec)

        # Get top-k excluding the word itself
        top_indices = np.argsort(-similarities)[1:topk+1]
        results = [(self.id2word[idx], float(similarities[idx])) for idx in top_indices]
        return results

    def analogy(self, pos_word1: int, pos_word2: int, neg_word: int, topk: int = 5) -> List[Tuple[str, float]]:
        """
        Solve analogy: neg_word1 is to neg_word as pos_word1 is to ?

        Formula: embedding(answer) ≈ embedding(pos_word1) - embedding(neg_word) + embedding(pos_word2)

        Example: king - man + woman ≈ queen
        """
        target_vec = (self.embeddings[pos_word1] - self.embeddings[neg_word] +
                      self.embeddings[pos_word2])
        target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-10)

        similarities = np.dot(self.embeddings, target_vec)

        # Exclude all three input words
        excluded = {pos_word1, pos_word2, neg_word}

        top_indices = np.argsort(-similarities)
        results = []
        for idx in top_indices:
            if idx not in excluded and len(results) < topk:
                results.append((self.id2word[idx], float(similarities[idx])))

        return results

    def embedding_statistics(self) -> Dict[str, float]:
        """Compute statistics about the embeddings."""
        norms = np.linalg.norm(self.embeddings, axis=1)
        return {
            'mean_norm': float(np.mean(norms)),
            'std_norm': float(np.std(norms)),
            'min_norm': float(np.min(norms)),
            'max_norm': float(np.max(norms)),
            'mean_var': float(np.mean(np.var(self.embeddings, axis=1))),
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
        }

    def nearest_neighbors_batch(self, word_ids: List[int], topk: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """Find neighbors for multiple words at once."""
        return {
            self.id2word[wid]: self.most_similar(wid, topk)
            for wid in word_ids
        }


class VocabularyAnalyzer:
    """Analyze vocabulary properties."""

    def __init__(self, word_counts: Dict[str, int], word2id: Dict[str, int]):
        """
        Initialize analyzer.

        :param word_counts: Word frequency counts
        :param word2id: Word to ID mapping
        """
        self.word_counts = word_counts
        self.word2id = word2id
        self.vocab_size = len(word2id)

    def frequency_distribution(self) -> Dict[str, float]:
        """Analyze word frequency distribution."""
        counts = list(self.word_counts.values())
        total = sum(counts)

        return {
            'vocab_size': self.vocab_size,
            'total_tokens': total,
            'avg_frequency': np.mean(counts),
            'median_frequency': float(np.median(counts)),
            'max_frequency': max(counts),
            'min_frequency': min(counts),
            'unique_ratio': self.vocab_size / total,  # Vocabulary richness
            'hapax_legomena': sum(1 for c in counts if c == 1),  # Words appearing once
        }

    def zipfian_exponent(self) -> float:
        """
        Estimate Zipfian exponent (should be ~1.5 for natural text).

        Zipf's law: frequency ≈ constant / rank^exponent
        """
        counts = sorted(list(self.word_counts.values()), reverse=True)

        # Use log-linear regression on log(rank) vs log(frequency)
        log_ranks = np.log(np.arange(1, len(counts) + 1))
        log_freqs = np.log(np.array(counts))

        # Least squares: slope is -exponent
        A = np.column_stack([log_ranks, np.ones(len(log_ranks))])
        coeffs = np.linalg.lstsq(A, log_freqs, rcond=None)[0]

        return float(-coeffs[0])


def example_usage():
    """Example of how to use these tools."""

    # Assume you have trained model and vocab
    # from main import ... (your training code)

    print("=" * 60)
    print("Example Usage: EmbeddingAnalyzer")
    print("=" * 60)

    print("""
    # After training your model:
    from src.utils import EmbeddingAnalyzer
    
    analyzer = EmbeddingAnalyzer(model.get_vectors(), vocab.id2word)
    
    # 1. Find similar words
    similar = analyzer.most_similar(vocab.word2id['king'], topk=5)
    print("Similar to 'king':")
    for word, sim in similar:
        print(f"  {word}: {sim:.4f}")
    
    # 2. Solve analogies
    results = analyzer.analogy(
        pos_word1=vocab.word2id['king'],
        pos_word2=vocab.word2id['woman'],
        neg_word=vocab.word2id['man'],
        topk=5
    )
    print("king - man + woman = ?")
    for word, sim in results:
        print(f"  {word}: {sim:.4f}")
    
    # 3. Embedding statistics
    stats = analyzer.embedding_statistics()
    print(f"Embedding statistics:")
    print(f"  Mean norm: {stats['mean_norm']:.4f}")
    print(f"  Std norm: {stats['std_norm']:.4f}")
    """)


if __name__ == "__main__":
    example_usage()

