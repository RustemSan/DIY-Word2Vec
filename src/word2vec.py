import numpy as np
from typing import Tuple
from tqdm import tqdm

class Word2Vec:
    """
    Word2Vec Skip-gram model with Negative Sampling, implemented from scratch using NumPy.

    Core idea:
    - For each word, predict its context words
    - Use negative sampling: minimize loss for 1 positive pair and N negative pairs
    - Learn embeddings through gradient descent
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 100,
                 learning_rate: float = 0.025,
                 min_learning_rate: float = 0.0001):
        """
        Initialize Word2Vec model.

        :param vocab_size: Size of the vocabulary
        :param embedding_dim: Dimension of word embeddings
        :param learning_rate: Initial learning rate for gradient descent
        :param min_learning_rate: Minimum learning rate (for annealing)
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.initial_lr = learning_rate
        self.learning_rate = learning_rate
        self.min_lr = min_learning_rate

        # Initialize embeddings with small random values
        # W: target word embeddings (shape: vocab_size x embedding_dim)
        # C: context word embeddings (shape: vocab_size x embedding_dim)
        self.W = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.C = np.random.randn(vocab_size, embedding_dim) * 0.01

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function: σ(x) = 1 / (1 + e^(-x))
        Clipped to prevent numerical overflow
        """
        # Clip to prevent overflow in exp
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def _sigmoid_derivative(self, sigmoid_output: np.ndarray) -> np.ndarray:
        """
        Derivative of sigmoid: σ'(x) = σ(x) * (1 - σ(x))
        """
        return sigmoid_output * (1.0 - sigmoid_output)

    def forward_pass(self, target_id: int, context_id: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute forward pass for one (target, context) pair.

        :param target_id: ID of the target word
        :param context_id: ID of the context word
        :return: (prediction, target_embedding, context_embedding)
        """
        # Get embeddings
        target_vec = self.W[target_id]          # Shape: (embedding_dim,)
        context_vec = self.C[context_id]        # Shape: (embedding_dim,)

        # Dot product between target and context embeddings
        # This is the score we want to be high for positive pairs
        score = np.dot(target_vec, context_vec)  # Scalar

        # Apply sigmoid to get probability
        prediction = self.sigmoid(score)

        return prediction, target_vec, context_vec

    def backward_pass(self,
                     target_id: int,
                     context_id: int,
                     prediction: float,
                     label: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients for one (target, context) pair using cross-entropy loss.

        Loss: L = -[label * log(σ) + (1-label) * log(1-σ)]

        Derivatives:
        ∂L/∂score = -(label - σ) = σ - label  (for negative sampling)
        ∂score/∂W_target = C_context
        ∂score/∂C_context = W_target

        :param target_id: ID of the target word
        :param context_id: ID of the context word
        :param prediction: Output from forward pass (probability)
        :param label: True label (1 for positive pair, 0 for negative)
        :return: (gradient_W, gradient_C)
        """
        # Get embeddings
        target_vec = self.W[target_id]
        context_vec = self.C[context_id]

        # Compute gradient of loss w.r.t. score
        # For cross-entropy loss: dL/d(score) = prediction - label
        grad_score = prediction - label  # Scalar

        # Gradient w.r.t. target embedding (W): chain rule
        # dL/dW = dL/dscore * dscore/dW = (pred - label) * C_context
        grad_W = grad_score * context_vec  # Shape: (embedding_dim,)

        # Gradient w.r.t. context embedding (C): chain rule
        # dL/dC = dL/dscore * dscore/dC = (pred - label) * W_target
        grad_C = grad_score * target_vec   # Shape: (embedding_dim,)

        return grad_W, grad_C

    def update_embeddings(self,
                         target_id: int,
                         context_id: int,
                         grad_W: np.ndarray,
                         grad_C: np.ndarray):
        """
        Update embeddings using gradient descent.

        W_new = W_old - learning_rate * grad_W
        C_new = C_old - learning_rate * grad_C

        :param target_id: ID of the target word
        :param context_id: ID of the context word
        :param grad_W: Gradient for target embedding
        :param grad_C: Gradient for context embedding
        """
        self.W[target_id] -= self.learning_rate * grad_W
        self.C[context_id] -= self.learning_rate * grad_C

    def train_batch(self, targets: np.ndarray, contexts: np.ndarray, labels: np.ndarray,
                    weight_decay: float = 1e-5) -> float:
        """
        Fully vectorized training on a batch of (target, context, label).
        No Python loops inside! This is much faster.
        """
        # 1. Forward pass (Vectorized)
        # Get all vectors at once using fancy indexing
        v_targets = self.W[targets]  # Shape: (batch_size, embedding_dim)
        v_contexts = self.C[contexts]  # Shape: (batch_size, embedding_dim)

        # Vectorized dot product (row-wise)
        scores = np.sum(v_targets * v_contexts, axis=1)  # Shape: (batch_size,)
        predictions = self.sigmoid(scores)

        # 2. Compute Loss (Vectorized)
        eps = 1e-10
        bce_loss = -np.mean(labels * np.log(predictions + eps) + (1 - labels) * np.log(1 - predictions + eps))
        l2_loss = weight_decay * 0.5 * (np.sum(self.W ** 2) + np.sum(self.C ** 2))
        total_loss = bce_loss + l2_loss

        # 3. Backward pass (Vectorized)
        errors = (predictions - labels).reshape(-1, 1)
        grad_W = errors * v_contexts + weight_decay * v_targets
        grad_C = errors * v_targets + weight_decay * v_contexts


        # 4. Update embeddings (Handling duplicate indices)
        # Using np.add.at is crucial because one word can appear multiple times in a batch
        np.add.at(self.W, targets, -self.learning_rate * grad_W)
        np.add.at(self.C, contexts, -self.learning_rate * grad_C)

        return total_loss


    def train_epoch(self,
                    batch_generator,
                    epoch: int = 0,
                    total_epochs: int = 1,
                    weight_decay: float = 1e-5,
                    total_batches=None) -> float:
        """
        Train for one epoch with a visual progress bar.
        """
        total_loss = 0.0
        batch_count = 0

        # Learning rate annealing: decay learning rate over epochs
        progress = epoch / total_epochs if total_epochs > 0 else 0
        self.learning_rate = self.initial_lr * (1 - progress) + self.min_lr * progress

        # Initialize tqdm progress bar
        pbar = tqdm(
            batch_generator,
            desc=f"Epoch {epoch + 1}/{total_epochs}",
            unit="batch",
            total=total_batches
        )

        for targets, contexts, labels in pbar:
            batch_loss = self.train_batch(targets, contexts, labels, weight_decay=weight_decay)
            total_loss += batch_loss
            batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        return avg_loss

    def get_word_vector(self, word_id: int) -> np.ndarray:
        """
        Get the embedding vector for a word.
        In practice, we typically use W (the target embeddings) for downstream tasks.

        :param word_id: Word ID
        :return: Embedding vector of shape (embedding_dim,)
        """
        return self.W[word_id].copy()

    def get_vectors(self) -> np.ndarray:
        """
        Get all word embedding vectors.

        :return: Matrix of shape (vocab_size, embedding_dim)
        """
        return self.W.copy()

    def most_similar(self, word_id: int, topk: int = 5) -> list:
        """
        Find most similar words based on embedding similarity (cosine distance).

        :param word_id: ID of the query word
        :param topk: Number of top similar words to return
        :return: List of (word_id, similarity_score) tuples
        """
        word_vec = self.W[word_id]

        # Normalize vectors for cosine similarity
        word_vec_norm = word_vec / (np.linalg.norm(word_vec) + 1e-10)
        all_vecs_norm = self.W / (np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-10)

        # Cosine similarity
        similarities = np.dot(all_vecs_norm, word_vec_norm)

        # Get top-k (excluding the word itself)
        top_indices = np.argsort(-similarities)[1:topk+1]
        top_sims = similarities[top_indices]

        return list(zip(top_indices, top_sims))

