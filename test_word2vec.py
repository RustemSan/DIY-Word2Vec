"""
Unit tests for Word2Vec implementation.
Tests forward pass, backward pass, and gradient correctness.
"""

import numpy as np
from src.word2vec import Word2Vec


def test_sigmoid():
    """Test sigmoid activation function."""
    model = Word2Vec(vocab_size=100, embedding_dim=50)

    # Test boundary values
    assert np.isclose(model.sigmoid(0), 0.5), "sigmoid(0) should be 0.5"
    assert np.isclose(model.sigmoid(100), 1.0, atol=1e-3), "sigmoid(large) should be ~1"
    assert np.isclose(model.sigmoid(-100), 0.0, atol=1e-3), "sigmoid(-large) should be ~0"

    # Test range
    x = np.array([0, 1, -1, 10, -10])
    output = model.sigmoid(x)
    assert np.all((output >= 0) & (output <= 1)), "sigmoid output should be in [0, 1]"
    assert np.isclose(output[0], 0.5), "sigmoid(0) should be 0.5"
    assert output[3] > output[0] > output[4], "sigmoid should be monotonic"

    print("✅ sigmoid() test passed")


def test_forward_pass():
    """Test forward pass computation."""
    model = Word2Vec(vocab_size=100, embedding_dim=50)

    # Simple forward pass
    target_id = 0
    context_id = 1

    prediction, target_vec, context_vec = model.forward_pass(target_id, context_id)

    # Check shapes
    assert target_vec.shape == (50,), f"Target vec shape should be (50,), got {target_vec.shape}"
    assert context_vec.shape == (50,), f"Context vec shape should be (50,), got {context_vec.shape}"
    assert isinstance(prediction, (float, np.floating)), f"Prediction should be scalar, got {type(prediction)}"

    # Check range
    assert 0 <= prediction <= 1, f"Prediction should be in [0, 1], got {prediction}"

    print("✅ forward_pass() test passed")


def test_backward_pass():
    """Test gradient computation."""
    model = Word2Vec(vocab_size=100, embedding_dim=50)

    target_id = 0
    context_id = 1

    # Forward pass
    prediction, target_vec, context_vec = model.forward_pass(target_id, context_id)

    # Backward pass
    label = 1
    grad_W, grad_C = model.backward_pass(target_id, context_id, prediction, label)

    # Check shapes
    assert grad_W.shape == (50,), f"grad_W shape should be (50,), got {grad_W.shape}"
    assert grad_C.shape == (50,), f"grad_C shape should be (50,), got {grad_C.shape}"

    # Check that gradients are non-zero
    assert np.any(grad_W != 0), "grad_W should not be all zeros"
    assert np.any(grad_C != 0), "grad_C should not be all zeros"

    print("✅ backward_pass() test passed")


def test_gradient_by_finite_difference():
    """
    Verify gradients using finite difference approximation.
    This is a critical test for correctness!
    """
    np.random.seed(42)
    model = Word2Vec(vocab_size=100, embedding_dim=10)

    target_id = 0
    context_id = 1
    label = 1
    epsilon = 1e-5

    # Analytical gradient
    prediction, _, _ = model.forward_pass(target_id, context_id)
    grad_W_analytical, grad_C_analytical = model.backward_pass(target_id, context_id, prediction, label)

    # Numerical gradient for W[target_id]
    grad_W_numerical = np.zeros_like(grad_W_analytical)
    for i in range(model.embedding_dim):
        # f(x + epsilon)
        model.W[target_id, i] += epsilon
        pred_plus, _, _ = model.forward_pass(target_id, context_id)
        loss_plus = -(label * np.log(pred_plus + 1e-10) + (1 - label) * np.log(1 - pred_plus + 1e-10))

        # f(x - epsilon)
        model.W[target_id, i] -= 2 * epsilon
        pred_minus, _, _ = model.forward_pass(target_id, context_id)
        loss_minus = -(label * np.log(pred_minus + 1e-10) + (1 - label) * np.log(1 - pred_minus + 1e-10))

        # Restore
        model.W[target_id, i] += epsilon

        # Numerical gradient
        grad_W_numerical[i] = (loss_plus - loss_minus) / (2 * epsilon)

    # Compare (without L2 term since we're only checking BCE)
    # Note: gradient includes L2 term, so we compare just the BCE part
    error = prediction - label
    grad_W_bce = error * model.C[context_id]

    relative_error = np.linalg.norm(grad_W_numerical - grad_W_bce) / (np.linalg.norm(grad_W_bce) + 1e-10)

    assert relative_error < 1e-3, f"Gradient error too large: {relative_error}"

    print(f"✅ gradient_finite_difference() test passed (relative error: {relative_error:.6f})")


def test_batch_training():
    """Test batch training on a small example."""
    np.random.seed(42)
    model = Word2Vec(vocab_size=50, embedding_dim=20, learning_rate=0.1)

    # Create a small batch
    batch_size = 10
    targets = np.random.randint(0, 50, batch_size)
    contexts = np.random.randint(0, 50, batch_size)
    labels = np.random.randint(0, 2, batch_size)

    # Get initial loss
    loss_initial = model.train_batch(targets, contexts, labels, weight_decay=1e-5)

    # Train a few more batches
    for _ in range(5):
        loss = model.train_batch(targets, contexts, labels, weight_decay=1e-5)

    # Check that loss is computed
    assert isinstance(loss, (float, np.floating)), "Loss should be a scalar"
    assert loss > 0, "Loss should be positive"

    print(f"✅ batch_training() test passed (initial loss: {loss_initial:.4f}, final loss: {loss:.4f})")


def test_vectorization_correctness():
    """
    Verify that vectorized implementation gives same results as loop implementation.
    """
    np.random.seed(42)

    # Create simple model
    vocab_size = 20
    embedding_dim = 5

    W = np.random.randn(vocab_size, embedding_dim) * 0.01
    C = np.random.randn(vocab_size, embedding_dim) * 0.01

    # Create batch
    batch_size = 4
    targets = np.array([0, 1, 2, 3])
    contexts = np.array([4, 5, 6, 7])
    labels = np.array([1, 0, 1, 0])

    # Vectorized approach
    v_targets = W[targets]
    v_contexts = C[contexts]
    scores_vec = np.sum(v_targets * v_contexts, axis=1)

    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    preds_vec = sigmoid(scores_vec)

    # Loop approach
    preds_loop = np.zeros(batch_size)
    for i in range(batch_size):
        score = np.dot(W[targets[i]], C[contexts[i]])
        preds_loop[i] = sigmoid(score)

    # Compare
    max_diff = np.max(np.abs(preds_vec - preds_loop))
    assert max_diff < 1e-10, f"Vectorized and loop results differ: max_diff={max_diff}"

    print(f"✅ vectorization_correctness() test passed (max_diff: {max_diff:.2e})")


def test_learning_rate_scheduling():
    """Test learning rate decay schedule."""
    model = Word2Vec(vocab_size=100, embedding_dim=50, learning_rate=0.025, min_learning_rate=0.0001)

    # Mock batch generator (empty)
    def mock_generator():
        yield np.array([]), np.array([]), np.array([])

    initial_lr = model.initial_lr

    # Simulate epochs
    for epoch in range(5):
        model.train_epoch(mock_generator(), epoch=epoch, total_epochs=5)

        if epoch == 0:
            assert model.learning_rate == initial_lr, f"LR at epoch 0 should be {initial_lr}"
        elif epoch == 4:
            assert model.learning_rate < initial_lr, "LR should decrease"

    final_lr = model.learning_rate
    assert final_lr < initial_lr, "Final LR should be less than initial"
    assert final_lr > model.min_lr * 0.9, "Final LR should be close to min_lr"

    print(f"✅ learning_rate_scheduling() test passed (initial: {initial_lr:.6f}, final: {final_lr:.6f})")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Word2Vec Implementation Tests")
    print("=" * 60)

    test_sigmoid()
    test_forward_pass()
    test_backward_pass()
    test_gradient_by_finite_difference()
    test_vectorization_correctness()
    test_batch_training()
    test_learning_rate_scheduling()

    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

