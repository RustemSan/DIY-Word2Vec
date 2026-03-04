import re
import numpy as np
from src.vocab import Vocab
from src.dataset import Word2VecDataset
from src.word2vec import Word2Vec
from src.optimizers import Word2VecAdaGrad
from src.utils import EmbeddingAnalyzer

def clean_text(text: str):
    """
    Simple cleaning: lowercase and remove non-alphabetical characters.
    """
    text = text.lower()
    # Remove everything except letters and spaces
    tokens = re.sub(r'[^a-z\s]', '', text).split()
    return tokens


def load_text_data(file_path: str) -> str:
    """
    Load text data from a file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return ""


def main():
    print("=" * 60)
    print("Word2Vec Implementation in Pure NumPy")
    print("=" * 60)

    # 1. Load and preprocess text
    print("\n[1] Loading text data...")
    try:
        raw_text = load_text_data("data/text8")
    except:
        print("Warning: text8 file not found, using dummy text")
        raw_text = """
        jetbrains creates professional software development tools
        developing with pycharm is efficient
        word2vec is an interesting algorithm for word embeddings
        neural networks are powerful machine learning models
        natural language processing is fascinating
        """ * 10  # Repeat for more data

    tokens = clean_text(raw_text)
    print(f"Total tokens: {len(tokens)}")
    print(f"Sample tokens: {tokens[:10]}")

    # 2. Build Vocabulary
    print("\n[2] Building vocabulary...")
    vocab = Vocab(min_count=2, subsample_threshold=1e-3)
    vocab.build_vocab(tokens)

    # 3. Convert tokens to IDs
    print("\n[3] Converting tokens to IDs...")
    token_ids = [vocab.word2id[word] for word in tokens if word in vocab.word2id]
    print(f"Tokens after vocabulary filtering: {len(token_ids)}")

    # 4. Subsample frequent words
    print("\n[4] Subsampling frequent words...")
    subsampled_ids = vocab.subsample(token_ids)
    print(f"Original: {len(token_ids)}, After subsampling: {len(subsampled_ids)}")

    # 5. Create dataset and batch generator
    print("\n[5] Creating Word2Vec dataset...")
    dataset = Word2VecDataset(
        tokens=subsampled_ids,
        vocab=vocab,
        window_size=5,
        n_negatives=5
    )

    # 6. Initialize Word2Vec model
    print("\n[6] Initializing Word2Vec model...")
    embedding_dim = 100
    model = Word2VecAdaGrad(
        vocab_size=vocab.vocab_size,
        embedding_dim=embedding_dim,
        learning_rate=0.025,
        min_learning_rate=0.0001
    )
    print(f"Model parameters:")
    print(f"  - Vocabulary size: {vocab.vocab_size}")
    print(f"  - Embedding dimension: {embedding_dim}")
    print(f"  - Total parameters: {vocab.vocab_size * embedding_dim * 2:,}")

    # 7. Training loop
    print("\n[7] Starting training...")
    num_epochs = 5
    batch_size = 128
    weight_decay = 1e-5  # L2 regularization coefficient

    total_batches = len(subsampled_ids) // batch_size

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        batch_gen = dataset.get_batches(batch_size)

        # Pass total_batches to train_epoch
        epoch_loss = model.train_epoch(
            batch_gen,
            epoch,
            num_epochs,
            weight_decay=weight_decay,
            total_batches=total_batches
        )

    # 8. Evaluate embeddings
    print("\n[8] Evaluating embeddings...")
    print("\nMost similar words for 'neural':" if 'neural' in vocab.word2id else "\nMost similar words:")

    # Find a word to use for similarity search
    test_words = ['neural', 'network', 'learning', 'model', 'algorithm']
    found_word = None
    for word in test_words:
        if word in vocab.word2id:
            found_word = word
            break

    if found_word:
        word_id = vocab.word2id[found_word]
        similar_words = model.most_similar(word_id, topk=5)
        print(f"Query word: '{found_word}'")
        for similar_id, similarity in similar_words:
            print(f"  - {vocab.id2word[similar_id]}: {similarity:.4f}")
    else:
        print("  (No test words found in vocabulary)")

    # 9. Save embeddings info
    print("\n[9] Training complete!")
    print(f"Final embedding matrix shape: {model.get_vectors().shape}")

    # 10. Optional: Analyze embedding space
    print("\n[10] Embedding space analysis:")
    embeddings = model.get_vectors()
    print(f"  - Mean norm: {np.mean(np.linalg.norm(embeddings, axis=1)):.4f}")
    print(f"  - Std norm: {np.std(np.linalg.norm(embeddings, axis=1)):.4f}")
    print(f"  - Min value: {embeddings.min():.4f}")
    print(f"  - Max value: {embeddings.max():.4f}")

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)

    print("\n[10] Solving analogies (e.g., king - man + woman)...")
    analyzer = EmbeddingAnalyzer(model.get_vectors(), vocab.id2word)

    analogies = [
        ('king', 'man', 'woman'),
        ('paris', 'france', 'italy'),  # paris - france + italy = rome?
        ('doctor', 'man', 'woman')
    ]

    for pos1, neg, pos2 in analogies:
        if all(w in vocab.word2id for w in [pos1, neg, pos2]):
            res = analyzer.analogy(vocab.word2id[pos1], vocab.word2id[pos2], vocab.word2id[neg])
            print(f"  - {pos1} - {neg} + {pos2} = {res[0][0]} ({res[0][1]:.4f})")

    print("\n[11] Exporting weights...")
    np.save("/kaggle/working/word_vectors.npy", model.v_matrix)
    print("✅ Success: Weights saved to /kaggle/working/word_vectors.npy")


if __name__ == "__main__":
    main()