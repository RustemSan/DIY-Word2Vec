import numpy as np

weights = np.load('word_vectors.npy')

print(f"Shape of weights: {weights.shape}")

print("Vector for word ID 0:")
print(weights[0])