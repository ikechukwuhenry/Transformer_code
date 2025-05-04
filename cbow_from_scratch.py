import numpy as np
import pandas as pd

corpus = "The quick brown fox jumps over the lazy dog"
corpus = corpus.lower().split()

# Define context window size
C = 2
context_target_pairs = []

# Generate context-target pairs
for i in range(C, len(corpus) - C):
    context = corpus[i - C:i] + corpus[i + 1:i + C + 1]
    target = corpus[i]
    context_target_pairs.append((context, target))


print("Context-Target Pairs", context_target_pairs)

# Create vocabulary and map each word to an index
vocab = set(corpus)
word_to_index = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_index.items()}

print("Word to Index Dictionary:", word_to_index)

def one_hot_encode(word, word_to_index):
    one_hot = np.zeros(len(word_to_index))
    one_hot[word_to_index[word]] = 1
    return one_hot

# Example usage for a word "quick"
context_one_hot = [one_hot_encode(word, word_to_index) for word in ['the', 'quick']]
print("One-Hot Encoding for 'quick':",  context_one_hot[1])

# Building CBOW Model from scratch
class CBOW:

    def __init__(self, vocab_size, embedding_dim):
        # Randomly initialize weights for the embedding and output layers
        self.W1 = np.random.randn(vocab_size, embedding_dim)
        self.W2 = np.random.randn(embedding_dim, vocab_size)

    def forward(self, context_words):
        # Calculate the hidden layer (average of context words)
        h = np.mean(context_words, axis=0)
        # Calculate the output layer ( softmax probabilities)
        output = np.dot(h, self.W2)
        return output
    
    def backward(self, context_words, target_word, learning_rate=0.01):
        # Forward pass
        h = np.mean(context_words, axis=0)
        output = np.dot(h, self.W2)

        # Calculate error and gradients
        error = target_word - output
        self.W2 += learning_rate * np.outer(h, error)
        self.W1 += learning_rate * np.outer(context_words, error)


# Example of creating a CBOW object
vocab_size = len(word_to_index)
embedding_dim = 5   # Let's assume 5-dimensional embeddings

cbow_model = CBOW(vocab_size, embedding_dim)

# Using random context words and target  ( as an example )
context_words = [one_hot_encode(word, word_to_index) for word in ['the', 'quick', 'fox', 'jumps']]
context_words = np.array(context_words)
context_words = np.mean(context_words, axis=0)  # average context words
target_word = one_hot_encode('brown', word_to_index)

# Forward pass through the CBOW model
output = cbow_model.forward(context_words)
print("output of CBOW forward pass:", output)