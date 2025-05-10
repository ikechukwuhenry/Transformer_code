import numpy as np
import random

# Parameters
V = 10              # Vocabulary size
N = 5               # Embedding size
window_size = 2
k = 3               # Negative samples
lr = 0.01           # Learning rate

# Randomly initialize input and output embedding matrices
W_in = np.random.randn(V, N) * 0.01     # Input vectors
W_out = np.random.randn(V, N) * 0.01    # Output vectors

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate fake dataset: [( context, center), ...]
# Words are represented by indices from 0 to V - 1
dataset = [
    ([1, 2], 3),
    ([0, 4], 2),
    ([3, 4], 1)
]

# Negative sampling distribution (uniform for simplicity)
def get_negative_samples(true_idx, k, V):
    samples = []
    while len(samples) < k:
        neg = random.randint(0, V - 1)
        if neg != true_idx:
            samples.append(neg)
        return samples
    
# Training loop (1 epoch)
for context, target in dataset:
    # Step 1: Compute context vector (average of embeddings)
    h = np.mean(W_in[context], axis=0)      # Shape: (N,)

    # Step 2: Positive example
    v_pos = W_out[target]
    score_pos = np.dot(v_pos, h)
    loss_pos = -np.log(sigmoid(score_pos))

    # Step 3: Negative samples
    neg_samples = get_negative_samples(target, k, V)
    loss_neg = 0
    grad_neg = np.zeros(N)
    for neg in neg_samples:
        v_neg = W_out[neg]
        score_neg = np.dot(v_neg, h)
        loss_neg += -np.log(sigmoid(-score_neg))
        grad_neg += sigmoid(score_neg) * v_neg

        # Update negative output vector
        W_out[neg] -= lr * sigmoid(score_neg) * h

    # Step 4: Gradient and update
    grad_common = (sigmoid(score_pos) - 1) * v_pos + grad_neg
    for word in context:
        W_in[word] -= lr * grad_common / len(context)

    # Update positive output vector
    W_out[target] -= lr * (sigmoid(score_pos) - 1) * h

    # (Optional) print loss
    total_loss = loss_pos + loss_neg
    print(f"Loss: {total_loss:.4f}")