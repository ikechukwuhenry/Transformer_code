import torch
import torch.nn as nn
import torch.optim as optim
import random

# Parameters
V = 10          # Vocabulary size
N = 5           # Embedding size

window_size = 2
k = 3           # Number of negative samples
lr = 0.01

# Fake dataset: (context, center)
dataset = [
    ([1, 2], 3),
    ([0, 4], 2),
    ([3, 4], 1),
]

# Negative sampler
def get_negative_samples(true_idx, k, vocab_size):
    negatives = []
    while len(negatives) < k:
        neg = random.randint(0, vocab_size - 1)
        if neg != true_idx:
            negatives.append(neg)
    return negatives

# CBOW Model with Negative Sampling
class CBOWNegativeSampling(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embed_dim)

        # Initialize weights
        nn.init.xavier_uniform_(self.in_embeddings.weight)
        nn.init.xavier_normal_(self.out_embeddings.weight)

    def forward(self, context_idxs, target_idx, negative_samples):
        # context_idx: (context_size,)
        # target_idx: scalar
        # negative_samples: (k,)

        h = self.in_embeddings(context_idxs)    # shape: (context_size, N)
        h = torch.mean(h, dim=0)                # shape: (N,)

        # Positive sample
        v_pos = self.out_embeddings(target_idx)
        score_pos = torch.dot(h, v_pos)
        loss_pos = -torch.log(torch.sigmoid(score_pos))

        # Negative samples
        v_negs = self.out_embeddings(negative_samples)      # shape: (k, N)
        score_negs = torch.matmul(v_negs, h)                # shape: (k,)
        loss_neg = -torch.log(torch.sigmoid(-score_negs))

        # Calculate total loss as the sum of positive and negative losses
        loss = loss_pos + torch.sum(loss_neg)          # Sum the negative losses
        return loss
    

# Instantiate model
model = CBOWNegativeSampling(V, N)
optimizer = optim.SGD(model.parameters(), lr=lr)

# Training loop
for epoch in range(5):
    total_loss = 0
    for context, center in dataset:
        context = torch.tensor(context, dtype=torch.long)
        target = torch.tensor(center, dtype=torch.long)
        negatives = torch.tensor(get_negative_samples(center, k, V), dtype=torch.long)

        loss = model(context, target, negatives)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss: .4f}")