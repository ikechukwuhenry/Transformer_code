import torch
import torch.nn as nn
import torch.optim as optim

# Sample data
sentence = "the quick brown fox jumps over the lazy dog"
words = sentence.split()
vocab = set(words)
word_to_index = {word: index for index, word in enumerate(vocab)}
index_to_word = {index: word for index, word in enumerate(vocab)}
vocab_size = len(vocab)

# Create context-target pairs
context_window = 2
data = []
for i in range(context_window, len(words) - context_window):
    context_indices = [word_to_index[words[i - j]] for j in range(1, context_window + 1)]
    context_indices.extend([word_to_index[words[i + j]] for j in range(1, context_window + 1)])
    target_index = word_to_index[words[i]]
    data.append((context_indices, target_index))

# CBOW model
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        summed_embeds = torch.sum(embeds, dim=0)
        out = self.linear(summed_embeds)
        return out

# Training
embedding_dim = 10
model = CBOWModel(vocab_size, embedding_dim)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    total_loss = 0
    for context, target in data:
        context_tensor = torch.tensor(context)
        target_tensor = torch.tensor([target])

        optimizer.zero_grad()
        output = model(context_tensor)
        loss = loss_function(output.unsqueeze(0), target_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch+1}, Loss: {total_loss/len(data)}')

# Prediction example
context = ['quick', 'brown', 'over', 'the']
# context2 = ["fox", "jumps", "the", "lazy"]
context_indices = [word_to_index[w] for w in context]
context_tensor = torch.tensor(context_indices)
prediction = model(context_tensor)
predicted_index = torch.argmax(prediction).item()
predicted_word = index_to_word[predicted_index]

print(f'Context: {context}, Predicted word: {predicted_word}')



# just to view the word vector embeddings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# print(model.embeddings.weight)

pca = PCA(n_components=2)
embeddings_weight = model.embeddings.weight
reduced_embeddings = pca.fit_transform(embeddings_weight.detach().numpy())
print(reduced_embeddings)

# Visualize the embeddings
plt.figure(figsize=(5, 5))
for word, idx in word_to_index.items():
    x, y = reduced_embeddings[idx]
    plt.scatter(x, y)
    plt.annotate(word, xy=(x, y), xytext=(5, 2),
                 textcoords='offset points', ha='right', va='bottom')
plt.title("Word Embeddings Visualized")
plt.show()

