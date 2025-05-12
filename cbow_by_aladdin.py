import torch
import torch.nn as nn
import torch.optim as optim
import spacy

class CBOW(nn.Module):

    def __init__(self, embedding_size=100, vocab_size=-1):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)      # embed(king) - embed(man) + embed(woman) --> queen
        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, inputs):
        # inputs: batch_size * 4
        embeddings = self.embeddings(inputs).mean(1).squeeze(1)
        return self.linear(embeddings)
    

def create_dataset():
    # read text file raw_text.txt as utf-8 in a string
    raw_text_path = "/Users/ikechukwumichael/Desktop/transformer_src/text_data/test.txt"
    # with open("raw_text.txt", "r", encoding="utf-8") as f:
    with open(raw_text_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    # tokenize raw_text with spacy
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 15000000 
    tokenized_text = [token.text for token in nlp(raw_text)]
    vocab = set(tokenized_text)

    # create word to index and index to word mapping
    word_to_idx = {word:i for i, word in enumerate(vocab)}
    idx_to_word = {i:word for i, word in enumerate(vocab)}

    # Generate training data with four words as context. two words before and after target word
    data = []
    for i in range(2, len(tokenized_text) - 2):
        context = [
            tokenized_text[i - 2],
            tokenized_text[i - 1],
            tokenized_text[i + 1],
            tokenized_text[i + 2],
        ]
        target = tokenized_text[i]

        # map context and target to indices and append to data
        context_idxs = [word_to_idx[w] for w in context]
        target_idxs = word_to_idx[target]
        data.append((context_idxs, target_idxs))

    return data, word_to_idx, idx_to_word


def main():
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("mps")
    EMBEDDING_SIZE = 100
    data, word_to_idx, idx_to_word = create_dataset()
    loss_func = nn.CrossEntropyLoss()
    net = CBOW(embedding_size=EMBEDDING_SIZE, vocab_size=len(word_to_idx)).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    context_data = torch.tensor([ex[0] for ex in data]).to(device)
    labels = torch.tensor([ex[1] for ex in data]).to(device)

    # Create dataset from tensors x,y and dataloader
    dataset = torch.utils.data.TensorDataset(context_data, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)

    # Train CBOW model
    for epoch in range(30):
        for context, label in dataloader:
            output = net(context)
            loss = loss_func(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch: {}, Loss: {}".format(epoch, loss.item()))

    
    # just to view the word vector embeddings
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # print(model.embeddings.weight)

    pca = PCA(n_components=2)
    embeddings_weight = net.embeddings.weight.to('cpu')
    reduced_embeddings = pca.fit_transform(embeddings_weight.detach().numpy())
    print(reduced_embeddings)

    # Visualize the embeddings
    plt.figure(figsize=(5, 5))
    for word, idx in word_to_idx.items():
        x, y = reduced_embeddings[idx]
        plt.scatter(x, y)
        plt.annotate(word, xy=(x, y), xytext=(5, 2),
                 textcoords='offset points', ha='right', va='bottom')
        if idx == 50: break
    plt.title("Word Embeddings Visualized")
    plt.show()


if __name__ == "__main__":
    main()
