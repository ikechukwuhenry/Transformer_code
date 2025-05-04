import tensorflow as tf
import numpy as np

# Define a simple CBOW model using Tensorflow
class CBOWModel(tf.keras.Model):
    
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.output_layer = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, context_words):
        embedded_context = self.embeddings(context_words)
        context_avg = tf.reduce_mean(embedded_context, axis=1)
        output = self.output_layer(context_avg)
        return output
    
# Example usage
model = CBOWModel(vocab_size=8, embedding_dim=5)
context_input = np.random.randint(0, 8, size=(1, 4))    # Random context input
context_input = tf.convert_to_tensor(context_input, dtype=tf.int32)

# Forward pass
output = model(context_input)
print("Output of Tensorflow CBOW model:", output.numpy())



# Using Gensim for CBOW
import gensim
from gensim.models import Word2Vec

# Prepare data ( list of lists of words)
corpus = [["the", "quick", "brown", "fox"], ["jumps", "over", "the", "lazy", "dog"]]

# Train the Word2Vec model using CBOW
model = Word2Vec(corpus, vector_size=5, window=2, min_count=1, sg=0)

# Get the vector representation of a word
vector = model.wv['fox']
print("Vector representation of 'fox':", vector)