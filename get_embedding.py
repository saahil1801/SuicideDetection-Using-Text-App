import numpy as np
import torchtext
from torchtext.vocab import GloVe
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# from datapreprocessing import dataprepare
import pickle


def create_embedding_matrix(tokenizer, embedding_dim=300):
    # Load GloVe vectors
    with open('glove.840B.300d.pkl', 'rb') as fp:
        glove_embedding = pickle.load(fp)
    # Initialize an empty embedding matrix
    vocabulary_size = len(tokenizer.word_index)+1
    embedding_matrix = np.zeros((vocabulary_size, embedding_dim))
    
    # Populate the embedding matrix
    for word, idx in tokenizer.word_index.items():
        embedding_vector=glove_embedding.get(word)
        if embedding_vector is not None:
             embedding_matrix[idx]=embedding_vector
    return embedding_matrix

# if __name__ ==  "__main__":
#     filepath = 'Suicide_Detection.csv' 
#     tokenizer, train_pad, test_pad = dataprepare(filepath)
#     embedding_matrix = create_embedding_matrix(tokenizer, 300)
#     print(len(embedding_matrix))
