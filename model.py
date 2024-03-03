from keras.models import Sequential
from keras.layers import Embedding, LSTM, Input ,GlobalMaxPooling1D, Dense
import tensorflow as tf
import numpy as np

def create_model(vocabulary_size, embedding_dim, embedding_matrix, input_length):
    model = Sequential([
        Input(shape=(input_length,)),
        Embedding(vocabulary_size, embedding_dim, weights=[embedding_matrix], trainable=False),
        LSTM(20, return_sequences=True),
        GlobalMaxPooling1D(),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model
