import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

def predict_text(text, model_path="model/model.h5", tokenizer_path="tokenizer/tokenizer.pkl", max_len=50):
    tokenizer = pickle.load(open(tokenizer_path, 'rb'))
    model = load_model(model_path)
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len)
    prediction = model.predict(pad)[0][0]
    return prediction
