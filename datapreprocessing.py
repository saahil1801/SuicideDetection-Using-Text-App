import pandas as pd
import numpy as np
import neattext.functions as nfx
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def dataprepare(filepath):
    data = pd.read_csv(filepath)
    train_data,test_data=train_test_split(data,test_size=0.2,random_state=10)
        
    classes = np.array(train_data['class'])
    train_output = np.where(classes == 'suicide', 1, 0)
    test_output = np.where(np.array(test_data['class']) == 'suicide', 1, 0)

    cleaned_train_text, train_text_length = clean_text(train_data['text'])
    cleaned_test_text, test_text_length = clean_text(test_data['text'])
    tokenizer, train_pad, test_pad = preprocess_text(cleaned_train_text, cleaned_test_text)
    return tokenizer, train_pad, test_pad , train_output , test_output



def clean_text(text):
    text_length=[]
    cleaned_text=[]
    for sent in tqdm(text):
        sent=sent.lower() 
        sent=nfx.remove_special_characters(sent)
        sent=nfx.remove_stopwords(sent)
        text_length.append(len(sent.split()))
        cleaned_text.append(sent)
    return cleaned_text,text_length

def preprocess_text(train_data, test_data, max_len=50):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data)
    train_seq = tokenizer.texts_to_sequences(train_data)
    test_seq = tokenizer.texts_to_sequences(test_data)
    train_pad = pad_sequences(train_seq, maxlen=max_len)
    test_pad = pad_sequences(test_seq, maxlen=max_len)
    return tokenizer, train_pad, test_pad

# if __name__ ==  "__main__":
#     filepath = 'Suicide_Detection.csv' 
#     tokenizer, train_pad, test_pad = dataprepare(filepath)
