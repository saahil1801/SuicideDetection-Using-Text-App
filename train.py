import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from model import create_model
# Import any necessary functions from data_preprocessing

def train_model(train_pad, train_output, test_pad, test_output, vocabulary_size, embedding_matrix):
    model = create_model(vocabulary_size, 300, embedding_matrix, 50)  # Assume 300 is the embedding dim and 50 is the input length
    model.compile(optimizer=keras.optimizers.SGD(0.1, momentum=0.09), loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(patience=5)
    reducelr = ReduceLROnPlateau(patience=3)
    r = model.fit(train_pad, train_output, validation_data=(test_pad, test_output),
                  epochs=5, batch_size=256, callbacks=[early_stop, reducelr])
    model.save("model/model.h5")
    print("train.py is being read")


