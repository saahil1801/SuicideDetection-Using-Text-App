from sklearn.metrics import classification_report
from keras.models import load_model
# Load the model and the tokenizer
# Define function to evaluate model

def evaluate_model(test_pad, test_output, model_path="model/model.h5"):
    model = load_model(model_path)
    predictions = model.predict(test_pad)
    predictions = (predictions > 0.5).astype(int)
    print(classification_report(test_output, predictions))
