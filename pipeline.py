from datapreprocessing import dataprepare
from get_embedding import create_embedding_matrix
from train import train_model
from evaluate import evaluate_model
from predict import predict_text
import pickle 

def main():
    # Step 1: Load and preprocess the data
    filepath = 'datasets/Suicide_Detection.csv' 
    tokenizer, train_pad, test_pad , train_output , test_output = dataprepare(filepath)
    pickle.dump(tokenizer, open('tokenizer/tokenizer.pkl', 'wb'))
    embedding_matrix = create_embedding_matrix(tokenizer, 300)

    train_model(train_pad, train_output, test_pad, test_output, len(tokenizer.word_index) + 1, embedding_matrix)
    evaluate_model(test_pad, test_output)

    sample_text = ['Through these past years thoughts of suicide, fear, anxiety I’m so close to my limit']
    result = predict_text(sample_text)
    print(result)

if __name__ ==  "__main__":
    main()
