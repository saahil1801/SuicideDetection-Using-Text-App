# Suicide Detection with Machine Learning

## Project Overview

This project aims to leverage machine learning to detect potential suicide ideation in text data. Using a dataset of text entries classified as either reflecting suicide ideation or not, we build and train a neural network model to predict the likelihood that a given text indicates suicidal thoughts. The model utilizes pre-trained GloVe embeddings for text representation and a LSTM neural network for classification.

## Features

- Text preprocessing including cleaning, tokenization, and padding.
- GloVe embedding matrix creation for text representation.
- LSTM neural network for binary classification of text data.
- Evaluation of model performance on a held-out test set.

## Installation

This project requires Python 3.6+ and the following Python libraries installed:

- NumPy
- pandas
- TensorFlow
- Keras
- scikit-learn
- tqdm
- neattext

You can install the requirements by executing the following command:

```
pip install -r requirements.txt
```

## Usage

To run the project, follow these steps:

1. **Data Preparation**: Ensure your dataset is in the correct format. The expected format is a CSV file with at least two columns: one containing the text data and another indicating the class (suicide ideation or not).

2. **Preprocess the Data**: Run the `data_preprocessing.py` script to clean, tokenize, and pad the text data. This script also splits the dataset into training and testing sets.

   ```
   python data_preprocessing.py
   ```

3. **Train the Model**: Execute the `train.py` script to train the neural network model using the preprocessed data. This script also saves the trained model for later use.

   ```
   python train.py
   ```

4. **Evaluate the Model**: Use the `evaluate.py` script to assess the model's performance on the test set.

   ```
   python evaluate.py
   ```

5. **Make Predictions**: With the `predict.py` script, you can use the trained model to make predictions on new text data.

   ```
   python predict.py 

   ```
6. **Pipeline Overview** 
The pipeline.py script is designed to streamline the process of taking raw text data through a series of steps culminating in the prediction of suicide ideation. This script automates the workflow, ensuring that each step is executed in the correct order and with the appropriate parameters. It encapsulates the entire machine learning pipeline, making the process of training and deploying the model both efficient and reproducible.

To execute the entire machine learning pipeline, simply run the pipeline.py script from the command line:

```
python pipeline.py
```

## Contributing

Contributions to this project are welcome. Please feel free to fork the repository, make changes, and submit pull requests. You can also open an issue if you find bugs or have suggestions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the creators of the GloVe embeddings and the various Python libraries used in this project.

---
