# Hate Speech Classifier README

## Project Description

This project focuses on building a classifier for detecting hate speech in tweets using a deep learning approach. The model is designed to filter tweets into different classes, indicating the level of hate speech. The code includes extensive data preprocessing steps such as text cleaning, tokenization, stopwords removal, and lemmatization. The project also utilizes class weights to handle imbalanced classes in the dataset.

## Table of Contents

1. [Project Description](#project-description)
2. [Requirements](#requirements)
3. [Dataset](#dataset)
4. [Data Preprocessing](#data-preprocessing)
5. [How to Run](#how-to-run)
6. [Model Architecture](#model-architecture)
7. [Training and Evaluation](#training-and-evaluation)

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- pandas
- scikit-learn
- nltk

You can install these requirements using the following command:

```bash
pip install tensorflow keras pandas scikit-learn nltk
```

## Dataset

The dataset should be in CSV format and should reside in a folder named `hatespeech`. The file is expected to be named `hatespeech.csv` and should contain columns:
- `tweet`: The tweet text.
- `class`: The class label indicating the level of hate speech.

## Data Preprocessing

The code includes a preprocessing function `preprocess_text` that performs the following tasks:

1. Converts the text to lowercase.
2. Removes special characters and numbers.
3. Tokenizes the text.
4. Removes stopwords.
5. Lemmatizes the tokens.

## How to Run

1. **Clone the Repository**:
    ```bash
    git clone <repository_link>
    ```

2. **Navigate to the Directory**:
    ```bash
    cd hate_speech_classifier
    ```

3. **Run the Script**:
    ```bash
    python hate_speech_classifier.py
    ```

4. **Load the Best Model for Further Analysis**:
    ```python
    model = tf.keras.models.load_model('hatespeech/best_model')
    ```

## Model Architecture

The model consists of:

1. **Text Vectorization**: Converts the text to integer tokens.
2. **Embedding Layer**: For converting tokens into embedding vectors.
3. **Bidirectional LSTM Layers**: To capture context from both the preceding and following parts of the token sequence.
4. **Dense Layers**: Fully connected layers for classification.
5. **Dropout Layers**: To prevent overfitting.

The model uses Sparse Categorical Crossentropy as its loss function and an Adam optimizer with an exponential decay learning rate schedule.

## Training and Evaluation

The model is trained for 10 epochs using class weights to handle class imbalance. Training employs a checkpoint system that saves the model with the best validation loss.

To evaluate the model's performance, use:

```bash
test_loss, test_acc = model.evaluate(test_dataset)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)
```

## Contribution

Feel free to fork this project, make your own changes, and open a PR for any improvements or fixes.

**Happy Coding!**
