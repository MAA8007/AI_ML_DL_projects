# Spam Classifier using RNN README

## Project Description

This project aims to build a spam classifier using a Recurrent Neural Network (RNN). The classifier categorizes messages into "ham" (non-spam) or "spam" categories. It utilizes TensorFlow for deep learning and pandas for data manipulation. 

## Table of Contents

1. [Project Description](#project-description)
2. [Requirements](#requirements)
3. [Dataset](#dataset)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Architecture](#model-architecture)
6. [Training and Evaluation](#training-and-evaluation)

## Requirements

- Python 3.x
- TensorFlow 2.x
- pandas
- scikit-learn

You can install the dependencies using the following command:

```bash
pip install tensorflow pandas scikit-learn
```

## Dataset

The dataset should be a CSV file named `spam.csv` located in a folder called `ham or spam rnn`. The CSV should have a header row and include the following columns:

- `message`: The content of the message.
- `class`: The class label, either "ham" or "spam".

## Data Preprocessing

The code performs basic preprocessing, which includes:

1. Reading the dataset using pandas.
2. Encoding class labels ("ham" as 0 and "spam" as 1).
3. Splitting the data into training and testing sets.

## Model Architecture

The model comprises the following layers:

1. **Text Vectorization**: Converts the text into integer tokens.
2. **Embedding Layer**: Converts tokens into embedding vectors.
3. **Bidirectional LSTM Layer**: To capture temporal dependencies.
4. **Dense Layers**: For classification.
  
The model uses Binary Crossentropy as its loss function and an Adam optimizer with a learning rate of `1e-4`.

## Training and Evaluation

The model is trained for 15 epochs and validated against a test dataset.

To evaluate the model performance:

```bash
test_loss, test_acc = model.evaluate(test_dataset)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)
```

After running the script, a model will be saved in the folder `ham or spam`.

## Contributions

Contributions, issues, and feature requests are welcome. Feel free to check the issues page if you want to contribute.

**Happy Coding!**
