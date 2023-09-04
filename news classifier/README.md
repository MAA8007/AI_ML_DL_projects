# News Classifier README

## Project Description
This project aims to build a News Classifier using Deep Learning. The model takes in news articles and categorizes them into different classes. The model architecture includes LSTM layers along with a custom Attention Layer for improved performance. The model is trained on a dataset that includes news headlines, short descriptions, authors, categories, and links. It achieves an accuracy of around 63%.

## Table of Contents
1. [Project Description](#project-description)
2. [Requirements](#requirements)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Evaluation](#evaluation)

## Requirements

- Python 3.x
- TensorFlow 2.x
- pandas
- scikit-learn
- JSON

You can install the requirements using the following command:
```bash
pip install tensorflow pandas scikit-learn
```

## Dataset
The dataset is in JSON format and resides in a file called `News_Category_Dataset_v3.json`. Each line in this file is a JSON object with the following keys:
- `link`
- `headline`
- `category`
- `short_description`
- `authors`
- `date`

## Model Architecture

The model is a Sequential model that includes the following layers:

1. **Text Vectorization**: Converts the text data into numerical tokens.
2. **Embedding Layer**: Maps each token to a higher-dimensional space.
3. **Bidirectional LSTM Layer**: Captures the sequence information.
4. **Attention Layer**: Custom layer to weigh the importance of different parts of the sequence.
5. **Dense Layers**: Fully connected layers.
6. **Dropout Layers**: To reduce overfitting.

The model uses `categorical_crossentropy` as the loss function and `Adam` optimizer with a learning rate of 0.001.

## Evaluation

The model achieves an accuracy of approximately 63% on the test dataset.

## Contribution

Feel free to fork the project, open a PR or an issue for any suggestions or improvements.

**Happy Coding!**
