# Sentiment Analysis using TensorFlow and Keras

## Overview

This project aims to perform sentiment analysis on a given dataset using deep learning techniques with TensorFlow and Keras. The code is divided into two main parts:

1. Data Exploration: Initial exploration and visualization of the dataset to understand its structure and distribution.
2. Model Building and Training: Creating a deep learning model for sentiment analysis, training it, and evaluating its performance.

## Requirements

- pandas
- matplotlib
- seaborn
- nltk
- tensorflow
- keras
- scikit-learn
- wordcloud

## Installation

Run the following command to install the necessary packages:

```bash
pip install pandas matplotlib seaborn nltk tensorflow keras scikit-learn wordcloud
```

## Steps

### Data Exploration

1. Load the dataset using `pandas`.
2. Visualize the distribution of sentiment labels.
3. Generate a word cloud to visualize the most common words in the dataset.
4. Visualize sentence length distribution.

```python
# Snippet for data exploration
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
...
```

### Model Building and Training

1. Preprocess the text data by cleaning and tokenization.
2. Convert categorical labels to one-hot encoded vectors.
3. Split the dataset into training and test sets.
4. Build a deep learning model using bidirectional LSTM layers.
5. Compile the model with CategoricalCrossentropy loss and Adam optimizer.
6. Train the model and evaluate its performance.

```python
# Snippet for model building and training
import tensorflow as tf
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.layers.experimental.preprocessing import TextVectorization
...
```

## Results

After training the model, the test accuracy and F1 score are displayed to evaluate the performance of the model.

