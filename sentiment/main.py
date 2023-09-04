# first, we'll explore the data

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the dataset
df = pd.read_csv('sentiment/data.csv')

# Display the first few rows
print(df.head())

# Basic information about the dataset
print(df.info())

# Statistical summary of numeric columns
print(df.describe())

# Check if there are any missing values
print(df.isnull().sum())

# Distribution of the target variable 'Sentiment'
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Sentiment')
plt.title('Distribution of Sentiment')
plt.show()

# Generate word cloud of the sentences to see most common words
text = ' '.join(df['Sentence'])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(text)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Most Common Words in the Sentences')
plt.show()

# You can also explore the length of sentences
sentence_lengths = df['Sentence'].apply(lambda x: len(x.split()))
plt.figure(figsize=(8, 5))
sns.histplot(sentence_lengths, bins=30, kde=True)
plt.title('Sentence Length Distribution')
plt.xlabel('Number of words in a sentence')
plt.ylabel('Frequency')
plt.show()








#Now, here's the ML/DL portion
import pandas as pd
import tensorflow as tf
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from keras.layers.experimental.preprocessing import TextVectorization
from keras.callbacks import ModelCheckpoint
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load dataset
df = pd.read_csv('sentiment/data.csv')

# Convert labels containing more than one emotion to just one (taking the first one)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    text = ' '.join(tokens)
    return text


import random
import nltk
from nltk.corpus import wordnet

def synonym_replacement(words, n=1):
    words = words.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word.isalnum()]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    sentence = ' '.join(new_words)
    return sentence

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

# Now you can apply this to your dataset. For example:
df["Sentence"] = df["Sentence"].apply(lambda x: synonym_replacement(x, n=1) if random.random() > 0.5 else x)

df["Sentence"] = df["Sentence"].apply(preprocess_text)

# Convert the categorical labels to integer codes
label_encoder = LabelEncoder()
df['Sentiment'] = label_encoder.fit_transform(df['Sentiment'])

# One-hot encode these integer codes
onehot_encoder = OneHotEncoder(sparse=False)
y_onehot = onehot_encoder.fit_transform(df['Sentiment'].values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(df["Sentence"], y_onehot, test_size=0.2, random_state=42,stratify=y_onehot)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Define encoder
VOCAB_SIZE = 5000
encoder = TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

# Batch the datasets
batch_size = 32
train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Build the model
num_classes = y_onehot.shape[1]
from keras.regularizers import l2

model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=128, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.5, recurrent_dropout=0.5)),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)), # Adding L2 regularization here
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)), # Adding L2 regularization here
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])


# Learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=1000,
    decay_rate=0.9)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
from keras import backend as K

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

# Inside your model.compile() call, you can use the custom f1_score metric


# Compile the model with CategoricalCrossentropy
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              optimizer=optimizer,
              metrics=[f1_score])

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=label_encoder.classes_, y=label_encoder.inverse_transform(df['Sentiment']))
class_weight_dict = dict(enumerate(class_weights))

# Define a model checkpoint callback
checkpoint = ModelCheckpoint('sentiment/best_model', monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min')

# Train the model with class weights and checkpoint
history = model.fit(train_dataset, epochs=30, validation_data=test_dataset, validation_steps=30,
                    class_weight=class_weight_dict, callbacks=[checkpoint])

# Load the best model
model = tf.keras.models.load_model('sentiment/best_model', custom_objects={'f1_score': f1_score})

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)

# Output the results
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)
