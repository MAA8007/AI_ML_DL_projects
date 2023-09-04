import pandas as pd
import tensorflow as tf
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras.layers.experimental.preprocessing import TextVectorization
from keras.callbacks import ModelCheckpoint
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Lowercasing the text
    text = text.lower()
    
    # Removing special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization (convert text into tokens)
    tokens = text.split()
    
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Re-join tokens
    text = ' '.join(tokens)
    
    return text

# Read the data
df = pd.read_csv("hatespeech/hatespeech.csv")

# Apply preprocessing to each tweet
df["tweet"] = df["tweet"].apply(preprocess_text)

x = df["tweet"]
y = df['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

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

# Build the model with increased complexity
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=128, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.5, recurrent_dropout=0.5)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=1000,
    decay_rate=0.9)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile the model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer=optimizer,
              metrics=['accuracy'])

# Compute class weightsfrom sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=[0, 1, 2], y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}

# Define a model checkpoint callback
checkpoint = ModelCheckpoint('hatespeech/best_model', monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min')

# Train the model with class weights and checkpoint
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset, validation_steps=30,
                    class_weight=class_weight_dict, callbacks=[checkpoint])

# Load the best model
model = tf.keras.models.load_model('hatespeech/best_model')

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)

# Output the results
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)
