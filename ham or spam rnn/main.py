import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

# Assuming your CSV file is named 'spam.csv' and has a header row
df = pd.read_csv("ham or spam rnn/spam.csv", encoding="ISO-8859-1")

df["class"] = df["class"].apply(lambda x: 0 if x == "ham" else 1)
X = df["message"]
y = df["class"]

# Split the data into training and testing sets (67% training, 33% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))










# Adapt the encoder to only the text samples before batching
VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text)) # Unbatched dataset

# Batch the datasets
batch_size = 32
train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)












model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()),output_dim=64,mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # Add sigmoid activation for binary classification
])





model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=15, validation_data=test_dataset, validation_steps=30)

test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

model.save('ham or spam/model')

