import pandas as pd
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Specify the file path
file_path = 'news classifier/News_Category_Dataset_v3.json'

# Lists to store data
links = []
headlines = []
categories = []
short_descriptions = []
authors = []
dates = []

# Open and read the file
with open(file_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        links.append(data['link'])
        headlines.append(data['headline'])
        categories.append(data['category'])
        short_descriptions.append(data['short_description'])
        authors.append(data['authors'])
        dates.append(data['date'])
# Create a DataFrame
df = pd.DataFrame({
    'link': links,
    'headline': headlines,
    'category': categories,
    'short_description': short_descriptions,
    'authors': authors,
    'date': dates
})

# Combine headline and short_description
df['text'] = df['headline'] + ' ' + df['short_description']

x = df['text']
y = df['category']








X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# One-hot encoding of the labels
label_binarizer = LabelBinarizer()
y_train_one_hot = label_binarizer.fit_transform(y_train)
y_test_one_hot = label_binarizer.transform(y_test)

# Batch size
batch_size = 32

# Define the model
VOCAB_SIZE = 20000  # Increased vocabulary size
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(X_train.to_list())

# Model with dropout to reduce overfitting

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1),
                                 initializer='normal')
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1),
                                 initializer='zeros')
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        et = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        at = tf.nn.softmax(et, axis=1)
        at = tf.transpose(at, perm=[0, 2, 1])
        output = tf.matmul(at, x)
        return tf.squeeze(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
# Model with added attention layer



model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=256, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5)),
    AttentionLayer(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(label_binarizer.classes_), activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), # Increase learning rate
              metrics=['accuracy'])

# Train the model
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_one_hot)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test_one_hot)).batch(batch_size)
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# Save the model
model.save('news_classifier_optimized')
#63% accuracy