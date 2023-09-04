import tensorflow as tf
import numpy as np
import re
import nltk
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

# Load the saved model
model = tf.keras.models.load_model('hatespeech/best_model')

# New text to make predictions on
new_texts = [
    "How are you doing?", #normal
    'they should gtfo. britain is only for us!!!',
    'fuck off' 
]

# Preprocess the new text
new_texts = [preprocess_text(text) for text in new_texts]

# Convert the new text to the format expected by the model
# Note: The model expects the input in batch, hence we expand dimensions to simulate a batch
new_texts = np.expand_dims(new_texts, -1)

# Make predictions
predictions = model.predict(new_texts)

# Define class names
class_names = ["hate speech", "toxic", "neither"]

# Output the predictions
for i, prediction in enumerate(predictions):
    # The prediction will be an array with the probabilities for each class.
    # Use np.argmax to get the index/class with the highest probability.
    predicted_class_index = np.argmax(prediction)
    if predicted_class_index == 1 or predicted_class_index == 0:
        predicted_class_index = 1
    predicted_class_name = class_names[predicted_class_index]
    print(f"Text: {new_texts[i]}")
    print(f"Predicted Class: {predicted_class_name}")
    print("\n")
