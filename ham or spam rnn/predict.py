import tensorflow as tf

# Load the previously saved model
model = tf.keras.models.load_model('ham or spam rnn/model')

# Function to predict if the text is ham or spam
def predict_text(text):
    # As the model expects a batch of data, we wrap the input text in a list
    predictions = model.predict([text])
    # We only have one input, so we can take the first (and only) prediction
    prediction = predictions[0]
    
    # Convert the prediction to a label 'ham' or 'spam'
    
    if prediction < 0.5:
        return "ham", prediction
    else:
        return "spam", prediction

# Example usage:
text = "Congratulations! you have won a $1,000 gift card. Call now to claim your prize!"
print(f"The message \"{text}\" is {predict_text(text)}")

text = "Hey, are we still on for lunch tomorrow?"
print(f"The message \"{text}\" is {predict_text(text)}")
