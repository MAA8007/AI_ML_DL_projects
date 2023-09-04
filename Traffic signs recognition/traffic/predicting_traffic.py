from keras.models import load_model
import cv2
import numpy as np

# Load your saved model
model = load_model("Traffic signs recognition/traffic/saved model/my_model")

# Load an image file

image_path = 'Traffic signs recognition/fc39.webp'
img = cv2.imread(image_path)

# Resize the image to match the input size that your model expects
img = cv2.resize(img, (30, 30))

# Normalize pixel values to [0, 1]
img = img / 255.0

# Add an extra dimension for the batch size
img = np.expand_dims(img, axis=0)

# Use the model to make a prediction
prediction = model.predict(img)

# The prediction is an array of probabilities, one for each category
# The category with the highest probability is the model's final prediction
predicted_category = np.argmax(prediction)

print(f'The model predicts that the image is in category {predicted_category}.')
