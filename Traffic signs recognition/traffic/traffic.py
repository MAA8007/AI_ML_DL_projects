import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 15
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.2


def main():
    # Set your data directory and model file paths
    data_dir = "/Users/muhammadarsalanamjad/Library/Mobile Documents/com~apple~CloudDocs/Personal/My_projects/AI : ML : DL/Traffic signs recognition/gtsrb"
    model_file = "Traffic signs recognition/traffic/saved model/my_model"

    # Get image arrays and labels for all image files
    images, labels = load_data(data_dir)

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data with validation data
    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test))

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    model.save(model_file)
    print(f"Model saved to {model_file}.")

    # Print validation accuracy
    val_accuracy = history.history['val_accuracy']
    print("Validation Accuracy:", val_accuracy)


def load_data(data_dir):
    images = []
    labels = []

    # List all directories in data_dir
    categories = os.listdir(data_dir)
    # Iterate through each category directory
    for category in categories:
        category_dir = os.path.join(data_dir, category)

        # Check if category_dir is a directory
        if os.path.isdir(category_dir):
            # List all files in category_dir
            files = os.listdir(category_dir)

            # Iterate through each file in category_dir
            for file in files:
                file_path = os.path.join(category_dir, file)

                # Read the image file and resize it
                img = cv2.imread(file_path)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

                # Append the image and its label to the lists
                images.append(img)
                labels.append(int(category))

    return images, labels


def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
#96% accuracy