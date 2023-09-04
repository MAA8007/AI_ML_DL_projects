import random
import csv
import numpy as np
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans



models = [
    KNeighborsClassifier(n_neighbors=4),
    svm.SVC(),
    GaussianNB(),
    Perceptron()
]

# Function to convert the data types in each row
def change_dt(row):
    """
    This function takes a row of the dataset as input,
    converts the data types for the features and returns the modified row.
    """
    # Convert the features to the appropriate data types
    return [
        int(row[0]),  # Administrative
        float(row[1]),  # Administrative_Duration
        int(row[2]),  # Informational
        float(row[3]),  # Informational_Duration
        int(row[4]),  # ProductRelated
        float(row[5]),  # ProductRelated_Duration
        float(row[6]),  # BounceRates
        float(row[7]),  # ExitRates
        float(row[8]),  # PageValues
        float(row[9]),  # SpecialDay
        month_to_index(row[10]),  # Month, converting month name to index
        int(row[11]),  # OperatingSystems
        int(row[12]),  # Browser
        int(row[13]),  # Region
        int(row[14]),  # TrafficType
        0 if row[15] == "not returning" else 1,  # VisitorType
        0 if row[16] == "False" else 1  # Weekend
    ]

# Function to convert month name to index
def month_to_index(month_name):
    """
    This function takes a month name as input,
    returns the index of the month (0 for Jan, 1 for Feb, ...).
    """
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return months.index(month_name)

# Read data from file
with open("shopping prediction/shopping.csv") as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    
    # Process and store the data
    data = []
    for row in reader:
        evidence = change_dt(row[:17])  # Process the row to get evidence
        label = "Shopped" if row[17] == "TRUE" else "Returned"  # Process the label
        data.append({
            "evidence": evidence,
            "label": label
        })

# Shuffle the data and split it into training and testing sets
holdout = int(0.40 * len(data))
random.shuffle(data)
testing = data[:holdout]
training = data[holdout:]

# Extract evidence and labels from the training data
X_training = [row["evidence"] for row in training]
y_training = [row["label"] for row in training]

# Fit the models to the training data
trained_models = []
for model in models:
    model.fit(X_training, y_training)
    trained_models.append(model)

# Extract evidence and labels from the testing data
X_testing = [row["evidence"] for row in testing]
y_testing = [row["label"] for row in testing]

# Make predictions on the testing set
predictions = []
for model in trained_models:
    model_predictions = model.predict(X_testing)
    predictions.append(model_predictions)

# Compute evaluation metrics
metrics = {
    "Accuracy": [],
}

for model_prediction in predictions:
    metrics["Accuracy"].append(accuracy_score(y_testing, model_prediction))

# Print evaluation metrics
print("Evaluation Metrics:")
for metric, values in metrics.items():
    print(f"{metric}: {values}")

#88% accuracy with Perceptron