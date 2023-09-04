import csv
import random

from sklearn import svm

model = svm.SVC()



# Read data in from file
def change_dtype(row):
    return [float(cell) for cell in row[:34]]

# Read data in from file
with open("college success prediction /college.csv") as f:
    reader = csv.reader(f)
    next(reader)
    
    data = []
    for row in reader:
        if row[34] != "Enrolled":
            evidence = change_dtype(row[:34])  # Processing data and getting evidence
            data.append({
                "evidence": evidence,
                "label": 1 if row[34] == "Graduate" else 0
            })



# Separate data into training and testing groups
holdout = int(0.40 * len(data))
random.shuffle(data)
testing = data[:holdout]
training = data[holdout:]

# Train model on training set
X_training = [row["evidence"] for row in training]
y_training = [row["label"] for row in training]
model.fit(X_training, y_training)

# Make predictions on the testing set
X_testing = [row["evidence"] for row in testing]
y_testing = [row["label"] for row in testing]
predictions = model.predict(X_testing)

# Compute how well we performed
correct = 0
incorrect = 0
total = 0
for actual, predicted in zip(y_testing, predictions):
    total += 1
    if actual == predicted:
        correct += 1
    else:
        incorrect += 1

# Print results
print(f"Results for model {type(model).__name__}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {100 * correct / total:.2f}%")
print("The results above show the accuracy of the model.")
print(" ")
print(" ")
print(" ")



#Real life scenario 
# Assume new_student_data is a list of values about the new student.
new_student_data = [0,1,3,9,1,1,15,1,1,9,10,0,0,0,1,0,1,21,1,0,6,8,6,13.975,0,0,6,7,6,14.942857142857142,0,16.2,0.4,-0.92]

# Then we can predict whether this student will graduate using our model.
predicted_label = model.predict([new_student_data])

if predicted_label == 1:
    print("The model predicts this student will graduate.")
else:
    print("The model predicts this student will not graduate.")

print(" ")
print(" ")
print(" ")
