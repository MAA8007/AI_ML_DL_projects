# College Success Prediction README

## Project Description

This project uses Support Vector Machines (SVM) to predict whether a student will graduate based on a set of provided features. The dataset is read from a CSV file, pre-processed, split into training and testing sets, and then used to train an SVM model. The model's performance is evaluated on the testing set, and it is then used to predict a real-world scenario with data from a new student.

## Table of Contents

1. [Project Description](#project-description)
2. [Requirements](#requirements)
3. [Dataset](#dataset)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Evaluation](#model-evaluation)
6. [Real-life Scenario Prediction](#real-life-scenario-prediction)
7. [How to Run](#how-to-run)

## Requirements

- Python 3.x
- scikit-learn
- csv module (builtin with Python)

You can install the scikit-learn library using the following command:

```bash
pip install scikit-learn
```

## Dataset

The dataset should be a CSV file named `college.csv` located in a folder called `college success prediction`. Each row in the CSV represents a student with:

- 34 feature columns (represented as floating-point numbers).
- A label column which can have values "Enrolled", "Graduate" or other values. 

The label column indicates the final status of the student, with "Graduate" indicating the student graduated and any other value indicating they did not.

## Data Preprocessing

1. The CSV file is read, and the header is skipped.
2. The data is processed to convert feature columns into float types.
3. The label is encoded as 1 for "Graduate" and 0 otherwise.
4. The processed data is split into training (60%) and testing (40%) sets.

## Model Evaluation

After training the SVM model on the training set, it is evaluated on the testing set. The following metrics are computed:

- Number of correct predictions.
- Number of incorrect predictions.
- Overall accuracy.

## Real-life Scenario Prediction

The code also includes a demonstration of how the trained model can be used to predict a real-life scenario. It uses a set of features (`new_student_data`) to predict whether a new student will graduate or not.

## How to Run

1. **Clone the Repository**:
    ```bash
    git clone <repository_link>
    ```

2. **Navigate to the Directory**:
    ```bash
    cd college_success_prediction
    ```

3. **Run the Script**:
    ```bash
    python college_success_predictor.py
    ```

After running the script, you will see the model's performance metrics printed on the screen, followed by a prediction for the `new_student_data`.

## Contributions

Contributions, issues, and feature requests are welcome. Feel free to check the issues page if you want to contribute.

**Happy Coding!**
