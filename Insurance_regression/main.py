import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from keras.models import Sequential
from keras.layers import Dense

# Load data
df = pd.read_csv("Insurance_regression/insurance.csv")

# Data Cleaning and Preprocessing
# We remove NaN values to avoid errors during training. One-hot encoding is used to convert categorical data to numerical form so it can be fed into the model.

df = df.dropna()
df = pd.get_dummies(df)




#data visualization:

#Correlation matrix
correlations = df.corr()
print(correlations)

# Heatmap of correlations
sns.heatmap(correlations, annot=True, cmap='coolwarm')
plt.title('Heatmap of Feature Correlations')
plt.show()

# Distribution of the target variable
sns.histplot(df['expenses'], kde=True)
plt.xlabel('Expenses')
plt.ylabel('Frequency')
plt.title('Distribution of Expenses')
plt.show()

# Scatter plot of age vs expenses
sns.scatterplot(x='age', y='expenses', data=df)
plt.xlabel('Age')
plt.ylabel('Expenses')
plt.title('Age vs Expenses')
plt.show()







# Define features and target
X = df.drop('expenses', axis=1)
y = df['expenses']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build neural network model
# We choose several dense layers with ReLU activations because it is a common configuration for regression tasks.
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

# Make predictions
nn_predictions = model.predict(X_test)

# Evaluation Metrics
#MAE tells us how far off our predictions are on average. A lower MAE is better.
mae = mean_absolute_error(y_test, nn_predictions)
print('Neural Network - Mean Absolute Error:', mae)



# Comparison to Other Models
# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

# Decision Tree
tree_model = DecisionTreeRegressor()
tree_model.fit(X_train, y_train)
tree_predictions = tree_model.predict(X_test)

# Metrics for comparison
print('Linear Regression - Mean Absolute Error:', mean_absolute_error(y_test, linear_predictions))
print('Decision Tree - Mean Absolute Error:', mean_absolute_error(y_test, tree_predictions))


#MAE is 2685 for Neural network. for decision tree, it is 3000. the requirement of freecodecamp's task was to have it less than 3500