import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns

df = pd.read_csv("regression housing/Melbourne_housing_FULL.csv")

#Handle Missing Values - note this might not be the best strategy for all cases
df = df.dropna()

# Drop unnecessary columns
df = df.drop(columns=['Suburb', 'Address', 'SellerG', 'Regionname', 'CouncilArea'])

#Convert Categorical Data to Numerical - here we use One-Hot Encoding 
df = pd.get_dummies(df)
print(df)

sns.pairplot(df.sample(n=100))
plt.show()

#Split Data into Features (X) and Target (y)
X = df.drop('Price', axis=1)
y = df['Price']

#Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Standardize/Normalize Data - only for continuous features
continuous_features = ['Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt', 'Propertycount']
scaler = StandardScaler()
X_train[continuous_features] = scaler.fit_transform(X_train[continuous_features])
X_test[continuous_features] = scaler.transform(X_test[continuous_features])


#Train model
model = LinearRegression()
model.fit(X_train, y_train)

#Make predictions on the testing set 
predictions = model.predict(X_test)

# Compute R-squared for the predictions
print('R-squared:', r2_score(y_test, predictions))
#0.65...