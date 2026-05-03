import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("data.csv")

# Features and target
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict([[1600, 3, 2]])
print("Predicted Price:", prediction[0])