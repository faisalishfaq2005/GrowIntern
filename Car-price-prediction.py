import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("car_price_prediction.csv")
print(df.head(2))

# Convert 'Levy' column to numeric
df['Levy'] = pd.to_numeric(df['Levy'], errors='coerce')
median_levy = df['Levy'].median()
df['Levy'] = df['Levy'].fillna(median_levy)
print(df['Levy'].head(5))

# Drop unnecessary columns
df = df.drop(columns=['Engine volume', 'ID', 'Prod. year'])

# Preprocess 'Mileage' column
df['Mileage'] = df['Mileage'].str.replace(' km', '').astype(float)

# Feature scaling for numerical columns
scaler = StandardScaler()
numerical_cols = ['Levy', 'Mileage', 'Cylinders', 'Price', 'Airbags']  # Selecting only numerical columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])  # Applying scaling only to numerical columns

# Encode categorical columns
label_encoder = LabelEncoder()
columns_to_encode = ['Manufacturer', 'Model', 'Category', 'Fuel type', 'Gear box type', 'Drive wheels', 'Doors', 'Wheel', 'Color', 'Leather interior', 'Mileage']
for column in columns_to_encode:
    df[column] = label_encoder.fit_transform(df[column].astype(str))

# Split the data into training and testing sets
X = df.drop(columns=['Price'])
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
print("Model predictions:")
print(model.predict(X_test))
print("Actual predictions:")
print(y_test)
print("Model accuracy:")
print(model.score(X_test, y_test))

print(df.head(6))
