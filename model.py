import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Load dataset
heart_data = pd.read_csv("heart.csv")

# Handling missing values (if any)
imputer = SimpleImputer(strategy='mean')
heart_data = pd.DataFrame(imputer.fit_transform(heart_data), columns=heart_data.columns)

# Splitting features and target
X = heart_data.drop(columns="target", axis=1)
Y = heart_data["target"]

# Standardizing numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=2)

# Model Training (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=2)
model.fit(X_train, Y_train)

# Save the model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
