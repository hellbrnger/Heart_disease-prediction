import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


heart_data = pd.read_csv("heart.csv")


imputer = SimpleImputer(strategy='mean')
heart_data = pd.DataFrame(imputer.fit_transform(heart_data), columns=heart_data.columns)


X = heart_data.drop(columns="target", axis=1)
Y = heart_data["target"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=2)


model.fit(X_train, Y_train)


with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
