import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load Dataset
df = pd.read_csv("D:/data science/diabetes.csv")


# Input and Output
X = df.drop('Outcome', axis=1)
y = df['Outcome']


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)


# Load Saved Model
model = joblib.load(
    "D:/data science/random_forest_diabetes.joblib"
)

print("Random Forest Model Loaded Successfully!")


# Testing Model
y_pred = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# New Prediction
new_data = pd.DataFrame(
    [[2, 120, 70, 25, 100, 30.5, 0.5, 25]],
    columns=X.columns
)

result = model.predict(new_data)

if result[0] == 1:
    print("\nNew Prediction: Diabetes")
else:
    print("\nNew Prediction: No Diabetes")