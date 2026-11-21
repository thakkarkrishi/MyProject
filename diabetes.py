import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load Dataset
df = pd.read_csv("D:/data science/diabetes.csv")


# EDA
print("First 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Description:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nOutcome Count:")
print(df['Outcome'].value_counts())

print("\nCorrelation:")
print(df.corr())


# Input and Output
X = df.drop('Outcome', axis=1)
y = df['Outcome']


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)


# Models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),
    "SVM": make_pipeline(
        StandardScaler(),
        SVC()
    ),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Naive Bayes": GaussianNB()
}


# Model Evaluation
model_names = []
accuracies = []

for name, model in models.items():

    print("\n------------------------------")
    print(name)
    print("------------------------------")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    model_names.append(name)
    accuracies.append(accuracy)

# Save Random Forest Model
with open("D:/data science/random_forest_diabetes.pkl", "wb") as file:
    pickle.dump(models["Random Forest"], file)

joblib.dump(
    models["Random Forest"],
    "D:/data science/random_forest_diabetes.joblib"
)

print("\nRandom Forest Model Saved Successfully!")


# Accuracy Comparison Graph
plt.figure(figsize=(10, 5))

plt.bar(model_names, accuracies)

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()