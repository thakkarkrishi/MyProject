
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv("D:/data science/air_quality_dataset.csv")

# All input columns
X = df[['PM2.5', 'PM10', 'NOx', 'NO2', 'SO2',
        'VOCs', 'CO', 'CO2', 'CH4']]

# Target column
y = df['Temperature']

# Correlation
print("Correlation with Temperature:")
print(df.corr(numeric_only=True)['Temperature'].sort_values(ascending=False))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Intercept
print("\nIntercept:", model.intercept_)

# Coefficients
print("\nCoefficients:")
print(pd.DataFrame({
    'Column': X.columns,
    'Coefficient': model.coef_
}))

# Metrics
print("\nR2 Score:", r2_score(y_test, y_pred) * 100)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# New Value Prediction
new_data = pd.DataFrame({
    'PM2.5': [39.96714153],
    'PM10': [57.92603548],
    'NOx': [116.1922135],
    'NO2': [55.23029856],
    'SO2': [4.531693251],
    'VOCs': [75.31726086],
    'CO': [2.789606408],
    'CO2': [427.6743473],
    'CH4': [1.706105123]
})

new_prediction = model.predict(new_data)

print("\nNew Input Values:")
print(new_data)

print("\nPredicted Temperature:", new_prediction[0])
								
