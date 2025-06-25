# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


data = pd.read_excel('D:\Quantum\Data quantum.xlsx')

X = data[['Vpp','Vrms','Vavg-rect','Vpeak',]]

y = data['Ground Truth']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Lasso Regressor
lasso_regressor = Lasso(alpha=0.3)  # You can adjust the regularization strength (alpha)

# Train the Lasso Regressor
lasso_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lasso_regressor.predict(X_test)
# Train and Test data 

y_train_pred = lasso_regressor.predict(X_train)
y_test_pred = lasso_regressor.predict(X_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("RMSE Value is:", np.sqrt(mse))
# Print the coefficients
print(f'Coefficients: {lasso_regressor.coef_}')
# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)
mard = (1 / len(y_test)) * sum(abs((y_test - y_pred) / y_test)) * 100
print("Mean Absolute Relative Difference:", mard)
print(lasso_regressor.score(X, y))
MAPE = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print("Mean percentage error:", MAPE)

print("Training RMSE:", train_rmse)
print("Testing RMSE:", test_rmse)
if train_rmse > test_rmse:
    print("The model is unusal need to investigate")
elif train_rmse < test_rmse:
    print("The model is overfit")
elif train_rmse==test_rmse:
    print("The model is balanced")
else:
    print("The model is underfitting")
