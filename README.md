# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries and load the California Housing dataset into a DataFrame.

2. Set input features `X` and output targets `Y`, then split the data into training and testing sets.

3. Standardize both features and targets using `StandardScaler`.

4. Initialize `SGDRegressor` and wrap it with `MultiOutputRegressor`, then train the model.

5. Predict on the test set, inverse transform the predictions, and calculate Mean Squared Error (MSE).

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Abishek Priyan M
RegisterNumber: 212224240004
*/
```
```py
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())

X = df.drop(columns=['AveOccup','HousingPrice'])
Y = df[['AveOccup','HousingPrice']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, Y_train)

Y_pred = multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
mse = mean_squared_error(Y_test, Y_pred)

print("Mean Squared Error:", mse)
print("\nPredictions:\n", Y_pred[:5])
```

## Output:
### Preview of Dataset
![image](https://github.com/user-attachments/assets/a644cf7e-0580-4a0d-9a7b-1e2c9667e0fb)

![image](https://github.com/user-attachments/assets/4809a757-0882-4384-81c2-0dbb6324490b)

### Prediction

![image](https://github.com/user-attachments/assets/e8800521-be87-4f9a-a056-431b13e9972b)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
