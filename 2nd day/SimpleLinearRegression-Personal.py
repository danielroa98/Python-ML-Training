# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 14:41:12 2022
Simple Linear Regression - Predict Continuous Value
Will be using just one X feature
@author: danielroa
"""
# =============================================================================
# Import Data & Libraries
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("01HR_Data.csv")
# =============================================================================
# X & Y Extraction
# =============================================================================
# 2D numpy array/matrix
# Since X is independent, it's being used in upper case because it's a coding convention
X = dataset.iloc[:,[0]].values
# 1D numpy array
y = dataset.iloc[:,1].values
# =============================================================================
# Train-Test Split, test_size = 1/3, train_size
# =============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# =============================================================================
# Model Implementation 
# =============================================================================
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# Machine is starting to learn
regressor.fit(X_train, y_train)

print(f"Slope b1 {regressor.coef_}\nIntercept b0{regressor.intercept_}")
# =============================================================================
# Model Testing
# X_test -> years of experience
# y_test -> 
# y_pred -> 
# r2_score -> percentage of variation in Y that is explained by X features
# r2_score 98% -> 98% variation in Salary can be explained by YrsExp
# =============================================================================
y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
error = mean_squared_error(y_test, y_pred)
errorUnits = np.sqrt(error)

from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)

# =============================================================================
# Add-on Visualization
# =============================================================================
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, c="red")
plt.show()

# -- Train data
# plt.scatter(X_train, y_train)
# plt.plot(X_train, regressor.predict(X), c="red")
# plt.show()



















