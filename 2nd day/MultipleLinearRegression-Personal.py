# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:17:48 2022
Value to predict in this exercise -> Profit
@author: danielroa
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Import dataset and extract features
# =============================================================================

dataset = pd.read_csv("02Companies.csv")
# X dataframe
X = dataset.iloc[:, :-1]
# y array
y = dataset.iloc[:, 4].values

# =============================================================================
# Convert Stats Column to dummy variable
# =============================================================================
X = pd.get_dummies(data=X, columns=["State"], drop_first=True)
#XNew = pd.get_dummies(data=X, columns=["State"])
#xNewDrop = pd.get_dummies(data=X, columns=["State"], drop_first=True)

X = X.values
# =============================================================================
# Train Test Split 80-20
# =============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# =============================================================================
# Model Implementation
# =============================================================================

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, y_train)

print(f"b0 / Intercept {regressor.intercept_}\nb1 / Slope {regressor.coef_}")

# =============================================================================
# Model Testing
# =============================================================================

y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

error = mean_squared_error(y_test, y_pred)
errorUnits = np.sqrt(error)
score = r2_score(y_test, y_pred)

# =============================================================================
# Model Evaluation - Extra information
# =============================================================================

datasetNew = pd.read_csv("02Companies.csv")
datasetNew = pd.get_dummies(data=datasetNew, columns=["State"], drop_first=True)

# 0.9729004656594832
datasetNew["RNDSpend"].corr(datasetNew["Profit"])
# 0.20071656826872128
datasetNew["Administration"].corr(datasetNew["Profit"])
# 0.7477657217414767
datasetNew["MarketingSpend"].corr(datasetNew["Profit"])
# 0.11624426298842248
datasetNew["State_Florida"].corr(datasetNew["Profit"])
# 0.03136760015130279
datasetNew["State_New York"].corr(datasetNew["Profit"])

# =============================================================================
# Model Testing
# =============================================================================























































