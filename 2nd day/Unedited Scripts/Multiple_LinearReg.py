# -*- coding: utf-8 -*-
"""
Multiple Linear Regression

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Import dataset & Extract Features
# =============================================================================

dataset = pd.read_csv("02Companies.csv")
X = dataset.iloc[:, :-1]    # X dataframe
y = dataset.iloc[:, 4].values   # y array

# =============================================================================
# Convert Stats Column to Dummy variable
# =============================================================================
X = pd.get_dummies(X,columns=["State"],drop_first=True)

X = X.values     # Dataframe to numpy 2d array

# =============================================================================
# Train Test Split 80-20 
# =============================================================================

from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# =============================================================================
# Model Implementation
# =============================================================================
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train,y_train)

print("b0 / Intercept ", regressor.intercept_)
print("b1 / Slope ", regressor.coef_)

# =============================================================================
# Model Testing
# =============================================================================

y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
error = mean_squared_error(y_test,y_pred)  

errorUnits = np.sqrt(error) 








