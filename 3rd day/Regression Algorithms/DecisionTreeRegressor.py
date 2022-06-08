# -*- coding: utf-8 -*-
"""
DecisionTree Regressor - Regression Tree - Predicting Continuos Values
@author: TSE
Modified by: danielroa
"""

# =============================================================================
# Import Libraries
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("./MLData/NewPC.csv")
# Average Income -> input

# =============================================================================
#  Feature Extraction
# =============================================================================
X = dataset.iloc[:,[0]].values
y = dataset.iloc[:,1].values

# =============================================================================
# Train Test Split 80-20
# =============================================================================
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# =============================================================================
# Model Implementation
# Fitting DT Regressor
# =============================================================================
from sklearn.tree import DecisionTreeRegressor
#regressor = DecisionTreeRegressor(random_state=0)
regressor = DecisionTreeRegressor(min_samples_split=9)

regressor.fit(X_train,y_train)

# =============================================================================
# Model Testing 
# min_samples_split = 2     error = 39.9                    errorDTrain = 1.3157894736842106
# min_samples_split = 3     error = 46.95                   errorDTrain = 92.80263157894737
# min_samples_split = 4     error = 46.1722222222222        errorDTrain = 545.4473684210526
# min_samples_split = 6     error = 84.4726666666669        errorDTrain = 570.265350877193
# min_samples_split = 9     error = 239.9102500000002       errorDTrain = 896.5758771929824
#
# The high/low er the error the better/worse.
# The high/lower the errorDTrain the better/worse.
#
# =============================================================================
y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
error_DT = mean_squared_error(y_test,y_pred)

y_predTrain = regressor.predict(X_train)

# =============================================================================
# The error_DTrain is an optional metric, it's not that used
# =============================================================================
from sklearn.metrics import mean_squared_error
error_DTrain = mean_squared_error(y_train, y_predTrain)
# =============================================================================
# Tree plotting
# =============================================================================

from sklearn.tree import export_graphviz

export_graphviz(regressor, out_file="treePC-9.dot", feature_names=["Average Salary"])

# =============================================================================
# To view the output from the file correctly, go to the following site:
# http://www.webgraphviz.com/?tab=map
#
# Additional information 
# The "value" in the graph 578.316 is the mean of y_train
# =============================================================================































