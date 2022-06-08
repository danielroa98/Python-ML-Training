# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:44:40 2022
Logistic Regression
@author: luroa
"""

import matplotlib.pyplot as plt
import numpy as np

# Define the function
def sigmoid(inputX):
    return 1/(1+np.exp(-inputX))


inputX = np.arange(-5.0, 5.0, 0.1)

y_pred = sigmoid(inputX)

plt.plot(y_pred, marker='o')
































































































