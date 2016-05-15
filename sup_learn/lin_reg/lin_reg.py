# -*- coding: utf-8 -*-
"""
Linear regression function
"""

import numpy as np

def linear_regression(X, y, intercept=True):
    """
    Function to perform linear regression
    on a set of features X to explain a 
    target y.
    Returns the coefficients for a linear
    regression of y onto X.
    """
    # add intercept column to X
    if intercept:
        X = np.c_[np.ones(X.shape[0]), X]

    # X*theta = y and we want to isolate theta, so:
    # get X transposed (so that Xt*X*theta = Xt*y)
    Xt = X.transpose()
    
    # get inverse of Xt*X  to do
    # ((Xt*X)**-1)*(Xt*X)*theta = ((Xt*X)**-1)*Xt*y or
    # I*theta = ((Xt*X)**-1)*Xt*y
    X_inv = np.linalg.inv(Xt.dot(X))    
        
    return X_inv.dot(Xt).dot(y)