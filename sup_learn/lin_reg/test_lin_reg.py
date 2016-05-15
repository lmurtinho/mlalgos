# -*- coding: utf-8 -*-
"""
Created on Sat May 14 21:23:14 2016

Tests for the linear regression function.

"""

from nose import run
from nose.tools import assert_true
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd

from lin_reg import linear_regression

# Testing with the Boston data set
boston = load_boston()
X = boston.data
y = boston.target
X_df = pd.DataFrame(X)
y_df = pd.DataFrame(y)
X_list = X.tolist()
y_list = y.tolist()

# tests to consider:
    # handles incorrect sizes of X and y?
    # handles non-numeric values?

def sk_results(X, y, intercept=True):
    """
    Returns array with intercept (if intercept=True)
    and coefficients of linear regression of y onto X
    using sklearn.
    """
    rgr = LinearRegression(fit_intercept=intercept)
    rgr.fit(X, y)
    ans = rgr.coef_
    if intercept:
        ans = np.insert(ans, 0, rgr.intercept_)
    return ans

def test_basic():
    sk_lr_result = sk_results(X, y)    
    my_lr = linear_regression(X, y)
    assert_true(np.allclose(sk_lr_result, my_lr))
    
def test_no_interc():
    sk_lr_result = sk_results(X, y, False)    
    my_lr = linear_regression(X, y, False)
    assert_true(np.allclose(sk_lr_result, my_lr))

def test_df():
    sk_lr_result = sk_results(X_df, y_df)    
    my_lr = linear_regression(X_df, y_df)
    assert_true(np.allclose(sk_lr_result, my_lr))
    
def test_list():
    sk_lr_result = sk_results(X_list, y_list)    
    my_lr = linear_regression(X_list, y_list)
    assert_true(np.allclose(sk_lr_result, my_lr))

run()