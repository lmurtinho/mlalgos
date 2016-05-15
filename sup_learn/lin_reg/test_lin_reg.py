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

from lin_reg import linear_regression

boston = load_boston()
X = boston.data
y = boston.target

# tests to consider:
# deals with lists (and lists of lists)?
# deals with data frames?
# handles incorrect sizes of X and y?


def test_boston():
    sk_lr = LinearRegression()
    sk_lr.fit(X, y)
    sk_lr_result = np.insert(sk_lr.coef_, 0, sk_lr.intercept_)
    my_lr = linear_regression(X, y)
    assert_true(np.allclose(sk_lr_result, np.array(my_lr)))

def test_boston_no_interc():
    sk_lr = LinearRegression(fit_intercept=False)
    sk_lr.fit(X, y)
    my_lr = linear_regression(X, y, False)
    assert_true(np.allclose(sk_lr.coef_, np.array(my_lr)))
    

run()