# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 08:54:10 2017

@author: Xiaowen Fang

This file implements kernel regression using kd-tree
"""

from sklearn.neighbors import KDTree
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import pairwise_kernels

class KernelRegression(BaseEstimator, RegressorMixin):
    """Nadaraya-Watson kernel regression with kd-tree.

    Parameters
    ----------
    kernel : string or callable, default="rbf"
        Kernel map to be approximated. A callable should accept two arguments
        and the keyword arguments passed to this object as kernel_params, and
        should return a floating point number.
    bandwidth : float, default=None
        Gamma parameter for the RBF ("bandwidth"), polynomial,
        exponential chi2 and sigmoid kernels. Interpretation of the default
        value is left to the kernel; see the documentation for
        sklearn.metrics.pairwise. Ignored by other kernels. If a sequence of
        values is given, one of these values is selected which minimizes
        the mean-squared-error of leave-one-out cross-validation.
    See also
    --------
    sklearn.metrics.pairwise.kernel_metrics : List of built-in kernels.
    """


    def __init__(self, kernel="rbf", bandwidth=0.7):
        self.kernel = kernel
        self.bandwidth = bandwidth

    def fit(self, X, y):
        """Fit the model
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values
        Returns
        -------
        self : object
            Returns self.
        """
        self.X = X
        self.y = y
        self.tree = KDTree(X, leaf_size=2)
        return self

    def predict(self, X):
        """Predict target values for X.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of shape = [n_samples]
            The predicted target value.
        """
        li = []
        ind = self.tree.query_radius(X, r=self.bandwidth)
        #print(ind)
        for i in range(len(X)):
            #compute kernel between grid point and all the data points near it
            try:
                K = pairwise_kernels(X[i].reshape(1,len(X[i])), self.X[ind[i]], metric=self.kernel, filter_params=True, gamma=0.7)
            except ValueError as e:
                print('The indices of neighbors are empty! ', ind[i])
                print('The lonely grid point is', X[i])

            estimator = np.inner(K, self.y[ind[i]]) / np.sum(K)
            li.append(estimator)
        return li

