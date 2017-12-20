# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 08:54:10 2017

@author: Xiaowen Fang

This file implements kernel regression using kd-tree
"""
from sklearn.neighbors import KDTree
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
#from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import gen_even_slices
from scipy.spatial import distance
from multiprocessing import cpu_count
import sys
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


    def __init__(self, kernel="gaussian", bandwidth=0.7, radius=3):
            # Helper functions - distance
        
        #self.kernel = self.kernel_dict[kernel]
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.radius = radius

    

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

    def predict_single(self, X):
        X = X.reshape(1,-1)
        assert(len(X) == 1)
        return self.predict(X)[0]
    
    #gradient of a point
    def fprime(self, X):
        #print('X is : ', X)
        X = X.reshape(1,-1)
        assert(len(X) == 1)
        try:
            ind = self.tree.query_radius(X, r=self.radius)[0]
            sample = self.X[ind]
        except BaseException as e:
            #print('ind is ? ', X)
            return np.array([10000, 10000])
        
        var = np.repeat(X, len(sample), axis=0)
        exponential = -(((var - sample) ** 2).sum(axis=1)) / (2*(self.bandwidth**2))
        chain = np.repeat(X[0][0], len(sample), axis=0) - sample[:,0]
        #print('var shape, sample shape, exponential shape : ', var.shape, sample.shape, exponential.shape, exponential)

        num1 = (-np.exp(exponential) * chain / (self.bandwidth**2) * self.y[ind]).sum()
        num2 = np.exp(exponential).sum()    
        num3 = (-np.exp(exponential) * chain / (self.bandwidth**2)).sum()    
        num4 = (np.exp(exponential) *  self.y[ind]).sum()
        
        fxn = (num1*num2-num3*num4)/num2 ** 2
        
        chain = np.repeat(X[0][1], len(sample)) - sample[:,1]
        
        num1 = (-np.exp(exponential) * chain / (self.bandwidth**2) * self.y[ind]).sum()
        num3 = (-np.exp(exponential) * chain / (self.bandwidth**2)).sum()    
        fyn = (num1*num2-num3*num4)/num2 ** 2
        
        return np.array((fxn, fyn))
    
    def predict(self, X):
        def gaussian_kernel(x, Y):
            dist = distance.cdist(x,Y)
            dist = - dist**2 / (2* self.bandwidth**2)
            return np.exp(dist)
    
        def epanechnikov_kernel(x, y):
            dist = distance.cdist(x,y)
            #dist[dist > self.bandwidth] = self.bandwidth
            dist = (dist**2) / (self.radius**2)
            return 1 - dist
    
        def tricube_kernel(x, y):
            dist = distance.cdist(x,y)
            #dist[dist > self.bandwidth] = self.bandwidth
            dist /= self.radius
            dist = (1 - dist ** 3) ** 3
            return 70/81 * dist
        def triweight_kernel(x,y):
            dist = distance.cdist(x,y)
            dist = (dist**2) / (self.radius**2)
            dist = (1 - dist) ** 3
            return 35/32 * dist
            
        KERNEL_DICT = {
        'gaussian': gaussian_kernel,
        'epanechnikov': epanechnikov_kernel,
        'tricube': tricube_kernel, 
        'triweight' : triweight_kernel }
        kernel = KERNEL_DICT[self.kernel]
        
        li = []
        try:
            ind = self.tree.query_radius(X, r=self.radius)
        except ValueError as e:
            print('WTF??? : ', X)
            sys.exit(0)
        for i in range(len(X)):
        #compute kernel between grid point and all the data points near it
            try:
                #K = pairwise_kernels(X[i].reshape(1,len(X[i])), self.X[ind[i]], metric=self.kernel, filter_params=True, gamma=0.7)
                Y = self.X[ind[i]]
                K = kernel(X[i].reshape(1,-1), Y)
                #K = list(map(kernel, [(X[i], y) for y in Y]))
            except ValueError as e:
                print('The indices of neighbors are empty! ', ind[i])
                print('The lonely grid point is', X[i])
                sys.exit()
            try:
                #print(K)
                #print("sum of K? ", np.sum(K))
                #if np.sum(K) == 0:
                    #print("How the fuck could this be?", K)
                    #print("ind is ? ", ind)
                    #print("neighbors ? ", Y)
                    #print('Now X[i] is : ',i,"   ", X[i])
                    #sys.exit(1)
                estimator = np.inner(K, self.y[ind[i]]) / np.sum(K) 
                li.append(estimator)
            except ValueError as e:
                print('What is wrong with kernel reg? ', K)
        return li
        
    def parallel_predict(self, X, n_jobs):
        """Predict target values for X.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of shape = [n_samples]
            The predicted target value.
        
        n_jobs : level of parallelism
        """
            
        if n_jobs < 0:
            n_jobs = max(cpu_count() + 1 + n_jobs, 1)
        
        if n_jobs == 1:
        # Special case without multiprocessing parallelism
            return self.predict(X)

        fd = delayed(self.predict)
        ret = Parallel(n_jobs=n_jobs, verbose=0)(
            fd(X[s])
            for s in gen_even_slices(X.shape[0], n_jobs))
        #print('What is the returnvalue? ', ret[0][:100])
        ret = [np.hstack(li) for li in ret]
        return np.hstack(ret)
        #sys.exit(0)
        

    
    
