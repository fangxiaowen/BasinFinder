# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:27:01 2017

@author: Xiaowen Fang
"""
import scipy
from basinfinder import alpha_shape, Basin_Finder
from descartes import PolygonPatch
import numpy as np
import shapely.geometry as geometry
#from shapely.ops import cascaded_union, polygonize
import math
from kernelregression import KernelRegression     #This is the kd_tree version written by Xiaowen Fang
#import rpy2.robjects as robjects
import matplotlib.pyplot as plt
from matplotlib  import cm
import matplotlib as mpl
import pickle
from time import clock
from multiprocessing import Process, Manager, Array, Pool, Lock
import multiprocessing
import os
import time, threading
import sys

if __name__ == '__main__':
    #multiprocessing.freeze_support()
    print('oaky here?')
    #manager = Manager()
    print('still okay here? Manager Object is created successfully!')
    
    #load data from the .txt file
    protein_data = np.dtype([('col1', 'S20'), ('col2', 'S20'),('col3', 'S20'), ('energy', 'f4'), ('col5', 'f4'), ('col6', 'f4'),('col7', 'f4'),('col8', 'f4'),('col9', 'f4'),('col10', 'f4'),('col11', 'f4'),('col12', 'f4'),('col13', 'f4'),('col14', 'f4'),('col15', 'f4')])
    sample_points = np.loadtxt(r"C:\Users\Administrator\Desktop\BasinFinder\Ras_WT_SoPrimp_2016.txt", dtype = protein_data)           #load data
    #Extrac value we care about
    energy = np.array([e[3] for e in sample_points])         # energy
    points = np.array([[p[5], p[6], i] for i, p in enumerate(sample_points)])
    e_loadData = clock()    #timing
    print('data loaded')
    #Draw the alpha shape from all data points
    concave_hull, edge_points = alpha_shape(np.delete(points, 2, 1), alpha=0.6)    #concave_hull contains all the polygons (all the boundary)
    print('alpha shape builded')    
    #define grid points, [pc1, pc2]
    
    gsz = 1 #Grid size. Could be user input.
    xgridlow, ygridlow, xgridhigh, ygridhigh = concave_hull.bounds
    xgrids =    np.arange(xgridlow - gsz, xgridhigh, gsz)
    ygrids =    np.arange(ygridlow - gsz, ygridhigh, gsz)                    #use generator or numpy
    origin_grids = [[x , y] for x in xgrids for y in ygrids]    #origin grid points, could use numpy
    native_grids = np.array([p for p in origin_grids if geometry.Point(p).within(concave_hull)])      #grid points only in the alpha shape
    native_grids = np.array([[p[0], p[1], i] for i,p in enumerate(native_grids)])
    print('grid points generated')
    # use kernel regression to estimate energy of grid points
    """
    The default kernel 'rbf' is gaussian kernel
    """
    kernel = 'gaussian'          #choose kernel function. More kernels see here  http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html
    try:
        kr = KernelRegression(kernel=kernel, bandwidth=0.7, radius=3)
    except BaseException as e:
        print("What's wrong with kernel reg?")
    kr_model = kr.fit(np.delete(points, 2, 1), energy)   #build kernel regression modle from data points

    f, fprime = kr_model.predict_single, kr_model.fprime
    
    s_newton = clock()
    for i in np.delete(native_grids, 2, 1):
            x = scipy.optimize.fmin_ncg(f, np.array(i), fprime)
            print(x)
    f_newton = clock()
    print('time is : ', f_newton - s_newton)