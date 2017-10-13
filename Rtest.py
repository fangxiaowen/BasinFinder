# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:22:09 2017

@author: Administrator
"""
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
import rpy2.robjects as robjects


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # load sample data
    protein_data = np.dtype([('col1', 'S20'), ('col2', 'S20'),('col3', 'S20'), ('energy', 'f4'), ('col5', 'f4'), ('col6', 'f4'),('col7', 'f4'),('col8', 'f4'),('col9', 'f4'),('col10', 'f4'),('col11', 'f4'),('col12', 'f4'),('col13', 'f4'),('col14', 'f4'),('col15', 'f4')])
    sample_points = np.loadtxt(r"C:\Users\Administrator\Desktop\BasinFinder\Ras_WT_SoPrimp_2016.txt", dtype = protein_data)           #load data
    points = np.array([[p[5], p[6], i] for i, p in enumerate(sample_points)])
    energy = np.array([e[3] for e in sample_points])         # energy
    
    #load R data
    robjects.r['load'](r'C:\Users\Administrator\Desktop\BasinFinder\results_WT_bd_0.7_gridsize_0.1.RData')
    
    # retrieve the matrix that was loaded from the file
    gridEnergy = robjects.r['info']

    # turn the R matrix into a numpy array
    a = np.array(gridEnergy)

    #compute alpha shape
    concave_hull, edge_points = alpha_shape(np.delete(points, 2, 1), alpha=0.6)    #concave_hull contains all the polygons (all the boundary)
    print('Ger alpha shape!')
    #get grid points
    origin_energy = list(a[4])
    origin_grids = np.array(a[3])    #origin grid points, could use numpy. a[0] is x, a[1] is y
    origin_grids = np.insert(origin_grids, 2, origin_energy, axis=1)
    native_grids = np.array([p for p in origin_grids if geometry.Point([p[0], p[1]]).within(concave_hull)])      #grid points only in the alpha shape
    native_energy = native_grids[:,2]
    print('Get grid points!')
    #kernel regression
    kernel = 'gaussian'          #choose kernel function. More kernels see here  http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html
    try:
        kr = KernelRegression(kernel=kernel, bandwidth=0.7, radius=3)
    except BaseException as e:
        print("What's wrong with kernel reg?")

    kr_model = kr.fit(np.delete(points, 2, 1), energy)   #build kernel regression modle from data points.
    grid_energy = kr_model.parallel_predict(np.delete(native_grids,2,axis=1), n_jobs=-1)   #predict energy of grid points
    print('get estimated energy!')

    print("Qiao's and mine are : ", native_energy[:10], "  ", grid_energy[:10])
    #compute the loss
    two_energy = [[grid_energy[i], native_energy[i]] for i in range(len(grid_energy))]
    
    mean_square_error = sum(map(lambda x: (x[0] - x[1])**2, two_energy))/ len(native_grids)
    abs_error = sum(map(lambda x: abs(x[0]-x[1]), two_energy)) / len(native_grids)
    ratio_1_square = math.sqrt(mean_square_error) / np.mean(native_energy)
    ratio_1_abs = abs_error / np.mean(native_energy)
    ratio_2_abs = sum(map(lambda x: abs(x[0] - x[1])/x[1], two_energy)) / len(native_grids)
    print('Get mean square error : %f' % mean_square_error)
    print('Get abs error %f: ' % abs_error)
    print('Get ratio mean square error : %f ' % ratio_1_square)
    print('Get ration abs error : %f ' %ratio_1_abs)
    print('Get abs error divided by mean energy : %f ' % ratio_2_abs)
    