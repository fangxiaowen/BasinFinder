# -*- coding: utf-8 -*-
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
import pickle


"""
Use SET to store points!!!!! instead of list!!!!!!!!!!

Notes on how to make it faster:
    1. Use parallel computing if possible. Including map/reduce and multiprocessing and multithreading.
    2. Use async if possible
    3. Use functional programming / module from third party to reduce code
    4.
"""

if __name__ == '__main__':
    #multiprocessing.freeze_support()
    print('oaky here?')
    #manager = Manager()
    #lock = Lock()
    print('still okay here? Manager Object is created successfully!')
    
    s_loadData = clock()    #timing
    #load data from the .txt file
    protein_data = np.dtype([('col1', 'S20'), ('col2', 'S20'),('col3', 'S20'), ('energy', 'f4'), ('col5', 'f4'), ('col6', 'f4'),('col7', 'f4'),('col8', 'f4'),('col9', 'f4'),('col10', 'f4'),('col11', 'f4'),('col12', 'f4'),('col13', 'f4'),('col14', 'f4'),('col15', 'f4')])
    sample_points = np.loadtxt(r"C:\Users\Administrator\Desktop\BasinFinder\Ras_WT_SoPrimp_2016.txt", dtype = protein_data)           #load data
    #Extrac value we care about
    energy = np.array([e[3] for e in sample_points])         # energy
    #points = np.array([[p[5], p[6], ] for p in sample_points])         #PC1 and PC2 we care about
    points = np.array([[p[5], p[6], i] for i, p in enumerate(sample_points)])
    e_loadData = clock()    #timing
    print('Loading data takes: ', e_loadData - s_loadData)

    s_alphaShape = clock()      #timing
    #Draw the alpha shape from all data points
    concave_hull, edge_points = alpha_shape(np.delete(points, 2, 1), alpha=0.6)    #concave_hull contains all the polygons (all the boundary)
    e_alphaShape = clock()
    print('Drawing alpha shape takes: ', e_alphaShape - s_alphaShape)
    
    s_grid = clock()        #timing
    #define grid points, [pc1, pc2]
    gsz = 0.1 #Grid size. Could be user input.
    xgridlow, ygridlow, xgridhigh, ygridhigh = concave_hull.bounds
    xgrids =    np.arange(xgridlow - gsz, xgridhigh, gsz)
    ygrids =    np.arange(ygridlow - gsz, ygridhigh, gsz)                    #use generator or numpy
    origin_grids = [[x , y] for x in xgrids for y in ygrids]    #origin grid points, could use numpy
    origin_grids = np.array([[p[0], p[1], i] for i, p in enumerate(origin_grids)])
    native_grids = np.array([p for p in origin_grids if geometry.Point(p).within(concave_hull)])      #grid points only in the alpha shape
    native_grids = np.array([[p[0], p[1], i] for i,p in enumerate(native_grids)])
    e_grid = clock()        #timing
    print('Generating grid points takes : ', e_grid - s_grid)
    
    # use kernel regression to estimate energy of grid points
    """
    The default kernel 'rbf' is gaussian kernel
    """
    s_krModel = clock()         #timing
    kernel = 'gaussian'          #choose kernel function. More kernels see here  http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html
    try:
        kr = KernelRegression(kernel=kernel, bandwidth=0.7, radius=3)
    except BaseException as e:
        print("What's wrong with kernel reg?")
    kr_model = kr.fit(np.delete(points, 2, 1), energy)   #build kernel regression modle from data points
    m_krModel = clock()         #timing
    print('Fitting model takes: ', m_krModel - s_krModel)
    #try:
    grid_energy = kr_model.parallel_predict(np.delete(origin_grids,2,1), n_jobs=-1)   #predict energy of grid points
    print(grid_energy)
    #except BaseException as e:
    #    print("Prediction is wrong!")
    c_min = np.amin(grid_energy)
    print('c_min is  : ', c_min)
    e_krModel = clock()     
    print('Predicting grid energy takes(%s): ' %(kernel), e_krModel - m_krModel)
    #Now we finnaly get grid points and their energy value.
    
    origin_grids = np.insert(origin_grids, 3, grid_energy, axis=1)
    print('Good to share grids and grid energy?')
    #native_grids = manager.list(native_grids)   #make native_grids shared between processes
    #grid_energy = manager.list(grid_energy)
    print('Good! grids and grid energy shared!')
    
    ge = open(r'C:\Users\Administrator\Desktop\origin_grid.pickle','wb')
    pickle.dump(origin_grids, ge)
    ge.close()
    sys.exit(0)
    #load grid energy data from file. So we don't need to regenerate it every time
    ge = open(r'C:\Users\Administrator\Desktop\native_grid.pickle', 'rb')
    native_grids = pickle.load(ge)
    #ge.close()
    
    
    #Generate some plot

    #This the the color map
    norm = mpl.colors.Normalize(vmin=-6650, vmax=-6410)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    #This is the figure we want
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    margin = .3
    x_min, y_min, x_max, y_max = concave_hull.bounds
    ax.set_xlim([x_min-margin, x_max+margin])
    ax.set_ylim([y_min-margin, y_max+margin])
    patch = PolygonPatch(concave_hull, fc=m.to_rgba(-6270), ec='#000000', fill=True, zorder=-1)
    np_grids = np.array(native_grids)
    ax.add_patch(patch)
    surf = ax.scatter(np_grids[:,0], np_grids[:,1], c=grid_energy, cmap=cm.jet)
    #surf2 = ax.scatter(points[:,0], points[:,1], c=energy, cmap=cm.jet)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    ge = open(r'C:\Users\Administrator\Desktop\grid_energy.pickle', 'wb')
    pickle.dump(fig,ge)
    #fig.savefig(r'C:\Users\Administrator\Desktop\kernelReg_' + kernel + '_.png')
    sys.exit(0)
    
    #Run it
    print('Okay to create basins list?')
    job_list = manager.Queue(10)
    basins = manager.list()
    Omega = manager.list()
    print('Now basin finder!!!')
    
    p = Pool(4)
    
    s_basin = clock()
    Basin_Finder(native_grids, -6300, Omega, 0.3, m, job_list, basins, True)
    
    wait_for_job = 300
    

    done = False
    while not done:
        try:
            k, S, c, jump = job_list.get(True, wait_for_job)
            p.apply_async(Basin_Finder, args=(S, c, Omega, 0.3, m, job_list, basins, jump))
        except BaseException as e:
            print('Done')
            done = True
    p.close()
    p.join()
    
    for basin in basins:
        ax.add_patch(basin)
        #print('okay now I add one patch')
    print('how many basins? ', len(basins))
    f_basin = clock()
    print('Basin Finder takes(with multiprocessing) : ', f_basin - s_basin)
    
    fig.savefig(r'C:\Users\Administrator\Desktop\grid_energy_6.png')