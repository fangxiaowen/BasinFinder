# -*- coding: utf-8 -*-
from basinfinder import alpha_shape, Basin_Finder
import basinfinder
from descartes import PolygonPatch
#from scipy.spatial import Delaunay
import numpy as np
#from statsmodels.nonparametric.kernel_regression import KernelReg
import shapely.geometry as geometry
#from shapely.ops import cascaded_union, polygonize
import math
from kernelregression import KernelRegression     #This is the kd_tree version written by Xiaowen Fang
#import rpy2.robjects as robjects
#from sklearn.metrics import pairwise
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib  import cm
import matplotlib as mpl
import pickle
from time import clock
from multiprocessing import Process, Manager, Array, Pool
import multiprocessing
import os
import time, threading
import sys


"""
Use SET to store points!!!!! instead of list!!!!!!!!!!

Notes on how to make it faster:
    1. Use parallel computing if possible. Including map/reduce and multiprocessing and multithreading.
    2. Use async if possible
    3. Use functional programming / module from third party to reduce code
    4.
"""
#model = open(r'C:\Users\Administrator\Desktop\kr_model.txt','wb')
#pickle.dump(kr_model, model)
#model.close()

#Just use the saved model to save time
#model = open(r'C:\Users\Administrator\Desktop\kr_model.txt', 'rb')
#kr_model = pickle.load(model)
#model.close()

#Multiprocessing
"""
with Manager() as manager:
    total_grid_energy = manager.list(range(len(native_grids)))
    print("What's wrong with manager?")
    Omega = manager.list()
print('Multithread is working!')
"""
#Use multithread to speed up prediction of grid energy
"""
grid_energy = [0] * len(native_grids)
partition = math.ceil(len(native_grids)/10) #partition the task into 10 subtasks

#method to do prediction on subtask
def addEnergy(grids, index):
    print('thread %s is running...' % threading.current_thread().name)
    global grid_energy
    global partition
    global kr_model
    grid_energy[i*partition:(i+1)*partition] = kr_model.predict(grids).tolist()
    print(grid_energy[i*partition : i*partition + 10])
    return

#multithreading
threads = []
for i in range(10):
    t = threading.Thread(target=addEnergy, args=(native_grids[i*partition:(i+1)*partition], i), name='kr thread')
    threads.append(t)

for i in range(len(threads)):
    threads[i].start()

for j in range(len(threads)):
    threads[j].join()
"""








if __name__ == '__main__':
    multiprocessing.freeze_support()
    print('oaky here?')
    manager = Manager()
    print('still okay here? Manager Object is created successfully!')
    
    s_loadData = clock()    #timing
    #load data from the .txt file
    protein_data = np.dtype([('col1', 'S20'), ('col2', 'S20'),('col3', 'S20'), ('energy', 'f4'), ('col5', 'f4'), ('col6', 'f4'),('col7', 'f4'),('col8', 'f4'),('col9', 'f4'),('col10', 'f4'),('col11', 'f4'),('col12', 'f4'),('col13', 'f4'),('col14', 'f4'),('col15', 'f4')])
    points = np.loadtxt(r"C:\Users\Administrator\Desktop\BasinFinder\Ras_WT_SoPrimp_2016.txt", dtype = protein_data)           #load data
    #Extrac value we care about
    energy = np.array([e[3] for e in points])         # energy
    points = np.array([[p[5], p[6]] for p in points])         #PC1 and PC2 we care about
    e_loadData = clock()    #timing
    print('Loading data takes: ', e_loadData - s_loadData)

    s_alphaShape = clock()      #timing
    #Draw the alpha shape from all data points
    concave_hull, edge_points = alpha_shape(points, alpha=0.6)    #concave_hull contains all the polygons (all the boundary)
    e_alphaShape = clock()
    print('Drawing alpha shape takes: ', e_alphaShape - s_alphaShape)
    
    s_grid = clock()        #timing
    #define grid points, [pc1, pc2]
    gsz = 0.1 #Grid size. Could be user input.
    xgridlow, ygridlow, xgridhigh, ygridhigh = concave_hull.bounds
    xgrids =    np.arange(xgridlow - gsz, xgridhigh, gsz)
    ygrids =    np.arange(ygridlow - gsz, ygridhigh, gsz)                    #use generator or numpy
    origin_grids = [[x , y] for x in xgrids for y in ygrids]    #origin grid points, could use numpy
    native_grids = [p for p in origin_grids if geometry.Point(p).within(concave_hull)]      #grid points only in the alpha shape
    e_grid = clock()        #timing
    print('Generating grid points takes : ', e_grid - s_grid)
    
    native_grids = manager.list(native_grids)   #make native_grids shared between processes
    # use kernel regression to estimate energy of grid points
    """
    The default kernel 'rbf' is gaussian kernel
    """
    s_krModel = clock()         #timing
    kernel = 'rbf'          #choose kernel function. More kernels see here  http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html
    kr = KernelRegression(kernel=kernel, bandwidth=3)
    kr_model = kr.fit(points, energy)   #build kernel regression modle from data points
    m_krModel = clock()         #timing
    print('Fitting model takes: ', m_krModel - s_krModel)
    
    grid_energy = kr_model.predict(np.array(native_grids))   #predict energy of grid points
    e_krModel = clock()
    print('Predicting grid energy takes(%s): ' %(kernel), e_krModel - m_krModel)
    #Now we finnaly get grid points and their energy value.
    print('Good to share grid energy?')
    grid_energy = manager.list(grid_energy)
    print('Good! grid energy shared!')
    #ge = open(r'C:\Users\Administrator\Desktop\grid_energy.txt','wb')
    #pickle.dump(grid_energy, ge)
    #ge.close()

    #load grid energy data from file. So we don't need to regenerate it every time
    #ge = open(r'C:\Users\Administrator\Desktop\grid_energy.txt', 'rb')
    #grid_energy = pickle.load(ge)
    #ge.close()
    
    
    #Generate some plot

    #This the the color map
    norm = mpl.colors.Normalize(vmin=-6600, vmax=-6250)
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
    #surf = ax.scatter(np_grids[:,0], np_grids[:,1], c=grid_energy, cmap=cm.jet)

    #fig.colorbar(surf, shrink=0.5, aspect=5)
    
    #Run it
    print('Okay to create basins list?')
    job_list = manager.Queue()
    basins = manager.list()
    Omega = manager.list()
    print('Now basin finder!!!')
    p = Pool(4)
    
    s_basin = clock()
    Basin_Finder(native_grids, -6400, Omega, 0.3, native_grids, grid_energy, m, job_list, basins)

    while not job_list.empty():
        k, S, c = job_list.get_nowait()
        p.apply_async(Basin_Finder, args=(S, c, Omega, 0.3, native_grids, grid_energy, m, job_list, basins))
        sleepTime = 0
        while job_list.empty() and sleepTime < 20:
            sleepTime += 1
            time.sleep(1)
            print('Sleep %d s ' % (sleepTime))
        print('are you empty? ', job_list.empty())       
    p.close()
    p.join()
    
    for basin in basins:
        ax.add_patch(basin)
        print('okay now I add one patch')
    print('how many basins? ', len(basins))
    f_basin = clock()
    print('Basin Finder takes(with multiprocessing) : ', f_basin - s_basin)
    fig.savefig(r'C:\Users\Administrator\Desktop\grid_energy_boundary_5.png')