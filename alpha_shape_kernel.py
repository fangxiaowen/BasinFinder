# -*- coding: utf-8 -*-
from basinfinder import alpha_shape, Basin_Finder
from descartes import PolygonPatch
from scipy.spatial import Delaunay
import numpy as np
#from statsmodels.nonparametric.kernel_regression import KernelReg
import shapely.geometry as geometry
from shapely.ops import cascaded_union, polygonize
import math
from kernel_regression import KernelRegression
#import rpy2.robjects as robjects
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib  import cm
import matplotlib as mpl
import pickle
from time import clock
from multiprocessing import Process, Manager, Array, Pool
import os
import time, threading
import sys
"""
Use SET to store points!!!!! instead of list!!!!!!!!!!

Notes on how to make it faster:
    1. Use parallel computing if possible. Including map/reduce and process and thread.
    2. Use async if possible
    3. Use functional programming / module from third party to reduce code
    4.
"""

#load data from the .txt file
protein_data = np.dtype([('col1', 'S20'), ('col2', 'S20'),('col3', 'S20'), ('energy', 'f4'), ('col5', 'f4'), ('col6', 'f4'),('col7', 'f4'),('col8', 'f4'),('col9', 'f4'),('col10', 'f4'),('col11', 'f4'),('col12', 'f4'),('col13', 'f4'),('col14', 'f4'),('col15', 'f4')])
points = np.loadtxt(r"C:\Users\Administrator\Desktop\bio\Ras_WT_SoPrimp_2016.txt", dtype = protein_data)           #load data
#Extrac value we care about
energy = np.array([e[3] for e in points])         # energy
points = np.array([[p[5], p[6]] for p in points])         #PC1 and PC2 we care about

#Draw the alpha shape from all data points
concave_hull, edge_points = alpha_shape(points, alpha=0.6)    #concave_hull contains all the polygons (all the boundary)

#define grid points, [pc1, pc2]
gsz = 0.1 #Grid size. Could be user input.
xgridlow, ygridlow, xgridhigh, ygridhigh = concave_hull.bounds
xgrids =    np.arange(xgridlow - gsz, xgridhigh, gsz)
ygrids =    np.arange(ygridlow - gsz, ygridhigh, gsz)                    #use generator or numpy
origin_grids = [[x , y] for x in xgrids for y in ygrids]    #origin grid points, could use numpy
native_grids = [p for p in origin_grids if geometry.Point(p).within(concave_hull)]      #grid points only in the alpha shape

# use kernel regression to estimate energy of grid points
"""
This method is from github. https://github.com/jmetzen/kernel_regression/blob/master/kernel_regression.py
The default kernel 'rbf' is gaussian kernel
"""
kr = KernelRegression(kernel="rbf", gamma=np.logspace(-2, 2, 10))
#kr_model = kr.fit(points, energy)   #build kernel regression modle from data points
#middle_kernel = clock()
#print('Fitting model takes: ', middle_kernel - start_kernel)

#model = open(r'C:\Users\Administrator\Desktop\kr_model.txt','wb')
#pickle.dump(kr_model, model)
#model.close()

#Just use the saved model to save time
model = open(r'C:\Users\Administrator\Desktop\kr_model.txt', 'rb')
kr_model = pickle.load(model)
model.close()

#Multiprocessing
"""
with Manager() as manager:
    total_grid_energy = manager.list(range(len(native_grids)))
    print("What's wrong with manager?")
    Omega = manager.list()
print('Multithread is working!')
"""
#Use multithread to speed up prediction of grid energy
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


#grid_energy = kr_model.predict(grids)   #predict energy of grid points

#Now we finnaly get grid points and their energy value.


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
#surf = ax.scatter(grids[:,0], grids[:,1], c=grid_energy, cmap=cm.jet)
ax.add_patch(patch)
#fig.colorbar(surf, shrink=0.5, aspect=5)


#Run it
Basin_Finder(native_grids, -6400, [], 0.3)




