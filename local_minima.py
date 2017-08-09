# -*- coding: utf-8 -*-
import scipy
from basinfinder import alpha_shape
from descartes import PolygonPatch
import numpy as np
import shapely.geometry as geometry
#from shapely.ops import cascaded_union, polygonize
from kernelregression import KernelRegression     #This is the kd_tree version written by Xiaowen Fang
import matplotlib.pyplot as plt
from matplotlib  import cm
import matplotlib as mpl
from time import clock
import sys
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances
import numdifftools as nd
import math
import pickle
import time


def find_saddle(x):
    """
    start from local extrema and find saddle point
    Parameters:
        - x : 2D point
    Returns:
        2D saddle point
    """
    
    sl = 0.1    #step length
    i=0
    try:
        while not ((abs(G(x)[0]) < 0.2) and (abs(G(x)[1]) < 0.2) and (np.linalg.eigvals(H(x)).prod() < 0)):     #stop critiaria. Saddle point condition. egenvalue1 * egenvalue2 < 0
        #for j in range(310): 
            i += 1
            if i > 100 or x[0] < -15 or x[0] > 20 or x[1] < -20 or x[1] > 15:
                return
            orthon = H(x)[0]    #orthonal to the searching direction    
            tagent = np.array([orthon[1], -orthon[0]])    #searching direction
                
            x_new = x + sl * normalize(tagent.reshape(1,-1))[0]
            #print("Gradient of new pont is : ",G(x_new))
            #print("\nEgienvalue isssssssssssssssssss : ",np.linalg.eigvals(H(x_new)))
            #time.sleep(1)
            trace.append(x_new)
            
            while not (abs(G(x_new)[0]) < 0.5):
                direc_derive = np.dot(H(x_new)[0], normalize(orthon.reshape(1,-1))[0])
                print('direc_derive is : ', direc_derive)
                #step length is soo tiny!
                x_new = x_new - 0.001 * direc_derive * normalize(orthon.reshape(1,-1))[0]
                print('X_new is : ', x_new)
                corrector.append(x_new)   
                print('Gradient of x_new is : ', G(x_new))
            print('corrector step is finished!\n')
            x = x_new
    except ValueError as e:
        print('Gradient and Hessian are : ', G(x),'\n', H(x))
        print('X_new is : ', x_new)
        print("x is : ", x)
        return
        #print("G(x) is : ", G(x), " H(x) is : ", H(x))
    
    return x

#trace = [x for x in trace if ((abs(G(x)[0] - 0) < 0.15) and (abs(G(x)[1]) < 0.1) and (np.linalg.eigvals(H(x)).prod() < 0))]

if __name__ == '__main__':
    #print('still okay here? Manager Object is created successfully!')
    
    #load data from the .txt file
    protein_data = np.dtype([('col1', 'S20'), ('col2', 'S20'),('col3', 'S20'), ('energy', 'f4'), ('col5', 'f4'), ('col6', 'f4'),('col7', 'f4'),('col8', 'f4'),('col9', 'f4'),('col10', 'f4'),('col11', 'f4'),('col12', 'f4'),('col13', 'f4'),('col14', 'f4'),('col15', 'f4')])
    sample_points = np.loadtxt(r"C:\Users\Administrator\Desktop\BasinFinder\Ras_WT_SoPrimp_2016.txt", dtype = protein_data)           #load data
    #Extrac value we care about
    energy = np.array([e[3] for e in sample_points])         # energy
    points = np.array([[p[5], p[6], i] for i, p in enumerate(sample_points)])
    e_loadData = clock()    #timing
    print('data loaded')
    #Draw the alpha shape from all data points
    #concave_hull, edge_points = alpha_shape(np.delete(points, 2, 1), alpha=0.6)    #concave_hull contains all the polygons (all the boundary)
    concave_hull = pickle.load(open(r'C:\Users\Administrator\Desktop\concave_hull.pickle', 'rb'))
    print('alpha shape builded')    
    #alpha_shape = open(r'C:\Users\Administrator\Desktop\concave_hull.pickle', 'wb')
    #pickle.dump(concave_hull,alpha_shape)    
    
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
    
    H = nd.Hessian(f)   #Hessian 
    G = nd.Gradient(f)  #Gradient
    print(H)
    #sys.exit(0)
    
    
    
    """
    s_newton = clock()
    local_para, local_minima = [], []
    
    for i in np.delete(native_grids, 2, 1):
            x = scipy.optimize.fmin_ncg(f, np.array(i), fprime)
            local_para.append(x)
            #local_minima.append(x[1])
            print(x)
    f_newton = clock()
    print('time to find local minima is : ', f_newton - s_newton)
    """
    
    #local = open(r'C:\Users\Administrator\Desktop\local_para.pickle', 'wb')
    #pickle.dump(local_para,local)    
    local_para = pickle.load(open(r'C:\Users\Administrator\Desktop\local_para.pickle', 'rb'))
    #local_para=local_para[~np.triu((pairwise_distances(local_para) < 0.1),1).any(0)]  #remove duplicates
    
    li = []
    trace, corrector = [], []
    for i in local_para:
        if not math.isnan(i[0]) and abs(i[0]) < 25 and abs(i[1]) < 25:
            li.append(find_saddle(np.array(i)))
    #li = pickle.load(open(r'C:\Users\Administrator\Desktop\saddles.pickle', 'rb'))     #list of saddle points
    
    all_trace = []
    all_trace.append(trace)
    sys.exit(0)
    #saddle = open(r'C:\Users\Administrator\Desktop\saddles.pickle', 'wb')
    #pickle.dump(li,saddle)
    #tr = open(r'C:\Users\Administrator\Desktop\trace1_0.4_direction2.pickle', 'wb')
    #pickle.dump(trace,tr)
    
    fig = pickle.load(open(r'C:\Users\Administrator\Desktop\trace1_color.pickle', 'rb'))
    #trace = pickle.load(open(r'C:\Users\Administrator\Desktop\trace1_0.4.pickle', 'rb'))
    
    
    
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
    #local_para = np.array(local_para)
    ax.add_patch(patch)
    

    ax = fig.add_subplot(111)
    surf = ax.scatter(np.array(local_para)[:,0], np.array(local_para)[:,1], color = 'yellow')

    surf2 = ax.scatter(np.array(trace)[:,0], np.array(trace)[:,1], s=0.5, color='blue')
    surf3 = ax.scatter(np.array(li)[:,0], np.array(li)[:,1], color='black')
    ge = open(r'C:\Users\Administrator\Desktop\saddle_points.pickle', 'wb')
    pickle.dump(fig,ge)    
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(r'C:\Users\Administrator\Desktop\saddle_points' + kernel + '_.png')