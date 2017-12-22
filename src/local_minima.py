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
from matplotlib import _cntr as cntr


intersect_point_x1, intersect_point_x2 = 0,0
# accurate saddles are stored as exact_saddles.pickle
def find_exact_saddle(x):
    """
    start from approximate saddle point and find it's conresponding accurate saddle point
    Parameters:
        - x : 2D point. A approximate saddle point
    Returns:
        2D saddle point
    """
    slt = 0.01
    i = 0
    print('Origin gradient is : ', G(x))
    while not ((abs(G(x)[0]) < 0.1) and (abs(G(x)[1]) < 0.1) and (np.linalg.eigvals(H(x)).prod() < 0)) and i < 20:
        i += 1
        x_corrector = -H(x)[0][0] if G(x)[0] > 0 else H(x)[0][0]
        y_corrector = -H(x)[1][1] if G(x)[1] > 0 else H(x)[1][1]
        corrector = np.array([x_corrector, y_corrector])  #fxx and fyy
        x = x + slt * normalize(corrector.reshape(1,-1))[0] # So we can get a more accurate saddle point
    
        print('Gradient of x is : ', G(x))
    return x


trace_x1, trace_x2 = [], []
def find_contour(x):
    """
    Start from saddle and find contour passing the saddle
    For more information, please contact Prof Wanli or Xiaowen Fang
    
    Parameters:
        - x : 2D point. Saddle point
    Returns:
        2D saddle point
    """
    global intersect_point_x1
    global intersect_point_x2
    print('X is : ', x)
    origin_energy = kr_model.predict_single(x)  # The energy level at saddle point
    #origin_energy = -6562
    print('Origin energy is : ', origin_energy)
    egienvalue = np.linalg.eig(H(x))[0]             
    egienvector = np.linalg.eig(H(x))[1][0] if egienvalue[0] < 0 else np.linalg.eig(H(x))[1][1]  # the egienvector we want corresponding to negtive egienvalue
    x1, x2 = x + 0.5*egienvector, x - 0.5*egienvector     # x1, x2 are points deviate a little from the origin saddle point. We want to draw contour of basin from x1 and x2.
    
    intersect_point_x1, intersect_point_x2 = x1, x2       # when we start from x1 and follow some trace to construct contour, it should overlap with the starting point when finishing the contour.
    inter_x1_energy, inter_x2_energy = kr_model.predict_single(intersec_point_x1), kr_model.predict_single(intersec_point_x2)
    slt = 0.1
    i = 0
    #trace_x1, trace_x2 = [], []
    """
    while(not (kr_model.predict_single(x1) - origin_energy) < 1): 
        #gradient desending
        if kr_model.predict_single(x1) > origin_energy:
            x1 -= slt * normalize(G(x1).reshape(1,-1))[0]
        else:
            x1 += slt * normalize(G(x1).reshape(1,-1))[0]
        print('Now you should be closer to origin energy : ', kr_model.predict_single(x1))
    """
    while(i < 1500):        # you can modify the value of i.
        i += 1
        tagent = np.array([G(x1)[1], -G(x1)[0]])    #tagent is perpendicular/orthonal to gradient
        x1 = x1 + slt * normalize(tagent.reshape(1,-1))[0]      # follow the tagent to construct contour
        while(not (kr_model.predict_single(x1) - inter_x1_energy) < 0.5):   # If it deviates from contour
        #gradient desending to return to desired energy level
            if kr_model.predict_single(x1) > inter_x1_energy:
                x1 -= slt * normalize(G(x1).reshape(1,-1))[0]
            else:
                x1 += slt * normalize(G(x1).reshape(1,-1))[0]
            #print('Now you should be closer to origin energy : ', kr_model.predict_single(x1))
        trace_x1.append(x1)
        #print('Cooool! Add a point in contour : ', x1)
        if math.hypot(x1[0] - intersect_point_x1[0],x1[1] - intersect_point_x1[1]) < 0.1:   # If we are close enought to the starting point, we assume we finish constructing contour
            if i > 10:      # do not break at first.
                break
    """
    #This is for the other contour line
    while(not (kr_model.predict_single(x2) - origin_energy) < 1): 
        #gradient desending
        if kr_model.predict_single(x2) > origin_energy:
            x2 -= slt * normalize(G(x2).reshape(1,-1))[0]
        else:
            x2 += slt * normalize(G(x2).reshape(1,-1))[0]
        print('Now you should be closer to origin energy : ', kr_model.predict_single(x2))
    """
    i = 0
    while(i < 1500):     #  might go back to the first point!!
        i += 1
        tagent = np.array([G(x2)[1], -G(x2)[0]])    #orthonal to gradient
        x2 = x2 + slt * normalize(tagent.reshape(1,-1))[0]
        while(not (kr_model.predict_single(x2) - inter_x2_energy) < 0.5):
        #gradient desending
            if kr_model.predict_single(x2) > inter_x2_energy:
                x2 -= slt * normalize(G(x2).reshape(1,-1))[0]
            else:
                x2 += slt * normalize(G(x2).reshape(1,-1))[0]
            #print('Now you should be closer to origin energy : ', kr_model.predict_single(x2))
        trace_x2.append(x2)
        #print('Cooool! Add a point in contour : ', x2)
        if math.hypot(x2[0] - intersect_point_x2[0],x2[1] - intersect_point_x2[1]) < 0.1:
            if i > 10:
                break
    
    


def find_saddle(x):
    """
    Start from local extrema(minima) and find saddle point.
    For details of how this works, please contact Prof Wanli
    
    Parameters:
        - x : 2D point. Should input local extrema(minima)
    Returns:
        2D saddle point. The saddle point
    """
    
    sl = 0.1    #step length
    i=0         # numbers of iteration
    try:
        while not ((abs(G(x)[0]) < 0.2) and (abs(G(x)[1]) < 0.2) and (np.linalg.eigvals(H(x)).prod() < 0)):     #stop critiaria. Saddle point condition : Hessian matrix's egenvalue1 * egenvalue2 < 0  and gradient is 0 (or say, close enought to 0).
        #for j in range(310): 
            i += 1
            if i > 100 or x[0] < -15 or x[0] > 20 or x[1] < -20 or x[1] > 15:       # If we have searched long enough or we have exceed the limit of the alpha shape, we want to exit. You can modify i to implicate how long you want to search.
                return
            orthon = H(x)[0]    # The first column of Hessian matrix is orthonal/perpendicular to the searching direction we want 
            tagent = np.array([orthon[1], -orthon[0]])    #searching direction. Orthonal/Perpendicular to orthon (the first column of Hessian matrix)
                
            x_new = x + sl * normalize(tagent.reshape(1,-1))[0]     # follow the searching direction.
            #print('X_NEW is : ', x_new)
            #print("Gradient of new pont is : ",G(x_new))
            #print("\nEgienvalue isssssssssssssssssss : ",np.linalg.eigvals(H(x_new)))
            #time.sleep(1)
            trace.append(x_new)     # add it to trace, so we can plot the trace.
            """
            while not (abs(G(x_new)[0]) < 0.5):
                direc_derive = np.dot(H(x_new)[0], normalize(orthon.reshape(1,-1))[0])
                #print('direc_derive is : ', direc_derive)
                x_old = x_new
                #step length is soo tiny!
                x_new = x_new - 0.001 * direc_derive * normalize(orthon.reshape(1,-1))[0]
                if abs(G(x_new)[0]) > abs(G(x_old)[0]):
                    x_new = x_old
                    break
                print('X_new is : ', x_new)
                corrector.append(x_new)   
                print('Gradient of x_new is : ', G(x_new))
            print('corrector step is finished!\n')
            """
            x = x_new
    except ValueError as e:
        print('Gradient and Hessian are : ', G(x),'\n', H(x))
        print('X_new is : ', x_new)
        print("x is : ", x)
        return
        #print("G(x) is : ", G(x), " H(x) is : ", H(x))   
    return x

#def corrector_step(x):
    
#trace = [x for x in trace if ((abs(G(x)[0] - 0) < 0.2) and (abs(G(x)[1]) < 0.2) and (np.linalg.eigvals(H(x)).prod() < 0))]

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
    gsz = 1 #Grid size. Could be user input. Note : Grid size is 1. We start from grid points, using gradient descending to find local minima 
    xgridlow, ygridlow, xgridhigh, ygridhigh = concave_hull.bounds
    xgrids =    np.arange(xgridlow - gsz, xgridhigh, gsz)
    ygrids =    np.arange(ygridlow - gsz, ygridhigh, gsz)                    #use generator or numpy
    origin_grids = [[x , y] for x in xgrids for y in ygrids]    #origin grid points, could use numpy
    native_grids = np.array([p for p in origin_grids if geometry.Point(p).within(concave_hull)])      #grid points only in the alpha shape
    native_grids = np.array([[p[0], p[1], i] for i,p in enumerate(native_grids)])
    print('grid points generated')
    # use kernel regression to estimate energy of grid points

    kernel = 'gaussian'          #choose kernel function. More kernels see here  http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html
    try:
        kr = KernelRegression(kernel=kernel, bandwidth=0.7, radius=3)
        #sys.exit(1)
    except BaseException as e:
        print("What's wrong with kernel reg?")
    kr_model = kr.fit(np.delete(points, 2, 1), energy)   #fit kernel regression modle from data points

    f, fprime = kr_model.predict_single, kr_model.fprime    # kr_model.fprime is the gradient of kr_model.predict_single
    
    
    # Use numerical method instead the formula above
    H = nd.Hessian(f)   #Hessian of kr_model.predict_single
    G = nd.Gradient(f)  #Gradient of kr_model.predict_single
    print(H)
    #time.sleep(10)

    sys.exit(0)
    


    
    
    s_newton = clock()  #Timing
    local_minima = []
    
    # Start from grid points, use newton-method to find local minima
    for i in np.delete(native_grids, 2, 1):
            x = scipy.optimize.fmin_ncg(f, np.array(i), fprime) # This is the scipy module to find local minima
            local_minima.append(x)
            print(x)
    f_newton = clock()  # Timing
    print('time to find local minima is : ', f_newton - s_newton)   # Print time used
    
    
    local = open(r'C:\Users\Administrator\Desktop\local_minima.pickle', 'wb')
    pickle.dump(local_para,local)    
    #local_minima = pickle.load(open(r'C:\Users\Administrator\Desktop\local_minima.pickle', 'rb'))      # Once you store it on the disk, you can load it next time instead of recomputing it again.
    local_minima=local_minima[~np.triu((pairwise_distances(local_minima) < 0.1),1).any(0)]  #remove duplicates. We might get duplicate local minimas, so we want to remove duplicate ones.
    
    
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
    
    
    
    # Draw some plot to see weather we are correct. You don't need to worry about this.
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
    
    
    
    
    
    
    
    
    """
    origin_grids = pickle.load(open(r'C:\Users\Administrator\Desktop\origin_grid.pickle','rb'))
    c = cntr.Cntr(origin_grids[:,0].reshape(390,414), origin_grids[:,1].reshape(390,414), origin_grids[:,3].reshape(390,414))
    res = c.trace(-6500)
    # result is a list of arrays of vertices and path codes
    # (see docs for matplotlib.path.Path)
    nseg = len(res) // 2
    segments, codes = res[:nseg], res[nseg:]

    p = plt.Polygon(segments[0], fill=False, color='w')
    ax.add_artist(p)
    
    
    
    
    
    ax.scatter(np.array(trace_x1[:,0]), np.array(trace_x1[:,1]))
    """
