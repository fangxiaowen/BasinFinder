# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:04:57 2017

@author: Xiaowen Fang
"""

__author__ = 'Xiaowen Fang'

from scipy.spatial import Delaunay
import numpy as np
import shapely.geometry as geometry
from shapely.ops import cascaded_union, polygonize
import math
#import matplotlib.pyplot as plt
from descartes import PolygonPatch
#from matplotlib  import cm
import sys
from multiprocessing import Process, Manager, Array, Pool
import multiprocessing

#This is also from http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/
def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        try:
            return geometry.MultiPoint(list(points)).convex_hull
        except BaseException as e:
            print('Fuck you! They are on the same line!', points)
            print('numpy.float64 is ', points)
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[[i,j]])
    coords = np.array(points)
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.simplices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        try:    #could have ZeroDivisionError and ValueError
            # Area of triangle by Heron's formula
            area = math.sqrt(s*(s-a)*(s-b)*(s-c))
            circum_r = a*b*c/(4.0*area)
            # Here's the radius filter.
            if circum_r < 1.0/alpha:
                add_edge(edges, edge_points, coords, ia, ib)
                add_edge(edges, edge_points, coords, ib, ic)
                add_edge(edges, edge_points, coords, ic, ia)
        except ZeroDivisionError as e:
            print("So points are? ", ia, ib, ic)
        except ValueError as e:
            print('a is %f, b is %f, c is %f, s is %f' %(a,b,c,s))
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    #return the union of polygons in this alpha shape
    return cascaded_union(triangles), edge_points


#The main basin finder algorithm
def Basin_Finder(S, c, Omega, delta2, native_grids, grid_energy, colormap, job_list, basins):
    """
    The main algorithm to find basins and saddles

    Parameters
    ----------
    S : set of grid points under certain energy level (c).
    c : highest energy level
    Omega : list of basins and saddles
    delta2 : step length c decreases
    """
    #global basins
    #global job_list
    #global ax
    if len(S) <= 3:    #Just want to end it earlier if less than 4 points
        print('I am returning', len(S), c)  #Debug message
        return
    try:
        bounds, edge_points = alpha_shape(S, alpha = 3)    #alpha shape
    except TypeError as e:
        print("What's wrong with S : ", S)
        sys.exit(0)
    k = 1 if not isinstance(bounds, geometry.multipolygon.MultiPolygon) else len(bounds)    #weather it's a single polygon or multiple polygons in the alpha shape

    #no basin splitting detected
    print('# of basin is ', k)   #Debug message
    if k == 1:
        if c > - 6500:
            c -= delta2 * 20    #make it decending faster
        else:
            c -= delta2
        # find points under updated energy level c
        S = [p for p in S if grid_energy[native_grids.index(p)] < c]      #points with energy less than c. Could be faster?
        print('Oh! We got S under c! Good!')
        job_list.put_nowait((k,S,c))

    #basin splitting detected
    else:
        #get points in different basins. Stored in S
        old_S = S[:]
        S = []
        for i in range(k):
            points_in_basin = [p for p in old_S if geometry.Point(p).within(bounds[i])]
            job_list.put_nowait((k,points_in_basin,c))
            old_S = [p for p in old_S if p not in points_in_basin]
            #print('How many points in each basin? ', len(x))
            print('We got points in a basin! Very good!')
            S.append(points_in_basin)     #append the points in a polygon
    
    print('We put stuff into jobList!VeryGood! So are you empty now? ', job_list.empty())
        #get saddles between neighbor basins
        #Could speed up by this way https://gis.stackexchange.com/questions/226085/fast-minimum-distance-between-two-sets-of-polygons/226143#226143

        #Pseudo code below. To be done
    """
        # distance_matriex is a n*n matrix : n is the size of bounds/basins
        distance_matrixes = [pairwise.pairwise_distances(X=S[i], Y=S[j], metric='euclidean', n_jobs=-1) for i in range(k) for j in range(k)]   #all distance matrixes
        old_distance_matrixes = np.array(distance_matrixes).reshape(k,k)       #could be wrong
        distance_matrixes = map(np.matrix.min(), distance_matrixes) #the shortest distance between 2 basins/bounds
        distance_matrixes = map(lambda x: -1 if (x > 20*0.6) or (x == 0) else x, distance_matrixes)
        try:
            distance_matrixes = np.array(distance_matrixes).reshape(k,k)    #reshape distance_matrixes to n*n matrix
        except TypeError as e:
            print('wtf???')
        #point_pairs = [np.where(old_distance_matrixes[i][j] == distance_matrixes[i][j]) for i in range(k) for j in range(k)].reshape(k.k)      #points in two polygons having shortest distance
        #saddles = map(lambda x, y: ((x[0]+y[0])/2, (x[1]+y[1])/2), [S[i][point_pairs[i][j][0]], S[j][point_pairs[i][j][1]] for i in S for j in S])

        #add saddles and basin bounds to Omega
        #Omega.append(saddles)
        """
    if not isinstance(bounds, geometry.multipolygon.MultiPolygon):  #No basin splitting detected
        print('ennergy level? ', c) #Debug
        print('how many basins so far? ', len(basins))  #debug
        print('Still not into Basin!')
        print('Points left:                ', len(S))
        #patch = PolygonPatch(bounds, fc=m.to_rgba(c), ec='#000000', fill=True, zorder=-1)  #Here, map c value to color code using color map
        #ax.add_patch(patch)
        #Basin_Finder(S, c, Omega, delta2, native_grids, grid_energy, colormap=colormap)
    
    else:       #Basin splitting detected
        for i in range(k):
            Omega.append(S[i])
            patch = PolygonPatch(bounds[i], fc=colormap.to_rgba(c), ec='#000000', fill=True, zorder=-1)  #Here, map c value to color code using color map
            basins.append(patch)
            print('Now we add a basin to basins! ', len(basins))
            #ax.add_patch(patch)     #Plot the boundaries of each basin
            print('how many basins? ', len(Omega))  #Debug
            print('ennergy level? ', c) #Debug
            #print('Now we go into a Basin!!!, # of points in this basin are:', len(S[i]))
            #p.apply_async(Basin_Finder, args=(S[i], c, Omega, delta2, native_grids, grid_energy, colormap))
            #Basin_Finder(S[i], c, Omega, delta2, native_grids, grid_energy, colormap, figure)
    
    #p.close()
    #p.join()

